//! A single-file translation of NoGraphicsAPI's Vulkan backend.
//!
//! The device owns Vulkan allocations and releases them on drop. Resource values contain
//! handles and metadata, with no device back-references. Explicit destroy operations consume
//! resources to release them early. Command buffers move out of the device's pool and back
//! into it on submission. Windows, timeline points, and attachment views use `Arc` for sharing.
//! Mutable operations require exclusive borrows; share a device across threads with
//! `Arc<Mutex<Device>>`. Window presentation must still follow winit's platform requirements.
//!
//! # Safety
//! Every resource passed to an unsafe operation must belong to that live device. Keep GPU
//! allocations alive until recorded and executing work has finished. Submit every begun
//! command buffer in the next batch, with the first begun buffer first. Swapchain views are
//! valid only for their acquired frame. Mapped memory access, Vulkan synchronization, ranges,
//! alignments, and shader interfaces remain the caller's responsibility. `Arc` extends the
//! lifetime of Rust metadata; allocations are owned by the device and freed when it drops.
#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use ash::vk::TaggedStructure as _;
use std::{
    collections::VecDeque,
    ffi::{CStr, c_void},
    mem::size_of,
    ptr,
    sync::Arc,
};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

const MAX_COLOR_ATTACHMENTS: usize = 8;
const IMAGE_BARRIER_BATCH_SIZE: usize = 64;
const INITIAL_COMMAND_CONTEXT_COUNT: usize = 2;
const MAX_SWAPCHAIN_IMAGES: usize = 8;
const GPU_ALLOCATION_ALIGNMENT: u64 = 16;
const SWAPCHAIN_PRESENT_MODE: vk::PresentModeKHR = vk::PresentModeKHR::FIFO;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Unsupported,
    DeviceLost,
    DriverError,
}

impl From<vk::Result> for Error {
    fn from(result: vk::Result) -> Self {
        match result {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY
            | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            | vk::Result::ERROR_TOO_MANY_OBJECTS => std::process::abort(),
            vk::Result::ERROR_DEVICE_LOST => Self::DeviceLost,
            vk::Result::ERROR_LAYER_NOT_PRESENT
            | vk::Result::ERROR_EXTENSION_NOT_PRESENT
            | vk::Result::ERROR_FEATURE_NOT_PRESENT
            | vk::Result::ERROR_INCOMPATIBLE_DRIVER
            | vk::Result::ERROR_FORMAT_NOT_SUPPORTED => Self::Unsupported,
            _ => Self::DriverError,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unsupported => "unsupported graphics device or configuration",
            Self::DeviceLost => "graphics device lost",
            Self::DriverError => "Vulkan driver error",
        })
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn require<T>(result: std::result::Result<T, vk::Result>) -> T {
    result.unwrap_or_else(|error| {
        eprintln!("NoGraphicsAPI: unexpected Vulkan failure: {error:?}");
        std::process::abort()
    })
}

fn require_error<T>(result: Result<T>) -> T {
    result.unwrap_or_else(|error| {
        eprintln!("NoGraphicsAPI: {error}");
        std::process::abort()
    })
}

fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert_ne!(alignment, 0);
    value.div_ceil(alignment) * alignment
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MemoryType {
    CpuVisible,
    GpuOnly,
    Readback,
    TextureDescriptorHeap,
    SamplerDescriptorHeap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Format {
    R8Srgb,
    Rg8Srgb,
    Rgba8Srgb,
    Bgra8Srgb,
    Rgba4Unorm,
    R5g5b5a1Unorm,
    R5g6b5Unorm,
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    R16Unorm,
    Rg16Unorm,
    Rgba16Unorm,
    R8Uint,
    Rg8Uint,
    Rgba8Uint,
    Bgra8Uint,
    R16Uint,
    Rg16Uint,
    Rgba16Uint,
    R32Uint,
    Rg32Uint,
    Rgb32Uint,
    Rgba32Uint,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgb32Float,
    Rgba32Float,
    Rgb10a2Unorm,
    Rg11b10Float,
    D16Unorm,
    D24UnormS8Uint,
    D32Float,
    S8Uint,
    D32FloatS8Uint,
    EacRg,
    Astc4x4Srgb,
    Astc4x4Unorm,
    Bc3Srgb,
    Bc3Unorm,
    Bc5Rg,
    Bc7Srgb,
    Bc7Unorm,
    Undefined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TextureType {
    OneD,
    TwoD,
    ThreeD,
    Cube,
    TwoDArray,
    CubeArray,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureUsage(pub u32);

impl TextureUsage {
    pub const NONE: Self = Self(0);
    pub const SAMPLED: Self = Self(1 << 0);
    pub const STORAGE: Self = Self(1 << 1);
    pub const COLOR_ATTACHMENT: Self = Self(1 << 2);
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(1 << 3);
    pub const TRANSFER_SOURCE: Self = Self(1 << 4);
    pub const TRANSFER_DESTINATION: Self = Self(1 << 5);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for TextureUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for TextureUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TextureDescriptorType {
    Sampled,
    Storage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TextureAspect {
    Automatic,
    Color,
    Depth,
    Stencil,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CullMode {
    None,
    Clockwise,
    CounterClockwise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BlendFactor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    DestinationColor,
    OneMinusDestinationColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    SourceAlphaSaturate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Minimum,
    Maximum,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IndexType {
    Uint16,
    Uint32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LoadOp {
    Load,
    Clear,
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StoreOp {
    Store,
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stage(pub u64);

impl Stage {
    pub const NONE: Self = Self(0);
    pub const INDIRECT: Self = Self(1 << 6);
    pub const INDEX_INPUT: Self = Self(1 << 7);
    pub const VERTEX: Self = Self(1 << 1);
    pub const MESH: Self = Self(1 << 9);
    pub const DEPTH_STENCIL_TESTS: Self = Self(1 << 8);
    pub const FRAGMENT: Self = Self(1 << 2);
    pub const COLOR_OUTPUT: Self = Self(1 << 4);
    pub const COMPUTE: Self = Self(1 << 3);
    pub const TRANSFER: Self = Self(1 << 0);
    pub const HOST: Self = Self(1 << 5);
    pub const ALL_COMMANDS: Self = Self(1 << 10);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Stage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Stage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Access(pub u64);

impl Access {
    pub const NONE: Self = Self(0);
    pub const TRANSFER_READ: Self = Self(1 << 0);
    pub const TRANSFER_WRITE: Self = Self(1 << 1);
    pub const SHADER_READ: Self = Self(1 << 2);
    pub const SHADER_WRITE: Self = Self(1 << 3);
    pub const COLOR_READ: Self = Self(1 << 4);
    pub const COLOR_WRITE: Self = Self(1 << 5);
    pub const DEPTH_STENCIL_READ: Self = Self(1 << 6);
    pub const DEPTH_STENCIL_WRITE: Self = Self(1 << 7);
    pub const INDIRECT_READ: Self = Self(1 << 8);
    pub const INDEX_READ: Self = Self(1 << 9);
    pub const HOST_READ: Self = Self(1 << 10);
    pub const DESCRIPTOR_READ: Self = Self(1 << 11);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Access {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Access {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

const FORMAT_COUNT: usize = Format::Undefined as usize;
const FORMATS: [Format; FORMAT_COUNT] = [
    Format::R8Srgb,
    Format::Rg8Srgb,
    Format::Rgba8Srgb,
    Format::Bgra8Srgb,
    Format::Rgba4Unorm,
    Format::R5g5b5a1Unorm,
    Format::R5g6b5Unorm,
    Format::R8Unorm,
    Format::Rg8Unorm,
    Format::Rgba8Unorm,
    Format::Bgra8Unorm,
    Format::R16Unorm,
    Format::Rg16Unorm,
    Format::Rgba16Unorm,
    Format::R8Uint,
    Format::Rg8Uint,
    Format::Rgba8Uint,
    Format::Bgra8Uint,
    Format::R16Uint,
    Format::Rg16Uint,
    Format::Rgba16Uint,
    Format::R32Uint,
    Format::Rg32Uint,
    Format::Rgb32Uint,
    Format::Rgba32Uint,
    Format::R16Float,
    Format::Rg16Float,
    Format::Rgba16Float,
    Format::R32Float,
    Format::Rg32Float,
    Format::Rgb32Float,
    Format::Rgba32Float,
    Format::Rgb10a2Unorm,
    Format::Rg11b10Float,
    Format::D16Unorm,
    Format::D24UnormS8Uint,
    Format::D32Float,
    Format::S8Uint,
    Format::D32FloatS8Uint,
    Format::EacRg,
    Format::Astc4x4Srgb,
    Format::Astc4x4Unorm,
    Format::Bc3Srgb,
    Format::Bc3Unorm,
    Format::Bc5Rg,
    Format::Bc7Srgb,
    Format::Bc7Unorm,
];

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Uint32x2 {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Uint32x3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct SizeAlign {
    pub size: u64,
    pub align: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TextureFormatInfo {
    pub block_extent: Uint32x2,
    pub bytes_per_block: u32,
    pub depth: bool,
    pub stencil: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct TextureDesc {
    pub r#type: TextureType,
    pub extent: Uint32x3,
    pub mip_levels: u32,
    pub layer_count: u32,
    pub format: Format,
    pub mutable_format: bool,
    pub usage: TextureUsage,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            r#type: TextureType::TwoD,
            extent: Uint32x3 { x: 1, y: 1, z: 1 },
            mip_levels: 1,
            layer_count: 1,
            format: Format::Rgba8Unorm,
            mutable_format: false,
            usage: TextureUsage::SAMPLED,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderViewDesc {
    pub mip_level: u32,
    pub slice: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct TextureDescriptorDesc {
    pub format: Format,
    pub aspect: TextureAspect,
    pub base_mip: u32,
    pub mip_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}

impl Default for TextureDescriptorDesc {
    fn default() -> Self {
        Self {
            format: Format::Undefined,
            aspect: TextureAspect::Automatic,
            base_mip: 0,
            mip_count: 0,
            base_layer: 0,
            layer_count: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TextureCopyDesc {
    pub mip_level: u32,
    pub base_slice: u32,
    pub slice_count: u32,
    pub offset: Uint32x3,
    pub extent: Uint32x3,
    pub row_pitch_bytes: u64,
    pub slice_pitch_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct SamplerDesc {
    pub min_filter: Filter,
    pub mag_filter: Filter,
    pub mip_filter: Filter,
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,
    pub anisotropic: bool,
    pub compare_enabled: bool,
    pub compare: CompareOp,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            min_filter: Filter::Linear,
            mag_filter: Filter::Linear,
            mip_filter: Filter::Linear,
            address_u: AddressMode::Repeat,
            address_v: AddressMode::Repeat,
            address_w: AddressMode::Repeat,
            anisotropic: false,
            compare_enabled: false,
            compare: CompareOp::LessEqual,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlendComponentState {
    pub source: BlendFactor,
    pub destination: BlendFactor,
    pub operation: BlendOp,
}

impl Default for BlendComponentState {
    fn default() -> Self {
        Self {
            source: BlendFactor::One,
            destination: BlendFactor::Zero,
            operation: BlendOp::Add,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BlendState {
    pub enabled: bool,
    pub color: BlendComponentState,
    pub alpha: BlendComponentState,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorTargetDesc {
    pub format: Format,
    pub blend: BlendState,
    pub write_mask: u8,
}

impl Default for ColorTargetDesc {
    fn default() -> Self {
        Self {
            format: Format::Undefined,
            blend: Default::default(),
            write_mask: 0xf,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RasterizationState {
    pub cull: CullMode,
    pub depth_bias_constant: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope: f32,
}

impl Default for RasterizationState {
    fn default() -> Self {
        Self {
            cull: CullMode::None,
            depth_bias_constant: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Scissor {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Default for Scissor {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StencilFaceState {
    pub compare: CompareOp,
    pub fail: StencilOp,
    pub pass: StencilOp,
    pub depth_fail: StencilOp,
    pub reference: u8,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            compare: CompareOp::Always,
            fail: StencilOp::Keep,
            pass: StencilOp::Keep,
            depth_fail: StencilOp::Keep,
            reference: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DepthStencilState {
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: CompareOp,
    pub stencil_test: bool,
    pub stencil_read_mask: u8,
    pub stencil_write_mask: u8,
    pub front: StencilFaceState,
    pub back: StencilFaceState,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_test: false,
            depth_write: false,
            depth_compare: CompareOp::LessEqual,
            stencil_test: false,
            stencil_read_mask: 0xff,
            stencil_write_mask: 0xff,
            front: Default::default(),
            back: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GraphicsPSODesc<'a> {
    pub vertex_spirv: &'a [u32],
    pub fragment_spirv: &'a [u32],
    pub color_targets: &'a [ColorTargetDesc],
    pub depth_format: Format,
    pub stencil_format: Format,
    pub rasterization: RasterizationState,
}

impl Default for GraphicsPSODesc<'_> {
    fn default() -> Self {
        Self {
            vertex_spirv: &[],
            fragment_spirv: &[],
            color_targets: &[],
            depth_format: Format::Undefined,
            stencil_format: Format::Undefined,
            rasterization: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeshPSODesc<'a> {
    pub mesh_spirv: &'a [u32],
    pub fragment_spirv: &'a [u32],
    pub color_targets: &'a [ColorTargetDesc],
    pub depth_format: Format,
    pub stencil_format: Format,
    pub rasterization: RasterizationState,
}

impl Default for MeshPSODesc<'_> {
    fn default() -> Self {
        Self {
            mesh_spirv: &[],
            fragment_spirv: &[],
            color_targets: &[],
            depth_format: Format::Undefined,
            stencil_format: Format::Undefined,
            rasterization: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ClearColor {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for ClearColor {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColorAttachment {
    pub render_view: Option<Arc<RenderView>>,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: ClearColor,
}

impl Default for ColorAttachment {
    fn default() -> Self {
        Self {
            render_view: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
            clear: Default::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DepthAttachment {
    pub render_view: Option<Arc<RenderView>>,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: f32,
}

impl Default for DepthAttachment {
    fn default() -> Self {
        Self {
            render_view: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
            clear: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StencilAttachment {
    pub render_view: Option<Arc<RenderView>>,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: u8,
}

impl Default for StencilAttachment {
    fn default() -> Self {
        Self {
            render_view: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
            clear: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RenderingDesc<'a> {
    pub colors: &'a [ColorAttachment],
    pub depth: DepthAttachment,
    pub stencil: StencilAttachment,
}

impl Format {
    fn vk(self) -> vk::Format {
        match self {
            Self::R8Srgb => vk::Format::R8_SRGB,
            Self::Rg8Srgb => vk::Format::R8G8_SRGB,
            Self::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            Self::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
            Self::Rgba4Unorm => vk::Format::R4G4B4A4_UNORM_PACK16,
            Self::R5g5b5a1Unorm => vk::Format::R5G5B5A1_UNORM_PACK16,
            Self::R5g6b5Unorm => vk::Format::R5G6B5_UNORM_PACK16,
            Self::R8Unorm => vk::Format::R8_UNORM,
            Self::Rg8Unorm => vk::Format::R8G8_UNORM,
            Self::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            Self::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            Self::R16Unorm => vk::Format::R16_UNORM,
            Self::Rg16Unorm => vk::Format::R16G16_UNORM,
            Self::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
            Self::R8Uint => vk::Format::R8_UINT,
            Self::Rg8Uint => vk::Format::R8G8_UINT,
            Self::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
            Self::Bgra8Uint => vk::Format::B8G8R8A8_UINT,
            Self::R16Uint => vk::Format::R16_UINT,
            Self::Rg16Uint => vk::Format::R16G16_UINT,
            Self::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
            Self::R32Uint => vk::Format::R32_UINT,
            Self::Rg32Uint => vk::Format::R32G32_UINT,
            Self::Rgb32Uint => vk::Format::R32G32B32_UINT,
            Self::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
            Self::R16Float => vk::Format::R16_SFLOAT,
            Self::Rg16Float => vk::Format::R16G16_SFLOAT,
            Self::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            Self::R32Float => vk::Format::R32_SFLOAT,
            Self::Rg32Float => vk::Format::R32G32_SFLOAT,
            Self::Rgb32Float => vk::Format::R32G32B32_SFLOAT,
            Self::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            Self::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
            Self::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
            Self::D16Unorm => vk::Format::D16_UNORM,
            Self::D24UnormS8Uint => vk::Format::D24_UNORM_S8_UINT,
            Self::D32Float => vk::Format::D32_SFLOAT,
            Self::S8Uint => vk::Format::S8_UINT,
            Self::D32FloatS8Uint => vk::Format::D32_SFLOAT_S8_UINT,
            Self::EacRg => vk::Format::EAC_R11G11_UNORM_BLOCK,
            Self::Astc4x4Srgb => vk::Format::ASTC_4X4_SRGB_BLOCK,
            Self::Astc4x4Unorm => vk::Format::ASTC_4X4_UNORM_BLOCK,
            Self::Bc3Srgb => vk::Format::BC3_SRGB_BLOCK,
            Self::Bc3Unorm => vk::Format::BC3_UNORM_BLOCK,
            Self::Bc5Rg => vk::Format::BC5_UNORM_BLOCK,
            Self::Bc7Srgb => vk::Format::BC7_SRGB_BLOCK,
            Self::Bc7Unorm => vk::Format::BC7_UNORM_BLOCK,
            Self::Undefined => vk::Format::UNDEFINED,
        }
    }
}

impl BlendFactor {
    fn vk(self) -> vk::BlendFactor {
        match self {
            Self::Zero => vk::BlendFactor::ZERO,
            Self::One => vk::BlendFactor::ONE,
            Self::SourceColor => vk::BlendFactor::SRC_COLOR,
            Self::OneMinusSourceColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            Self::DestinationColor => vk::BlendFactor::DST_COLOR,
            Self::OneMinusDestinationColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            Self::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
            Self::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            Self::DestinationAlpha => vk::BlendFactor::DST_ALPHA,
            Self::OneMinusDestinationAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            Self::SourceAlphaSaturate => vk::BlendFactor::SRC_ALPHA_SATURATE,
        }
    }
}

impl Stage {
    fn vk(self) -> vk::PipelineStageFlags2 {
        let mut result = vk::PipelineStageFlags2::empty();
        if self.intersects(Self::INDIRECT) {
            result |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        }
        if self.intersects(Self::INDEX_INPUT) {
            result |= vk::PipelineStageFlags2::INDEX_INPUT;
        }
        if self.intersects(Self::VERTEX) {
            result |= vk::PipelineStageFlags2::VERTEX_SHADER;
        }
        if self.intersects(Self::MESH) {
            result |= vk::PipelineStageFlags2::MESH_SHADER_EXT;
        }
        if self.intersects(Self::DEPTH_STENCIL_TESTS) {
            result |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        }
        if self.intersects(Self::FRAGMENT) {
            result |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if self.intersects(Self::COLOR_OUTPUT) {
            result |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        }
        if self.intersects(Self::COMPUTE) {
            result |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        if self.intersects(Self::TRANSFER) {
            result |= vk::PipelineStageFlags2::COPY;
        }
        if self.intersects(Self::HOST) {
            result |= vk::PipelineStageFlags2::HOST;
        }
        if self.intersects(Self::ALL_COMMANDS) {
            result |= vk::PipelineStageFlags2::ALL_COMMANDS;
        }
        result
    }
}

impl Access {
    fn vk(self) -> vk::AccessFlags2 {
        let mut result = vk::AccessFlags2::empty();
        if self.intersects(Self::TRANSFER_READ) {
            result |= vk::AccessFlags2::TRANSFER_READ;
        }
        if self.intersects(Self::TRANSFER_WRITE) {
            result |= vk::AccessFlags2::TRANSFER_WRITE;
        }
        if self.intersects(Self::SHADER_READ) {
            result |= vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_SAMPLED_READ;
        }
        if self.intersects(Self::SHADER_WRITE) {
            result |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
        }
        if self.intersects(Self::COLOR_READ) {
            result |= vk::AccessFlags2::COLOR_ATTACHMENT_READ;
        }
        if self.intersects(Self::COLOR_WRITE) {
            result |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        }
        if self.intersects(Self::DEPTH_STENCIL_READ) {
            result |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        }
        if self.intersects(Self::DEPTH_STENCIL_WRITE) {
            result |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
        if self.intersects(Self::INDIRECT_READ) {
            result |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
        }
        if self.intersects(Self::INDEX_READ) {
            result |= vk::AccessFlags2::INDEX_READ;
        }
        if self.intersects(Self::HOST_READ) {
            result |= vk::AccessFlags2::HOST_READ;
        }
        if self.intersects(Self::DESCRIPTOR_READ) {
            result |=
                vk::AccessFlags2::SAMPLER_HEAP_READ_EXT | vk::AccessFlags2::RESOURCE_HEAP_READ_EXT;
        }
        result
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct GpuRange {
    pub gpu: u64,
    pub size: u64,
}

/// An allocation owned by the device. Destroy it explicitly to release it early.
pub struct GpuHeap {
    range: GpuRange,
    backing: BackingBuffer,
    mapped_offset: usize,
}

impl GpuHeap {
    /// Returns the usable GPU address range, excluding alignment padding and reserved bytes.
    pub fn range(&self) -> GpuRange {
        self.range
    }

    /// Borrows the CPU mapping exclusively. The caller must synchronize GPU access.
    /// Bytes must be initialized before they are read or consumed by the GPU.
    ///
    /// # Safety
    /// The owning device must be alive, the allocation must be mapped, and the caller must
    /// exclude concurrent GPU or external host access to the returned bytes.
    ///
    /// ```compile_fail
    /// use no_graphics::GpuHeap;
    /// unsafe fn alias_mapping(heap: &mut GpuHeap) {
    ///     let first = heap.mapped_bytes().unwrap();
    ///     let second = heap.mapped_bytes().unwrap();
    ///     first[0].write(1);
    ///     second[0].write(2);
    /// }
    /// ```
    pub unsafe fn mapped_bytes(&mut self) -> Option<&mut [std::mem::MaybeUninit<u8>]> {
        let mapped = self.backing.mapped?;
        assert!(
            self.range.size <= isize::MAX as u64,
            "mapping exceeds slice size limit"
        );
        Some(unsafe {
            std::slice::from_raw_parts_mut(
                mapped
                    .as_ptr()
                    .cast::<std::mem::MaybeUninit<u8>>()
                    .add(self.mapped_offset),
                usize::try_from(self.range.size).expect("mapping exceeds address space"),
            )
        })
    }
}

impl From<&GpuHeap> for GpuRange {
    fn from(heap: &GpuHeap) -> Self {
        heap.range
    }
}

pub struct TextureHeap {
    size: u64,
    memory: vk::DeviceMemory,
}

impl TextureHeap {
    pub fn size(&self) -> u64 {
        self.size
    }
}

#[derive(Clone, Debug)]
pub struct TimelinePoint {
    pub semaphore: Arc<TimelineSemaphore>,
    pub value: u64,
}

#[derive(Clone, Debug)]
pub struct SwapchainFrame {
    pub render_view: Arc<RenderView>,
    pub extent: Uint32x2,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceCaps {
    pub device_name: String,
    pub max_push_data_size: u64,
    pub texture_heap_alignment: u64,
    pub texture_descriptor_size: u64,
    pub sampler_descriptor_size: u64,
    pub timestamp_period_ns: f32,
    pub sub_texel_precision_bits: u32,
    pub texture_compression_bc: bool,
    pub texture_compression_astc: bool,
    pub storage_input_output16: bool,
}

pub struct DeviceDesc {
    pub window: Option<Arc<Window>>,
    pub swapchain_format: Format,
    pub desired_swapchain_image_count: u32,
    pub timestamp_query_count: u32,
}

impl Default for DeviceDesc {
    fn default() -> Self {
        Self {
            window: None,
            swapchain_format: Format::Undefined,
            desired_swapchain_image_count: 2,
            timestamp_query_count: 256,
        }
    }
}

pub const fn get_texture_format_info(format: Format) -> TextureFormatInfo {
    use Format::*;
    let (bytes, block) = match format {
        R8Srgb | R8Unorm | R8Uint | S8Uint => (1, 1),
        Rg8Srgb | Rgba4Unorm | R5g5b5a1Unorm | R5g6b5Unorm | Rg8Unorm | R16Unorm | Rg8Uint
        | R16Uint | R16Float | D16Unorm => (2, 1),
        Rgba8Srgb | Bgra8Srgb | Rgba8Unorm | Bgra8Unorm | Rg16Unorm | Rgba8Uint | Bgra8Uint
        | Rg16Uint | R32Uint | Rg16Float | R32Float | Rgb10a2Unorm | Rg11b10Float
        | D24UnormS8Uint | D32Float => (4, 1),
        Rgba16Unorm | Rgba16Uint | Rg32Uint | Rgba16Float | Rg32Float | D32FloatS8Uint => (8, 1),
        Rgb32Uint | Rgb32Float => (12, 1),
        Rgba32Uint | Rgba32Float => (16, 1),
        EacRg | Astc4x4Srgb | Astc4x4Unorm | Bc3Srgb | Bc3Unorm | Bc5Rg | Bc7Srgb | Bc7Unorm => {
            (16, 4)
        }
        Undefined => (0, 0),
    };
    TextureFormatInfo {
        block_extent: Uint32x2 { x: block, y: block },
        bytes_per_block: bytes,
        depth: matches!(
            format,
            D16Unorm | D24UnormS8Uint | D32Float | D32FloatS8Uint
        ),
        stencil: matches!(format, S8Uint | D24UnormS8Uint | D32FloatS8Uint),
    }
}

fn compatible_view_formats(image: Format, view: Format) -> bool {
    if image == view {
        return true;
    }
    let a = get_texture_format_info(image);
    let b = get_texture_format_info(view);
    if a.depth || a.stencil || b.depth || b.stencil {
        return false;
    }
    if a.block_extent.x == 1 && b.block_extent.x == 1 {
        return a.bytes_per_block != 0 && a.bytes_per_block == b.bytes_per_block;
    }
    use Format::*;
    matches!(
        (image, view),
        (Astc4x4Unorm, Astc4x4Srgb)
            | (Astc4x4Srgb, Astc4x4Unorm)
            | (Bc3Unorm, Bc3Srgb)
            | (Bc3Srgb, Bc3Unorm)
            | (Bc7Unorm, Bc7Srgb)
            | (Bc7Srgb, Bc7Unorm)
    )
}

fn image_aspects(format: Format) -> vk::ImageAspectFlags {
    let info = get_texture_format_info(format);
    let mut result = vk::ImageAspectFlags::empty();
    if info.depth {
        result |= vk::ImageAspectFlags::DEPTH;
    }
    if info.stencil {
        result |= vk::ImageAspectFlags::STENCIL;
    }
    if result.is_empty() {
        result = vk::ImageAspectFlags::COLOR;
    }
    result
}

impl TextureType {
    fn vk(self) -> vk::ImageType {
        match self {
            Self::OneD => vk::ImageType::TYPE_1D,
            Self::ThreeD => vk::ImageType::TYPE_3D,
            _ => vk::ImageType::TYPE_2D,
        }
    }

    fn vk_view(self) -> vk::ImageViewType {
        match self {
            Self::OneD => vk::ImageViewType::TYPE_1D,
            Self::TwoD => vk::ImageViewType::TYPE_2D,
            Self::ThreeD => vk::ImageViewType::TYPE_3D,
            Self::Cube => vk::ImageViewType::CUBE,
            Self::TwoDArray => vk::ImageViewType::TYPE_2D_ARRAY,
            Self::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }
}

fn required_format_features(usage: TextureUsage) -> vk::FormatFeatureFlags2 {
    let mut result = vk::FormatFeatureFlags2::empty();
    if usage.intersects(TextureUsage::SAMPLED) {
        result |= vk::FormatFeatureFlags2::SAMPLED_IMAGE;
    }
    if usage.intersects(TextureUsage::STORAGE) {
        result |= vk::FormatFeatureFlags2::STORAGE_IMAGE
            | vk::FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT
            | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT;
    }
    if usage.intersects(TextureUsage::COLOR_ATTACHMENT) {
        result |= vk::FormatFeatureFlags2::COLOR_ATTACHMENT;
    }
    if usage.intersects(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        result |= vk::FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT;
    }
    if usage.intersects(TextureUsage::TRANSFER_SOURCE) {
        result |= vk::FormatFeatureFlags2::TRANSFER_SRC;
    }
    if usage.intersects(TextureUsage::TRANSFER_DESTINATION) {
        result |= vk::FormatFeatureFlags2::TRANSFER_DST;
    }
    result
}

fn image_usage(usage: TextureUsage) -> vk::ImageUsageFlags {
    let mut result = vk::ImageUsageFlags::empty();
    if usage.intersects(TextureUsage::SAMPLED) {
        result |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.intersects(TextureUsage::STORAGE) {
        result |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.intersects(TextureUsage::COLOR_ATTACHMENT) {
        result |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage.intersects(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        result |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    if usage.intersects(TextureUsage::TRANSFER_SOURCE) {
        result |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.intersects(TextureUsage::TRANSFER_DESTINATION) {
        result |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    result
}

const UNIVERSAL_BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);
const CPU_VISIBLE_MEMORY_PROPERTIES: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw()
        | vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);
const FORBIDDEN_MEMORY_PROPERTIES: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::LAZILY_ALLOCATED.as_raw()
        | vk::MemoryPropertyFlags::PROTECTED.as_raw()
        | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD.as_raw()
        | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD.as_raw(),
);
const ADDRESS_FLAGS: vk::AddressCommandFlagsKHR = vk::AddressCommandFlagsKHR::from_raw(
    vk::AddressCommandFlagsKHR::FULLY_BOUND.as_raw()
        | vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE.as_raw(),
);
fn is_usable_memory_type(properties: &vk::PhysicalDeviceMemoryProperties, index: u32) -> bool {
    let memory = properties.memory_types[index as usize];
    !memory
        .property_flags
        .intersects(FORBIDDEN_MEMORY_PROPERTIES)
        && !properties.memory_heaps[memory.heap_index as usize]
            .flags
            .intersects(vk::MemoryHeapFlags::TILE_MEMORY_QCOM)
}

fn has_name(values: &[vk::ExtensionProperties], name: &CStr) -> bool {
    values
        .iter()
        .any(|value| unsafe { CStr::from_ptr(value.extension_name.as_ptr()) == name })
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _: *mut c_void,
) -> vk::Bool32 {
    if severity.intersects(
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
    ) && !data.is_null()
        && !(*data).p_message.is_null()
    {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr(),
            "NoGraphicsAPI validation: {}",
            CStr::from_ptr((*data).p_message).to_string_lossy()
        );
    }
    vk::FALSE
}

macro_rules! device_functions {
    ($($field:ident: $ty:ident = $name:literal),* $(,)?) => {
        struct DeviceFunctions { $($field: vk::$ty,)* }
        impl DeviceFunctions {
            unsafe fn load(instance: &ash::Instance, device: &ash::Device) -> Result<Self> {
                Ok(Self { $($field: std::mem::transmute::<unsafe extern "system" fn(), vk::$ty>(
                    instance.get_device_proc_addr(device.handle(), $name.as_ptr()).ok_or(Error::DriverError)?),)* })
            }
        }
    };
}

device_functions! {
    write_sampler_descriptors: PFN_vkWriteSamplerDescriptorsEXT = c"vkWriteSamplerDescriptorsEXT",
    write_resource_descriptors: PFN_vkWriteResourceDescriptorsEXT = c"vkWriteResourceDescriptorsEXT",
    cmd_bind_sampler_heap: PFN_vkCmdBindSamplerHeapEXT = c"vkCmdBindSamplerHeapEXT",
    cmd_bind_texture_heap: PFN_vkCmdBindResourceHeapEXT = c"vkCmdBindResourceHeapEXT",
    cmd_push_data: PFN_vkCmdPushDataEXT = c"vkCmdPushDataEXT",
    cmd_bind_index_buffer: PFN_vkCmdBindIndexBuffer3KHR = c"vkCmdBindIndexBuffer3KHR",
    cmd_draw_indirect: PFN_vkCmdDrawIndirect2KHR = c"vkCmdDrawIndirect2KHR",
    cmd_draw_indexed_indirect: PFN_vkCmdDrawIndexedIndirect2KHR = c"vkCmdDrawIndexedIndirect2KHR",
    cmd_dispatch_indirect: PFN_vkCmdDispatchIndirect2KHR = c"vkCmdDispatchIndirect2KHR",
    cmd_draw_mesh_tasks: PFN_vkCmdDrawMeshTasksEXT = c"vkCmdDrawMeshTasksEXT",
    cmd_draw_mesh_tasks_indirect: PFN_vkCmdDrawMeshTasksIndirect2EXT = c"vkCmdDrawMeshTasksIndirect2EXT",
    cmd_copy_memory: PFN_vkCmdCopyMemoryKHR = c"vkCmdCopyMemoryKHR",
    cmd_copy_memory_to_image: PFN_vkCmdCopyMemoryToImageKHR = c"vkCmdCopyMemoryToImageKHR",
    cmd_copy_image_to_memory: PFN_vkCmdCopyImageToMemoryKHR = c"vkCmdCopyImageToMemoryKHR",
    cmd_copy_query_pool_results_to_memory: PFN_vkCmdCopyQueryPoolResultsToMemoryKHR = c"vkCmdCopyQueryPoolResultsToMemoryKHR",
}

#[derive(Default)]
struct BackingBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: Option<ptr::NonNull<c_void>>,
    address: u64,
}

// SAFETY: the mapping has no thread affinity. Access requires an exclusive heap borrow;
// callers of mapped_bytes must separately synchronize GPU and external host accesses.
unsafe impl Send for BackingBuffer {}
unsafe impl Sync for BackingBuffer {}

#[derive(Debug)]
pub struct TimelineSemaphore {
    semaphore: vk::Semaphore,
}

/// An exclusively owned recording, consumed by submission.
///
/// ```compile_fail
/// use no_graphics::{CommandBuffer, Device, TimelinePoint};
/// unsafe fn submit_twice(device: &mut Device, commands: CommandBuffer, point: &TimelinePoint) {
///     device.submit(vec![commands], point);
///     device.submit(vec![commands], point); // already moved
/// }
/// ```
#[must_use = "submit command buffers in the next batch"]
pub struct CommandBuffer {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    timestamp_pool: vk::QueryPool,
    timestamp_destinations: Vec<u64>,
    timestamp_count: u32,
    retire_value: u64,
    swapchain: bool,
}

impl CommandBuffer {
    fn new() -> Self {
        Self {
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            timestamp_pool: vk::QueryPool::null(),
            timestamp_destinations: Vec::new(),
            timestamp_count: 0,
            retire_value: 0,
            swapchain: false,
        }
    }
}

#[derive(Default)]
struct TextureInitialization {
    image: vk::Image,
    aspect_mask: vk::ImageAspectFlags,
    mip_levels: u32,
    array_layers: u32,
}

#[derive(Clone, Copy, Default)]
struct PresentContext {
    acquired: vk::Semaphore,
    rendered: vk::Semaphore,
    presented: vk::Fence,
    swapchain: vk::SwapchainKHR,
    present_pending: bool,
}

#[derive(Clone, Copy, Default)]
struct RetiredSwapchain {
    handle: vk::SwapchainKHR,
    views: [vk::ImageView; MAX_SWAPCHAIN_IMAGES],
    view_count: usize,
}

struct DeferredSwapchainImage {
    retire_value: u64,
    swapchain: vk::SwapchainKHR,
    view: vk::ImageView,
}

#[derive(Default)]
struct SwapchainDeleteQueue {
    entries: VecDeque<DeferredSwapchainImage>,
}

impl SwapchainDeleteQueue {
    fn push(&mut self, retire_value: u64, swapchain: vk::SwapchainKHR, view: vk::ImageView) {
        debug_assert!(swapchain != vk::SwapchainKHR::null() && view != vk::ImageView::null());
        debug_assert!(
            self.entries
                .back()
                .is_none_or(|entry| entry.retire_value <= retire_value)
        );
        self.entries.push_back(DeferredSwapchainImage {
            retire_value,
            swapchain,
            view,
        });
    }

    unsafe fn collect(
        &mut self,
        device: &ash::Device,
        swapchains: &ash::khr::swapchain::Device,
        completed: u64,
    ) {
        while self
            .entries
            .front()
            .is_some_and(|entry| entry.retire_value <= completed)
        {
            let entry = self.entries.pop_front().unwrap();
            device.destroy_image_view(entry.view, None);
            if self
                .entries
                .front()
                .is_none_or(|next| next.swapchain != entry.swapchain)
            {
                swapchains.destroy_swapchain(entry.swapchain, None);
            }
        }
    }
}

pub struct Texture {
    image: vk::Image,
    width: u32,
    height: u32,
    depth: u32,
    layer_count: u32,
    r#type: TextureType,
    format: Format,
}

#[derive(Debug)]
pub struct RenderView {
    view: vk::ImageView,
    width: u32,
    height: u32,
    swapchain_view: bool,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    images: [vk::Image; MAX_SWAPCHAIN_IMAGES],
    render_views: [Option<Arc<RenderView>>; MAX_SWAPCHAIN_IMAGES],
    initialized: [bool; MAX_SWAPCHAIN_IMAGES],
    image_count: usize,
    image_index: u32,
    width: u32,
    height: u32,
    format: Format,
    transform: vk::SurfaceTransformFlagsKHR,
    composite_alpha: vk::CompositeAlphaFlagsKHR,
    present_context: Option<usize>,
    transition_commands: Option<vk::CommandBuffer>,
    acquired: bool,
    recreate_required: bool,
}

pub struct PSO {
    pso: vk::Pipeline,
    bind_point: vk::PipelineBindPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Allocation {
    Buffer(vk::Buffer, vk::DeviceMemory, bool),
    Memory(vk::DeviceMemory),
    Image(vk::Image),
    View(vk::ImageView),
    Pipeline(vk::Pipeline),
    Semaphore(vk::Semaphore),
    Commands(vk::CommandPool, vk::QueryPool),
}

impl Allocation {
    unsafe fn destroy(self, device: &ash::Device) {
        match self {
            Self::Buffer(buffer, memory, mapped) => {
                if mapped {
                    device.unmap_memory(memory);
                }
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            Self::Memory(memory) => device.free_memory(memory, None),
            Self::Image(image) => device.destroy_image(image, None),
            Self::View(view) => device.destroy_image_view(view, None),
            Self::Pipeline(pipeline) => device.destroy_pipeline(pipeline, None),
            Self::Semaphore(semaphore) => device.destroy_semaphore(semaphore, None),
            Self::Commands(pool, queries) => {
                if queries != vk::QueryPool::null() {
                    device.destroy_query_pool(queries, None);
                }
                device.destroy_command_pool(pool, None);
            }
        }
    }
}

pub struct Device {
    allocations: Vec<Allocation>,
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: ash::ext::debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_api: ash::khr::surface::Instance,
    surface_caps_api: ash::khr::get_surface_capabilities2::Instance,
    surface: vk::SurfaceKHR,
    window: Option<Arc<Window>>,
    device: Option<ash::Device>,
    swapchain_api: Option<ash::khr::swapchain::Device>,
    functions: Option<DeviceFunctions>,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    timestamp_query_count: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    physical_properties: vk::PhysicalDeviceProperties,
    heap_properties: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    max_timeline_value_difference: u64,
    texture_heap_alignment: u64,
    texture_memory_type: u32,
    caps: DeviceCaps,
    format_features: [vk::FormatFeatureFlags2; FORMAT_COUNT],
    texture_compression_etc2: bool,
    pending_texture_initializations: VecDeque<TextureInitialization>,
    command_contexts: VecDeque<CommandBuffer>,
    command_submit_infos: Vec<vk::CommandBufferSubmitInfo<'static>>,
    command_retirement: vk::Semaphore,
    command_retirement_value: u64,
    completed_command_retirement: u64,
    swapchain_delete_queue: SwapchainDeleteQueue,
    present_contexts: [PresentContext; MAX_SWAPCHAIN_IMAGES],
    retired_swapchains: [RetiredSwapchain; MAX_SWAPCHAIN_IMAGES],
    swapchain: Option<Swapchain>,
    active_command_buffers: usize,
    present_context_count: usize,
    next_present_context: usize,
}

impl Device {
    unsafe fn release(&mut self, allocation: Allocation) {
        let index = self
            .allocations
            .iter()
            .position(|entry| *entry == allocation)
            .expect("resource does not belong to this device or was already destroyed");
        self.allocations.remove(index).destroy(self.vk());
    }

    fn with_swapchain<T>(&mut self, operation: impl FnOnce(&mut Self, &mut Swapchain) -> T) -> T {
        let mut swapchain = self.swapchain.take().expect("device has no swapchain");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            operation(self, &mut swapchain)
        }));
        self.swapchain = Some(swapchain);
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn vk(&self) -> &ash::Device {
        self.device.as_ref().unwrap()
    }

    fn functions(&self) -> &DeviceFunctions {
        self.functions.as_ref().unwrap()
    }

    fn swapchains(&self) -> &ash::khr::swapchain::Device {
        self.swapchain_api.as_ref().unwrap()
    }

    fn find_memory_type(
        &self,
        bits: u32,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        minimum_heap_size: u64,
        avoided: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let mut best: Option<(bool, u32, u64, u32)> = None;
        for index in 0..self.memory_properties.memory_type_count {
            let memory = self.memory_properties.memory_types[index as usize];
            let heap = self.memory_properties.memory_heaps[memory.heap_index as usize];
            if bits & (1 << index) == 0
                || !memory.property_flags.contains(required)
                || !is_usable_memory_type(&self.memory_properties, index)
                || heap.size < minimum_heap_size
            {
                continue;
            }
            let rank = (
                !memory.property_flags.intersects(avoided),
                (memory.property_flags & preferred).as_raw().count_ones(),
                heap.size,
            );
            if best.is_none_or(|b| rank > (b.0, b.1, b.2)) {
                best = Some((rank.0, rank.1, rank.2, index));
            }
        }
        best.map(|b| b.3)
    }

    unsafe fn create_backing_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        avoided: vk::MemoryPropertyFlags,
    ) -> Result<BackingBuffer> {
        let mut result = BackingBuffer {
            buffer: self.vk().create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?,
            ..Default::default()
        };
        let requirements = self.vk().get_buffer_memory_requirements(result.buffer);
        let Some(memory_type) = self.find_memory_type(
            requirements.memory_type_bits,
            required,
            preferred,
            requirements.size,
            avoided,
        ) else {
            self.vk().destroy_buffer(result.buffer, None);
            return Err(Error::Unsupported);
        };
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocation = vk::MemoryAllocateInfo::default()
            .push(&mut flags)
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        result.memory = match self.vk().allocate_memory(&allocation, None) {
            Ok(memory) => memory,
            Err(error) => {
                self.vk().destroy_buffer(result.buffer, None);
                return Err(error.into());
            }
        };
        if let Err(error) = self
            .vk()
            .bind_buffer_memory(result.buffer, result.memory, 0)
        {
            self.vk().destroy_buffer(result.buffer, None);
            self.vk().free_memory(result.memory, None);
            return Err(error.into());
        }
        if required.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            result.mapped = match self.vk().map_memory(
                result.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            ) {
                Ok(mapped) => ptr::NonNull::new(mapped),
                Err(error) => {
                    self.vk().destroy_buffer(result.buffer, None);
                    self.vk().free_memory(result.memory, None);
                    return Err(error.into());
                }
            };
        }
        result.address = self.vk().get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(result.buffer),
        );
        Ok(result)
    }
}

#[derive(Default)]
struct QueriedFeatures {
    core: vk::PhysicalDeviceFeatures2<'static>,
    vulkan11: vk::PhysicalDeviceVulkan11Features<'static>,
    vulkan12: vk::PhysicalDeviceVulkan12Features<'static>,
    vulkan13: vk::PhysicalDeviceVulkan13Features<'static>,
    vulkan14: vk::PhysicalDeviceVulkan14Features<'static>,
    descriptor_heap: vk::PhysicalDeviceDescriptorHeapFeaturesEXT<'static>,
    address_commands: vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR<'static>,
    untyped_pointers: vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR<'static>,
    unified_image_layouts: vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR<'static>,
    mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    swapchain_maintenance1: vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR<'static>,
}

impl QueriedFeatures {
    // Link only after the structure reaches its call-site address; never move a linked chain.
    fn link(&mut self, presentation: bool, unified: bool) {
        self.core.p_next = ptr::from_mut(&mut self.vulkan11).cast();
        self.vulkan11.p_next = ptr::from_mut(&mut self.vulkan12).cast();
        self.vulkan12.p_next = ptr::from_mut(&mut self.vulkan13).cast();
        self.vulkan13.p_next = ptr::from_mut(&mut self.vulkan14).cast();
        self.vulkan14.p_next = ptr::from_mut(&mut self.descriptor_heap).cast();
        self.descriptor_heap.p_next = ptr::from_mut(&mut self.address_commands).cast();
        self.address_commands.p_next = ptr::from_mut(&mut self.untyped_pointers).cast();
        self.untyped_pointers.p_next = if unified {
            ptr::from_mut(&mut self.unified_image_layouts).cast()
        } else {
            ptr::from_mut(&mut self.mesh_shader).cast()
        };
        self.unified_image_layouts.p_next = ptr::from_mut(&mut self.mesh_shader).cast();
        self.mesh_shader.p_next = if presentation {
            ptr::from_mut(&mut self.swapchain_maintenance1).cast()
        } else {
            ptr::null_mut()
        };
    }
}

#[derive(Default)]
struct Candidate {
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    unified_image_layouts: bool,
    image_cube_array: bool,
    texture_compression_bc: bool,
    texture_compression_astc: bool,
    texture_compression_etc2: bool,
    storage_input_output16: bool,
    khr_swapchain_maintenance1: bool,
    heap_properties: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    vulkan12_properties: vk::PhysicalDeviceVulkan12Properties<'static>,
}

const REQUIRED_DEVICE_EXTENSIONS: [&CStr; 4] = [
    ash::ext::descriptor_heap::NAME,
    ash::khr::device_address_commands::NAME,
    ash::khr::shader_untyped_pointers::NAME,
    ash::ext::mesh_shader::NAME,
];
unsafe fn inspect_candidate(
    state: &Device,
    physical_device: vk::PhysicalDevice,
    khr_surface_maintenance1: bool,
    ext_surface_maintenance1: bool,
) -> Result<Candidate> {
    let extensions = state
        .instance
        .enumerate_device_extension_properties(physical_device)?;
    if extensions.len() > 512
        || REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .any(|name| !has_name(&extensions, name))
    {
        return Err(Error::Unsupported);
    }
    let unified_extension = has_name(&extensions, ash::khr::unified_image_layouts::NAME);
    let khr_maintenance =
        khr_surface_maintenance1 && has_name(&extensions, ash::khr::swapchain_maintenance1::NAME);
    let ext_maintenance =
        ext_surface_maintenance1 && has_name(&extensions, ash::ext::swapchain_maintenance1::NAME);
    let presentation = state.surface != vk::SurfaceKHR::null();
    if presentation
        && (!has_name(&extensions, ash::khr::swapchain::NAME)
            || !(khr_maintenance || ext_maintenance))
    {
        return Err(Error::Unsupported);
    }
    let mut result = Candidate {
        physical_device,
        ..Default::default()
    };
    {
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push(&mut result.heap_properties)
            .push(&mut result.vulkan12_properties);
        state
            .instance
            .get_physical_device_properties2(physical_device, &mut properties);
        result.properties = properties.properties;
    }
    result.heap_properties.p_next = ptr::null_mut();
    result.vulkan12_properties.p_next = ptr::null_mut();
    if result.properties.api_version < vk::API_VERSION_1_4 {
        return Err(Error::Unsupported);
    }
    result.memory_properties = state
        .instance
        .get_physical_device_memory_properties(physical_device);
    if !(0..result.memory_properties.memory_type_count).any(|index| {
        is_usable_memory_type(&result.memory_properties, index)
            && result.memory_properties.memory_types[index as usize]
                .property_flags
                .contains(CPU_VISIBLE_MEMORY_PROPERTIES)
    }) {
        return Err(Error::Unsupported);
    }
    let mut features = QueriedFeatures::default();
    features.link(presentation, unified_extension);
    state
        .instance
        .get_physical_device_features2(physical_device, &mut features.core);
    let core = features.core.features;
    let required = core.shader_int16 == vk::TRUE
        && core.sampler_anisotropy == vk::TRUE
        && core.depth_bias_clamp == vk::TRUE
        && core.independent_blend == vk::TRUE
        && core.fragment_stores_and_atomics == vk::TRUE
        && core.vertex_pipeline_stores_and_atomics == vk::TRUE
        && core.shader_storage_image_read_without_format == vk::TRUE
        && core.shader_storage_image_write_without_format == vk::TRUE
        && core.multi_draw_indirect == vk::TRUE
        && core.draw_indirect_first_instance == vk::TRUE
        && features.vulkan11.storage_buffer16_bit_access == vk::TRUE
        && features.vulkan11.storage_push_constant16 == vk::TRUE
        && features.vulkan11.shader_draw_parameters == vk::TRUE
        && features.vulkan12.shader_float16 == vk::TRUE
        && features.vulkan12.scalar_block_layout == vk::TRUE
        && features.vulkan12.buffer_device_address == vk::TRUE
        && features.vulkan12.timeline_semaphore == vk::TRUE
        && features.vulkan13.synchronization2 == vk::TRUE
        && features.vulkan13.dynamic_rendering == vk::TRUE
        && features.vulkan13.maintenance4 == vk::TRUE
        && features.vulkan14.maintenance5 == vk::TRUE
        && features.descriptor_heap.descriptor_heap == vk::TRUE
        && features.address_commands.device_address_commands == vk::TRUE
        && features.untyped_pointers.shader_untyped_pointers == vk::TRUE
        && features.mesh_shader.mesh_shader == vk::TRUE
        && (core.texture_compression_bc == vk::TRUE
            || core.texture_compression_astc_ldr == vk::TRUE)
        && (!presentation || features.swapchain_maintenance1.swapchain_maintenance1 == vk::TRUE);
    if !required {
        return Err(Error::Unsupported);
    }
    result.unified_image_layouts =
        unified_extension && features.unified_image_layouts.unified_image_layouts == vk::TRUE;
    result.image_cube_array = core.image_cube_array == vk::TRUE;
    result.texture_compression_bc = core.texture_compression_bc == vk::TRUE;
    result.texture_compression_astc = core.texture_compression_astc_ldr == vk::TRUE;
    result.texture_compression_etc2 = core.texture_compression_etc2 == vk::TRUE;
    result.storage_input_output16 = features.vulkan11.storage_input_output16 == vk::TRUE;
    result.khr_swapchain_maintenance1 = khr_maintenance;
    let queues = state
        .instance
        .get_physical_device_queue_family_properties(physical_device);
    if queues.len() > 64 {
        return Err(Error::Unsupported);
    }
    for (index, queue) in queues.iter().enumerate() {
        if queue.queue_count == 0
            || queue.timestamp_valid_bits != 64
            || !queue
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        {
            continue;
        }
        if !presentation
            || state.surface_api.get_physical_device_surface_support(
                physical_device,
                index as u32,
                state.surface,
            )?
        {
            result.queue_family = index as u32;
            return Ok(result);
        }
    }
    Err(Error::Unsupported)
}

impl Device {
    /// Creates a headless device when `window` is `None`. A windowed device uses the supplied
    /// swapchain format and must be created on its winit event-loop thread.
    pub fn new(desc: &DeviceDesc) -> Result<Self> {
        unsafe {
            debug_assert!(
                (1..=MAX_SWAPCHAIN_IMAGES as u32).contains(&desc.desired_swapchain_image_count)
            );
            let presentation = desc.window.is_some();
            let entry = ash::Entry::load().map_err(|_| Error::Unsupported)?;
            if entry
                .try_enumerate_instance_version()?
                .unwrap_or(vk::API_VERSION_1_0)
                < vk::API_VERSION_1_4
            {
                return Err(Error::Unsupported);
            }
            let extensions = entry.enumerate_instance_extension_properties(None)?;
            if extensions.len() > 256 {
                return Err(Error::Unsupported);
            }
            let khr_maintenance =
                presentation && has_name(&extensions, ash::khr::surface_maintenance1::NAME);
            let ext_maintenance =
                presentation && has_name(&extensions, ash::ext::surface_maintenance1::NAME);
            let mut enabled_extensions = Vec::with_capacity(8);
            if let Some(window) = &desc.window {
                if !(khr_maintenance || ext_maintenance)
                    || !has_name(&extensions, ash::khr::get_surface_capabilities2::NAME)
                {
                    return Err(Error::Unsupported);
                }
                for &name in ash_window::enumerate_required_extensions(
                    window
                        .display_handle()
                        .map_err(|_| Error::Unsupported)?
                        .as_raw(),
                )? {
                    if !has_name(&extensions, CStr::from_ptr(name)) {
                        return Err(Error::Unsupported);
                    }
                    enabled_extensions.push(name);
                }
                enabled_extensions.push(ash::khr::get_surface_capabilities2::NAME.as_ptr());
                if khr_maintenance {
                    enabled_extensions.push(ash::khr::surface_maintenance1::NAME.as_ptr());
                }
                if ext_maintenance {
                    enabled_extensions.push(ash::ext::surface_maintenance1::NAME.as_ptr());
                }
            }
            let debug_available =
                cfg!(debug_assertions) && has_name(&extensions, ash::ext::debug_utils::NAME);
            let mut enabled_layers = Vec::new();
            if cfg!(debug_assertions) {
                let layers = entry.enumerate_instance_layer_properties()?;
                if layers.len() > 64 {
                    return Err(Error::Unsupported);
                }
                if layers.iter().any(|layer| {
                    CStr::from_ptr(layer.layer_name.as_ptr()) == c"VK_LAYER_KHRONOS_validation"
                }) {
                    enabled_layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
                }
            }
            if debug_available {
                enabled_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            }
            let application = vk::ApplicationInfo::default()
                .application_name(c"NoGraphicsAPI application")
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .engine_name(c"NoGraphicsAPI")
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(vk::API_VERSION_1_4);
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&application)
                    .enabled_extension_names(&enabled_extensions)
                    .enabled_layer_names(&enabled_layers),
                None,
            )?;
            let mut state = Device {
                allocations: Vec::new(),
                debug_utils: ash::ext::debug_utils::Instance::load(&entry, &instance),
                surface_api: ash::khr::surface::Instance::load(&entry, &instance),
                surface_caps_api: ash::khr::get_surface_capabilities2::Instance::load(
                    &entry, &instance,
                ),
                entry,
                instance,
                debug_messenger: vk::DebugUtilsMessengerEXT::null(),
                surface: vk::SurfaceKHR::null(),
                window: desc.window.clone(),
                device: None,
                swapchain_api: None,
                functions: None,
                physical_device: vk::PhysicalDevice::null(),
                queue: vk::Queue::null(),
                queue_family: 0,
                timestamp_query_count: desc.timestamp_query_count,
                memory_properties: Default::default(),
                physical_properties: Default::default(),
                heap_properties: Default::default(),
                max_timeline_value_difference: 0,
                texture_heap_alignment: GPU_ALLOCATION_ALIGNMENT,
                texture_memory_type: vk::MAX_MEMORY_TYPES as u32,
                caps: Default::default(),
                format_features: [vk::FormatFeatureFlags2::empty(); FORMAT_COUNT],
                texture_compression_etc2: false,
                pending_texture_initializations: Default::default(),
                command_contexts: VecDeque::new(),
                command_submit_infos: Vec::new(),
                command_retirement: vk::Semaphore::null(),
                command_retirement_value: 0,
                completed_command_retirement: 0,
                swapchain_delete_queue: Default::default(),
                present_contexts: [PresentContext::default(); MAX_SWAPCHAIN_IMAGES],
                retired_swapchains: [RetiredSwapchain::default(); MAX_SWAPCHAIN_IMAGES],
                swapchain: None,
                active_command_buffers: 0,
                present_context_count: if presentation {
                    desc.desired_swapchain_image_count as usize
                } else {
                    0
                },
                next_present_context: 0,
            };
            if debug_available {
                for name in [
                    c"vkCreateDebugUtilsMessengerEXT",
                    c"vkDestroyDebugUtilsMessengerEXT",
                ] {
                    if state
                        .entry
                        .get_instance_proc_addr(state.instance.handle(), name.as_ptr())
                        .is_none()
                    {
                        return Err(Error::DriverError);
                    }
                }
                state.debug_messenger = state.debug_utils.create_debug_utils_messenger(
                    &vk::DebugUtilsMessengerCreateInfoEXT::default()
                        .message_severity(
                            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                        )
                        .message_type(
                            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                        )
                        .pfn_user_callback(Some(debug_callback)),
                    None,
                )?;
            }
            if let Some(window) = &state.window {
                let display = window
                    .display_handle()
                    .map_err(|_| Error::Unsupported)?
                    .as_raw();
                let handle = window
                    .window_handle()
                    .map_err(|_| Error::Unsupported)?
                    .as_raw();
                state.surface =
                    ash_window::SurfaceFactory::new(&state.entry, &state.instance, display)?
                        .create_surface(handle, None)?;
            }
            let devices = state.instance.enumerate_physical_devices()?;
            if devices.len() > 32 {
                return Err(Error::Unsupported);
            }
            let mut selected = None;
            for physical_device in devices {
                let candidate = match inspect_candidate(
                    &state,
                    physical_device,
                    khr_maintenance,
                    ext_maintenance,
                ) {
                    Ok(candidate) => candidate,
                    Err(Error::Unsupported) => continue,
                    Err(error) => return Err(error),
                };
                let discrete =
                    candidate.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                if selected.is_none() || discrete {
                    selected = Some(candidate);
                }
                if discrete {
                    break;
                }
            }
            let selected = selected.ok_or(Error::Unsupported)?;
            state.physical_device = selected.physical_device;
            state.queue_family = selected.queue_family;
            state.physical_properties = selected.properties;
            state.memory_properties = selected.memory_properties;
            state.heap_properties = selected.heap_properties;
            state.max_timeline_value_difference = selected
                .vulkan12_properties
                .max_timeline_semaphore_value_difference;
            state.texture_compression_etc2 = selected.texture_compression_etc2;
            for format in FORMATS {
                let mut features = vk::FormatProperties3::default();
                let mut properties = vk::FormatProperties2::default().push(&mut features);
                state.instance.get_physical_device_format_properties2(
                    state.physical_device,
                    format.vk(),
                    &mut properties,
                );
                state.format_features[format as usize] = features.optimal_tiling_features;
            }
            let mut enabled = QueriedFeatures::default();
            enabled.core.features = vk::PhysicalDeviceFeatures::default()
                .image_cube_array(selected.image_cube_array)
                .sampler_anisotropy(true)
                .shader_int16(true)
                .depth_bias_clamp(true)
                .independent_blend(true)
                .texture_compression_bc(selected.texture_compression_bc)
                .texture_compression_astc_ldr(selected.texture_compression_astc)
                .texture_compression_etc2(selected.texture_compression_etc2)
                .fragment_stores_and_atomics(true)
                .vertex_pipeline_stores_and_atomics(true)
                .shader_storage_image_read_without_format(true)
                .shader_storage_image_write_without_format(true)
                .multi_draw_indirect(true)
                .draw_indirect_first_instance(true);
            enabled.vulkan11 = vk::PhysicalDeviceVulkan11Features::default()
                .storage_buffer16_bit_access(true)
                .storage_push_constant16(true)
                .storage_input_output16(selected.storage_input_output16)
                .shader_draw_parameters(true);
            enabled.vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
                .shader_float16(true)
                .scalar_block_layout(true)
                .timeline_semaphore(true)
                .buffer_device_address(true);
            enabled.vulkan13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true)
                .maintenance4(true);
            enabled.vulkan14.maintenance5 = vk::TRUE;
            enabled.descriptor_heap.descriptor_heap = vk::TRUE;
            enabled.address_commands.device_address_commands = vk::TRUE;
            enabled.untyped_pointers.shader_untyped_pointers = vk::TRUE;
            enabled.unified_image_layouts.unified_image_layouts =
                selected.unified_image_layouts.into();
            enabled.mesh_shader.mesh_shader = vk::TRUE;
            enabled.swapchain_maintenance1.swapchain_maintenance1 = vk::TRUE;
            enabled.link(presentation, selected.unified_image_layouts);
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(state.queue_family)
                .queue_priorities(&priorities)];
            let mut device_extensions: Vec<_> = REQUIRED_DEVICE_EXTENSIONS
                .iter()
                .map(|name| name.as_ptr())
                .collect();
            if selected.unified_image_layouts {
                device_extensions.push(ash::khr::unified_image_layouts::NAME.as_ptr());
            }
            if presentation {
                device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
                device_extensions.push(if selected.khr_swapchain_maintenance1 {
                    ash::khr::swapchain_maintenance1::NAME.as_ptr()
                } else {
                    ash::ext::swapchain_maintenance1::NAME.as_ptr()
                });
            }
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&device_extensions);
            device_info.p_next = ptr::from_ref(&enabled.core).cast();
            state.device = Some(state.instance.create_device(
                state.physical_device,
                &device_info,
                None,
            )?);
            state.swapchain_api = Some(ash::khr::swapchain::Device::load(
                &state.instance,
                state.vk(),
            ));
            state.queue = state.vk().get_device_queue(state.queue_family, 0);
            if !supports_gpu_heap_memory(&state) || !select_texture_memory_type(&mut state) {
                return Err(Error::Unsupported);
            }
            state.functions = Some(DeviceFunctions::load(&state.instance, state.vk())?);
            state.create_command_contexts()?;
            for index in 0..state.present_context_count {
                state.create_present_context(index)?;
            }
            state.caps = DeviceCaps {
                device_name: CStr::from_ptr(selected.properties.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
                max_push_data_size: state.heap_properties.max_push_data_size,
                texture_heap_alignment: state.texture_heap_alignment,
                texture_descriptor_size: state.heap_properties.image_descriptor_size,
                sampler_descriptor_size: state.heap_properties.sampler_descriptor_size,
                timestamp_period_ns: selected.properties.limits.timestamp_period,
                sub_texel_precision_bits: selected.properties.limits.sub_texel_precision_bits,
                texture_compression_bc: selected.texture_compression_bc,
                texture_compression_astc: selected.texture_compression_astc,
                storage_input_output16: selected.storage_input_output16,
            };
            if presentation {
                let mut swapchain = Swapchain {
                    handle: vk::SwapchainKHR::null(),
                    images: [vk::Image::null(); MAX_SWAPCHAIN_IMAGES],
                    render_views: std::array::from_fn(|_| None),
                    initialized: [false; MAX_SWAPCHAIN_IMAGES],
                    image_count: 0,
                    image_index: 0,
                    width: 0,
                    height: 0,
                    format: desc.swapchain_format,
                    transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
                    composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                    present_context: None,
                    transition_commands: None,
                    acquired: false,
                    recreate_required: false,
                };
                recreate_swapchain(&mut state, &mut swapchain)?;
                state.swapchain = Some(swapchain);
            }
            Ok(state)
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            if self.device.is_some() {
                // Dropping an unsubmitted recording must not leak its command pool.
                self.active_command_buffers = 0;
                if let Some(index) = self
                    .swapchain
                    .as_ref()
                    .filter(|swapchain| swapchain.acquired)
                    .and_then(|swapchain| swapchain.present_context)
                {
                    // WSI acquisition may still be signaling its semaphore. Consume the
                    // signal before destroying an acquired frame that was never submitted.
                    let waits = [vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.present_contexts[index].acquired)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
                    let submissions = [vk::SubmitInfo2::default().wait_semaphore_infos(&waits)];
                    let _ = self
                        .vk()
                        .queue_submit2(self.queue, &submissions, vk::Fence::null());
                }
                if self.vk().device_wait_idle().is_ok() {
                    // Idle also covers a retirement value reserved by a failed submission.
                    self.completed_command_retirement = self.command_retirement_value;
                }
                self.drain_contexts();
                if let Some(mut swapchain) = self.swapchain.take() {
                    retire_swapchain_handle(self, &mut swapchain);
                }
                self.collect_swapchains();
                debug_assert!(self.swapchain_delete_queue.entries.is_empty());
                self.destroy_command_contexts();
                for index in 0..self.present_context_count {
                    self.destroy_present_context(index);
                }
                // Reverse creation order releases views before images, and images before heaps.
                while let Some(allocation) = self.allocations.pop() {
                    allocation.destroy(self.vk());
                }
                self.vk().destroy_device(None);
            }
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_api.destroy_surface(self.surface, None);
            }
            if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                self.debug_utils
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn buffer_memory_requirements(
    device: &Device,
    usage: vk::BufferUsageFlags,
) -> vk::MemoryRequirements {
    let buffer = vk::BufferCreateInfo::default()
        .size(1)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let info = vk::DeviceBufferMemoryRequirements::default().create_info(&buffer);
    let mut requirements = vk::MemoryRequirements2::default();
    device
        .vk()
        .get_device_buffer_memory_requirements(&info, &mut requirements);
    requirements.memory_requirements
}

unsafe fn supports_gpu_heap_memory(device: &Device) -> bool {
    let ordinary = buffer_memory_requirements(device, UNIVERSAL_BUFFER_USAGE);
    if device
        .find_memory_type(
            ordinary.memory_type_bits,
            CPU_VISIBLE_MEMORY_PROPERTIES,
            vk::MemoryPropertyFlags::empty(),
            ordinary.size,
            vk::MemoryPropertyFlags::empty(),
        )
        .is_none()
        || device
            .find_memory_type(
                ordinary.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
                ordinary.size,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            )
            .is_none()
    {
        return false;
    }
    let descriptor = buffer_memory_requirements(
        device,
        UNIVERSAL_BUFFER_USAGE | vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT,
    );
    device
        .find_memory_type(
            descriptor.memory_type_bits,
            CPU_VISIBLE_MEMORY_PROPERTIES,
            vk::MemoryPropertyFlags::empty(),
            descriptor.size,
            vk::MemoryPropertyFlags::empty(),
        )
        .is_some()
}

fn fits_image_format_properties(
    info: &vk::ImageCreateInfo<'_>,
    properties: &vk::ImageFormatProperties,
) -> bool {
    info.extent.width <= properties.max_extent.width
        && info.extent.height <= properties.max_extent.height
        && info.extent.depth <= properties.max_extent.depth
        && info.mip_levels <= properties.max_mip_levels
        && info.array_layers <= properties.max_array_layers
}

unsafe fn image_format_properties(
    device: &Device,
    info: &vk::ImageCreateInfo<'_>,
) -> Option<vk::ImageFormatProperties> {
    let format = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(info.format)
        .ty(info.image_type)
        .tiling(info.tiling)
        .usage(info.usage)
        .flags(info.flags);
    let mut properties = vk::ImageFormatProperties2::default();
    match device
        .instance
        .get_physical_device_image_format_properties2(
            device.physical_device,
            &format,
            &mut properties,
        ) {
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => None,
        result => {
            require(result);
            fits_image_format_properties(info, &properties.image_format_properties)
                .then_some(properties.image_format_properties)
        }
    }
}

unsafe fn image_memory_requirements(
    device: &Device,
    info: &vk::ImageCreateInfo<'_>,
) -> vk::MemoryRequirements {
    let mut requirements = vk::MemoryRequirements2::default();
    device.vk().get_device_image_memory_requirements(
        &vk::DeviceImageMemoryRequirements::default().create_info(info),
        &mut requirements,
    );
    requirements.memory_requirements
}

unsafe fn select_texture_memory_type(device: &mut Device) -> bool {
    let color_features = device.format_features[Format::Rgba8Unorm as usize];
    if !color_features.contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE) {
        return false;
    }
    let probe_size = device
        .physical_properties
        .limits
        .max_image_dimension2_d
        .min(2048);
    let mut usage = vk::ImageUsageFlags::SAMPLED;
    if color_features.contains(vk::FormatFeatureFlags2::COLOR_ATTACHMENT) {
        usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    let mut info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width: probe_size,
            height: probe_size,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    if image_format_properties(device, &info).is_none() {
        info.usage = vk::ImageUsageFlags::SAMPLED;
        if image_format_properties(device, &info).is_none() {
            return false;
        }
    }
    let requirements = image_memory_requirements(device, &info);
    let mut memory_type_bits = requirements.memory_type_bits;
    device.texture_heap_alignment = device.texture_heap_alignment.max(requirements.alignment);
    let broad = TextureUsage::SAMPLED | TextureUsage::STORAGE | TextureUsage::TRANSFER_DESTINATION;
    if device.format_features[Format::Rgba32Float as usize]
        .contains(required_format_features(broad))
    {
        let size = device
            .physical_properties
            .limits
            .max_image_dimension3_d
            .min(2048);
        info.image_type = vk::ImageType::TYPE_3D;
        info.format = vk::Format::R32G32B32A32_SFLOAT;
        info.extent = vk::Extent3D {
            width: size,
            height: size,
            depth: size.min(4),
        };
        info.mip_levels = 32 - size.leading_zeros();
        info.usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_DST;
        if image_format_properties(device, &info).is_some() {
            device.texture_heap_alignment = device
                .texture_heap_alignment
                .max(image_memory_requirements(device, &info).alignment);
        }
    }
    for format in [
        Format::D16Unorm,
        Format::D24UnormS8Uint,
        Format::D32Float,
        Format::S8Uint,
        Format::D32FloatS8Uint,
    ] {
        let features = device.format_features[format as usize];
        let format_info = get_texture_format_info(format);
        let combined = format_info.depth && format_info.stencil;
        let compatibility_usage = if features.contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE) {
            vk::ImageUsageFlags::SAMPLED
        } else if features.contains(vk::FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT) {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else if features.contains(required_format_features(TextureUsage::STORAGE)) {
            vk::ImageUsageFlags::STORAGE
        } else if !combined && features.contains(vk::FormatFeatureFlags2::TRANSFER_SRC) {
            vk::ImageUsageFlags::TRANSFER_SRC
        } else if !combined && features.contains(vk::FormatFeatureFlags2::TRANSFER_DST) {
            vk::ImageUsageFlags::TRANSFER_DST
        } else {
            continue;
        };
        info.image_type = vk::ImageType::TYPE_2D;
        info.format = format.vk();
        info.extent = vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        };
        info.mip_levels = 1;
        info.usage = compatibility_usage;
        let Some(properties) = image_format_properties(device, &info) else {
            continue;
        };
        info.extent = vk::Extent3D {
            width: 512,
            height: 512,
            depth: 1,
        };
        if features.contains(
            vk::FormatFeatureFlags2::SAMPLED_IMAGE
                | vk::FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT,
        ) {
            info.usage =
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        let mut supported = if info.usage == compatibility_usage {
            fits_image_format_properties(&info, &properties)
        } else {
            image_format_properties(device, &info).is_some()
        };
        if !supported && info.usage != compatibility_usage {
            info.usage = compatibility_usage;
            supported = fits_image_format_properties(&info, &properties);
        }
        if !supported {
            info.extent = vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            };
            info.usage = compatibility_usage;
        }
        let requirements = image_memory_requirements(device, &info);
        memory_type_bits &= requirements.memory_type_bits;
        device.texture_heap_alignment = device.texture_heap_alignment.max(requirements.alignment);
    }
    match device.find_memory_type(
        memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
        1,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    ) {
        Some(index) => {
            device.texture_memory_type = index;
            true
        }
        None => false,
    }
}

impl Device {
    unsafe fn create_command_contexts(&mut self) -> Result<()> {
        let mut timeline =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        self.command_retirement = self.vk().create_semaphore(
            &vk::SemaphoreCreateInfo::default().push(&mut timeline),
            None,
        )?;
        for _ in 0..INITIAL_COMMAND_CONTEXT_COUNT {
            self.grow_command_context_pool()?;
        }
        Ok(())
    }

    unsafe fn create_command_context(&mut self, context: &mut CommandBuffer) -> Result<()> {
        context.command_pool = self.vk().create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(self.queue_family),
            None,
        )?;
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(context.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let result = (self.vk().fp_v1_0().allocate_command_buffers)(
            self.vk().handle(),
            &allocation,
            &mut context.command_buffer,
        );
        if result != vk::Result::SUCCESS {
            self.destroy_command_context(context);
            return Err(result.into());
        }
        if self.timestamp_query_count != 0 {
            context.timestamp_pool = match self.vk().create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(self.timestamp_query_count),
                None,
            ) {
                Ok(pool) => pool,
                Err(error) => {
                    self.destroy_command_context(context);
                    return Err(error.into());
                }
            };
            context
                .timestamp_destinations
                .resize(self.timestamp_query_count as usize, 0);
        }
        self.allocations.push(Allocation::Commands(
            context.command_pool,
            context.timestamp_pool,
        ));
        Ok(())
    }

    unsafe fn grow_command_context_pool(&mut self) -> Result<()> {
        let mut context = CommandBuffer::new();
        self.create_command_context(&mut context)?;
        self.command_contexts.push_back(context);
        Ok(())
    }

    unsafe fn destroy_command_context(&self, context: &mut CommandBuffer) {
        if context.timestamp_pool != vk::QueryPool::null() {
            self.vk().destroy_query_pool(context.timestamp_pool, None);
        }
        if context.command_pool != vk::CommandPool::null() {
            self.vk().destroy_command_pool(context.command_pool, None);
        }
        *context = CommandBuffer::new();
    }

    unsafe fn destroy_command_contexts(&mut self) {
        self.command_contexts.clear();
        if self.command_retirement != vk::Semaphore::null() {
            self.vk().destroy_semaphore(self.command_retirement, None);
        }
        self.command_retirement = vk::Semaphore::null();
    }

    unsafe fn reset_command_context(&self, context: &mut CommandBuffer) {
        debug_assert!(context.retire_value <= self.completed_command_retirement);
        require(
            self.vk()
                .reset_command_pool(context.command_pool, vk::CommandPoolResetFlags::empty()),
        );
        context.retire_value = 0;
    }

    unsafe fn reset_retired_command_contexts(&mut self) {
        let completed = self.completed_command_retirement;
        let device = self.device.as_ref().unwrap();
        for context in &mut self.command_contexts {
            if context.retire_value != 0 && context.retire_value <= completed {
                require(
                    device.reset_command_pool(
                        context.command_pool,
                        vk::CommandPoolResetFlags::empty(),
                    ),
                );
                context.retire_value = 0;
            }
        }
    }

    unsafe fn acquire_command_context(&mut self) -> CommandBuffer {
        if self.active_command_buffers == 0 {
            self.poll_command_retirement();
        }
        let available = self
            .command_contexts
            .iter()
            .position(|context| context.retire_value <= self.completed_command_retirement);
        let mut context = if let Some(index) = available {
            self.command_contexts.remove(index).unwrap()
        } else {
            let mut context = CommandBuffer::new();
            require_error(self.create_command_context(&mut context));
            context
        };
        if context.retire_value != 0 {
            self.reset_command_context(&mut context);
        }
        context
    }

    unsafe fn collect_swapchains(&mut self) {
        if let (Some(device), Some(swapchains)) = (&self.device, &self.swapchain_api) {
            self.swapchain_delete_queue.collect(
                device,
                swapchains,
                self.completed_command_retirement,
            );
        }
    }

    unsafe fn poll_command_retirement(&mut self) {
        if self.command_retirement == vk::Semaphore::null()
            || self.completed_command_retirement == self.command_retirement_value
        {
            return;
        }
        let completed = require(
            self.vk()
                .get_semaphore_counter_value(self.command_retirement),
        );
        debug_assert!(
            completed >= self.completed_command_retirement
                && completed <= self.command_retirement_value
        );
        if completed == self.completed_command_retirement {
            return;
        }
        self.completed_command_retirement = completed;
        self.reset_retired_command_contexts();
        self.collect_swapchains();
    }

    unsafe fn wait_command_retirement(&mut self, value: u64) {
        debug_assert!(value <= self.command_retirement_value);
        if value > self.completed_command_retirement {
            require(
                self.vk().wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.command_retirement])
                        .values(&[value]),
                    u64::MAX,
                ),
            );
            self.completed_command_retirement = value;
        }
        self.reset_retired_command_contexts();
        self.collect_swapchains();
    }

    unsafe fn next_command_retirement(&mut self) -> u64 {
        let next = self.command_retirement_value + 1;
        if next - self.completed_command_retirement > self.max_timeline_value_difference {
            self.poll_command_retirement();
            if next - self.completed_command_retirement > self.max_timeline_value_difference {
                self.wait_command_retirement(next - self.max_timeline_value_difference);
            }
        }
        self.command_retirement_value = next;
        next
    }

    unsafe fn create_present_context(&mut self, index: usize) -> Result<()> {
        self.present_contexts[index].acquired = self
            .vk()
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
        self.present_contexts[index].rendered = self
            .vk()
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
        self.present_contexts[index].presented = self
            .vk()
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        Ok(())
    }

    unsafe fn destroy_present_context(&mut self, index: usize) {
        let context = std::mem::take(&mut self.present_contexts[index]);
        debug_assert!(!context.present_pending);
        if context.acquired != vk::Semaphore::null() {
            self.vk().destroy_semaphore(context.acquired, None);
        }
        if context.rendered != vk::Semaphore::null() {
            self.vk().destroy_semaphore(context.rendered, None);
        }
        if context.presented != vk::Fence::null() {
            self.vk().destroy_fence(context.presented, None);
        }
    }

    unsafe fn finish_present_context(&mut self, index: usize) {
        debug_assert!(!self.present_contexts[index].present_pending);
        let completed = std::mem::replace(
            &mut self.present_contexts[index].swapchain,
            vk::SwapchainKHR::null(),
        );
        for slot in 0..MAX_SWAPCHAIN_IMAGES {
            if self.retired_swapchains[slot].handle != completed {
                continue;
            }
            if self
                .present_contexts
                .iter()
                .any(|pending| pending.present_pending && pending.swapchain == completed)
            {
                return;
            }
            let retired = std::mem::take(&mut self.retired_swapchains[slot]);
            self.queue_retired_swapchain(retired);
            return;
        }
    }

    unsafe fn wait_present_context(&mut self, index: usize) {
        if !self.present_contexts[index].present_pending {
            return;
        }
        require(self.vk().wait_for_fences(
            &[self.present_contexts[index].presented],
            true,
            u64::MAX,
        ));
        self.present_contexts[index].present_pending = false;
        self.finish_present_context(index);
    }

    unsafe fn poll_present_contexts(&mut self) {
        for index in 0..self.present_context_count {
            if self.present_contexts[index].present_pending
                && require(
                    self.vk()
                        .get_fence_status(self.present_contexts[index].presented),
                )
            {
                self.present_contexts[index].present_pending = false;
                self.finish_present_context(index);
            }
        }
    }

    unsafe fn queue_retired_swapchain(&mut self, retired: RetiredSwapchain) {
        debug_assert!(retired.handle != vk::SwapchainKHR::null() && retired.view_count != 0);
        debug_assert_eq!(self.active_command_buffers, 0);
        for &view in &retired.views[..retired.view_count] {
            self.swapchain_delete_queue
                .push(self.command_retirement_value, retired.handle, view);
        }
        self.collect_swapchains();
    }

    unsafe fn drain_contexts(&mut self) {
        self.wait_command_retirement(self.command_retirement_value);
        for index in 0..self.present_context_count {
            self.wait_present_context(index);
        }
        debug_assert!(
            self.retired_swapchains
                .iter()
                .all(|retired| retired.handle == vk::SwapchainKHR::null())
        );
        self.collect_swapchains();
    }
}

struct PreparedTexture {
    view_formats: [vk::Format; FORMAT_COUNT],
    view_format_count: usize,
    format_list: vk::ImageFormatListCreateInfo<'static>,
    image_info: vk::ImageCreateInfo<'static>,
}

impl PreparedTexture {
    fn new(device: &Device, desc: &TextureDesc) -> Self {
        let mut view_formats = [vk::Format::UNDEFINED; FORMAT_COUNT];
        view_formats[0] = desc.format.vk();
        let mut count = 1;
        if desc.mutable_format {
            for format in FORMATS {
                if format == desc.format || !compatible_view_formats(desc.format, format) {
                    continue;
                }
                let features = device.format_features[format as usize];
                let sampled = desc.usage.intersects(TextureUsage::SAMPLED)
                    && features.contains(required_format_features(TextureUsage::SAMPLED));
                let storage = desc.usage.intersects(TextureUsage::STORAGE)
                    && features.contains(required_format_features(TextureUsage::STORAGE));
                if sampled || storage {
                    view_formats[count] = format.vk();
                    count += 1;
                }
            }
        }
        let mut flags = vk::ImageCreateFlags::empty();
        if matches!(desc.r#type, TextureType::Cube | TextureType::CubeArray) {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        if count > 1 {
            flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        Self {
            view_formats,
            view_format_count: count,
            format_list: Default::default(),
            image_info: vk::ImageCreateInfo::default()
                .flags(flags)
                .image_type(desc.r#type.vk())
                .format(desc.format.vk())
                .extent(vk::Extent3D {
                    width: desc.extent.x,
                    height: desc.extent.y,
                    depth: desc.extent.z,
                })
                .mip_levels(desc.mip_levels)
                .array_layers(desc.layer_count)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(image_usage(desc.usage))
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        }
    }

    fn info(&mut self) -> &vk::ImageCreateInfo<'_> {
        self.format_list.view_format_count = self.view_format_count as u32;
        self.format_list.p_view_formats = self.view_formats.as_ptr();
        self.image_info.p_next = if self.view_format_count > 1 {
            ptr::from_ref(&self.format_list).cast()
        } else {
            ptr::null()
        };
        &self.image_info
    }
}

unsafe fn retire_swapchain_handle(device: &mut Device, swapchain: &mut Swapchain) {
    if swapchain.handle == vk::SwapchainKHR::null() {
        debug_assert_eq!(swapchain.image_count, 0);
        swapchain.width = 0;
        swapchain.height = 0;
        return;
    }
    device.poll_present_contexts();
    let mut retired = RetiredSwapchain {
        handle: swapchain.handle,
        view_count: swapchain.image_count,
        ..Default::default()
    };
    for index in 0..retired.view_count {
        retired.views[index] = swapchain.render_views[index].as_ref().unwrap().view;
        swapchain.render_views[index] = None;
        swapchain.images[index] = vk::Image::null();
        swapchain.initialized[index] = false;
    }
    swapchain.image_count = 0;
    swapchain.handle = vk::SwapchainKHR::null();
    swapchain.width = 0;
    swapchain.height = 0;
    if !device
        .present_contexts
        .iter()
        .any(|context| context.present_pending && context.swapchain == retired.handle)
    {
        device.queue_retired_swapchain(retired);
        return;
    }
    for slot in &mut device.retired_swapchains {
        if slot.handle == vk::SwapchainKHR::null() {
            *slot = retired;
            return;
        }
    }
    for index in 0..device.present_context_count {
        let context = device.present_contexts[index];
        if context.present_pending && context.swapchain == retired.handle {
            device.wait_present_context(index);
        }
    }
    device.queue_retired_swapchain(retired);
}

fn choose_composite_alpha(supported: vk::CompositeAlphaFlagsKHR) -> vk::CompositeAlphaFlagsKHR {
    for choice in [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ] {
        if supported.contains(choice) {
            return choice;
        }
    }
    panic!("surface exposes no composite alpha mode")
}

fn drawable_extent(device: &Device, capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX
        && capabilities.current_extent.height != u32::MAX
    {
        return capabilities.current_extent;
    }
    let size = device.window.as_ref().unwrap().inner_size();
    if size.width == 0 || size.height == 0 {
        return vk::Extent2D::default();
    }
    vk::Extent2D {
        width: size.width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: size.height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

unsafe fn swapchain_surface_configuration_changed(device: &Device, swapchain: &Swapchain) -> bool {
    let capabilities = require(
        device
            .surface_api
            .get_physical_device_surface_capabilities(device.physical_device, device.surface),
    );
    let extent = drawable_extent(device, &capabilities);
    extent.width != swapchain.width
        || extent.height != swapchain.height
        || capabilities.current_transform != swapchain.transform
        || choose_composite_alpha(capabilities.supported_composite_alpha)
            != swapchain.composite_alpha
}

unsafe fn recreate_swapchain(device: &mut Device, swapchain: &mut Swapchain) -> Result<()> {
    debug_assert!(
        !swapchain.acquired
            && !device
                .swapchain
                .as_ref()
                .is_some_and(|swapchain| swapchain.acquired)
            && device.active_command_buffers == 0
    );
    let mut mode = vk::SurfacePresentModeKHR::default().present_mode(SWAPCHAIN_PRESENT_MODE);
    let surface_info = vk::PhysicalDeviceSurfaceInfo2KHR::default()
        .push(&mut mode)
        .surface(device.surface);
    let mut capabilities_info = vk::SurfaceCapabilities2KHR::default();
    device
        .surface_caps_api
        .get_physical_device_surface_capabilities2(
            device.physical_device,
            &surface_info,
            &mut capabilities_info,
        )?;
    let capabilities = capabilities_info.surface_capabilities;
    let extent = drawable_extent(device, &capabilities);
    if extent.width == 0 || extent.height == 0 {
        swapchain.width = 0;
        swapchain.height = 0;
        swapchain.recreate_required = true;
        return Ok(());
    }
    if !capabilities
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    {
        return Err(Error::Unsupported);
    }
    let formats = device
        .surface_api
        .get_physical_device_surface_formats(device.physical_device, device.surface)?;
    if formats.is_empty() || formats.len() > 64 {
        return Err(Error::Unsupported);
    }
    let requested_format = swapchain.format.vk();
    if !formats.iter().any(|format| {
        (format.format == requested_format || format.format == vk::Format::UNDEFINED)
            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
    }) {
        return Err(Error::Unsupported);
    }
    let mut requested_count =
        (device.present_context_count as u32).max(capabilities.min_image_count);
    if capabilities.max_image_count != 0 {
        requested_count = requested_count.min(capabilities.max_image_count);
    }
    if requested_count == 0 || requested_count > MAX_SWAPCHAIN_IMAGES as u32 {
        return Err(Error::Unsupported);
    }
    let alpha = choose_composite_alpha(capabilities.supported_composite_alpha);
    let modes = [SWAPCHAIN_PRESENT_MODE];
    let mut modes_info = vk::SwapchainPresentModesCreateInfoKHR::default().present_modes(&modes);
    let info = vk::SwapchainCreateInfoKHR::default()
        .push(&mut modes_info)
        .surface(device.surface)
        .min_image_count(requested_count)
        .image_format(requested_format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(alpha)
        .present_mode(SWAPCHAIN_PRESENT_MODE)
        .clipped(true)
        .old_swapchain(swapchain.handle);
    let handle = match device.swapchains().create_swapchain(&info, None) {
        Ok(handle) => handle,
        Err(error) => {
            retire_swapchain_handle(device, swapchain);
            return Err(error.into());
        }
    };
    let images = match device.swapchains().get_swapchain_images(handle) {
        Ok(images) if !images.is_empty() && images.len() <= MAX_SWAPCHAIN_IMAGES => images,
        result => {
            device.swapchains().destroy_swapchain(handle, None);
            retire_swapchain_handle(device, swapchain);
            return Err(match result {
                Err(error) => error.into(),
                _ => Error::Unsupported,
            });
        }
    };
    let mut views = [vk::ImageView::null(); MAX_SWAPCHAIN_IMAGES];
    for (index, &image) in images.iter().enumerate() {
        let view = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(requested_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        views[index] = match device.vk().create_image_view(&view, None) {
            Ok(view) => view,
            Err(error) => {
                for &view in &views[..index] {
                    device.vk().destroy_image_view(view, None);
                }
                device.swapchains().destroy_swapchain(handle, None);
                retire_swapchain_handle(device, swapchain);
                return Err(error.into());
            }
        };
    }
    retire_swapchain_handle(device, swapchain);
    swapchain.handle = handle;
    swapchain.image_count = images.len();
    swapchain.width = extent.width;
    swapchain.height = extent.height;
    swapchain.transform = capabilities.current_transform;
    swapchain.composite_alpha = alpha;
    swapchain.recreate_required = false;
    for (index, &image) in images.iter().enumerate() {
        swapchain.images[index] = image;
        swapchain.render_views[index] = Some(Arc::new(RenderView {
            view: views[index],
            width: extent.width,
            height: extent.height,
            swapchain_view: true,
        }));
    }
    Ok(())
}

unsafe fn create_raster_pso(
    device: &mut Device,
    first_stage_spirv: &[u32],
    fragment_spirv: &[u32],
    color_targets: &[ColorTargetDesc],
    depth_format: Format,
    stencil_format: Format,
    rasterization_state: &RasterizationState,
    mesh: bool,
) -> Result<PSO> {
    debug_assert!(color_targets.len() <= MAX_COLOR_ATTACHMENTS);
    let mut first_module = vk::ShaderModuleCreateInfo::default().code(first_stage_spirv);
    let mut fragment_module = vk::ShaderModuleCreateInfo::default().code(fragment_spirv);
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .push(&mut first_module)
            .stage(if mesh {
                vk::ShaderStageFlags::MESH_EXT
            } else {
                vk::ShaderStageFlags::VERTEX
            })
            .name(if mesh { c"meshMain" } else { c"vertexMain" }),
        vk::PipelineShaderStageCreateInfo::default()
            .push(&mut fragment_module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(c"fragmentMain"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default();
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(if rasterization_state.cull == CullMode::None {
            vk::CullModeFlags::NONE
        } else {
            vk::CullModeFlags::BACK
        })
        .front_face(if rasterization_state.cull == CullMode::CounterClockwise {
            vk::FrontFace::CLOCKWISE
        } else {
            vk::FrontFace::COUNTER_CLOCKWISE
        })
        .depth_bias_enable(
            rasterization_state.depth_bias_constant != 0.0
                || rasterization_state.depth_bias_clamp != 0.0
                || rasterization_state.depth_bias_slope != 0.0,
        )
        .depth_bias_constant_factor(rasterization_state.depth_bias_constant)
        .depth_bias_clamp(rasterization_state.depth_bias_clamp)
        .depth_bias_slope_factor(rasterization_state.depth_bias_slope)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
    let mut attachments = [vk::PipelineColorBlendAttachmentState::default(); MAX_COLOR_ATTACHMENTS];
    let mut formats = [vk::Format::UNDEFINED; MAX_COLOR_ATTACHMENTS];
    for (index, target) in color_targets.iter().enumerate() {
        attachments[index] = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(target.blend.enabled)
            .src_color_blend_factor(target.blend.color.source.vk())
            .dst_color_blend_factor(target.blend.color.destination.vk())
            .color_blend_op(vk::BlendOp::from_raw(target.blend.color.operation as i32))
            .src_alpha_blend_factor(target.blend.alpha.source.vk())
            .dst_alpha_blend_factor(target.blend.alpha.destination.vk())
            .alpha_blend_op(vk::BlendOp::from_raw(target.blend.alpha.operation as i32))
            .color_write_mask(vk::ColorComponentFlags::from_raw(target.write_mask as u32));
        formats[index] = target.format.vk();
    }
    let blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&attachments[..color_targets.len()]);
    let dynamic_states = [
        vk::DynamicState::VIEWPORT_WITH_COUNT,
        vk::DynamicState::SCISSOR_WITH_COUNT,
        vk::DynamicState::DEPTH_TEST_ENABLE,
        vk::DynamicState::DEPTH_WRITE_ENABLE,
        vk::DynamicState::DEPTH_COMPARE_OP,
        vk::DynamicState::STENCIL_TEST_ENABLE,
        vk::DynamicState::STENCIL_OP,
        vk::DynamicState::STENCIL_COMPARE_MASK,
        vk::DynamicState::STENCIL_WRITE_MASK,
        vk::DynamicState::STENCIL_REFERENCE,
    ];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats[..color_targets.len()])
        .depth_attachment_format(depth_format.vk())
        .stencil_attachment_format(stencil_format.vk());
    let mut flags = vk::PipelineCreateFlags2CreateInfo::default()
        .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .push(&mut flags)
        .push(&mut rendering)
        .stages(&stages[..if fragment_spirv.is_empty() { 1 } else { 2 }])
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .base_pipeline_index(-1);
    if !mesh {
        info = info
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly);
    }
    let mut pipeline = vk::Pipeline::null();
    let result = (device.vk().fp_v1_0().create_graphics_pipelines)(
        device.vk().handle(),
        vk::PipelineCache::null(),
        1,
        &info,
        ptr::null(),
        &mut pipeline,
    );
    if result != vk::Result::SUCCESS {
        if pipeline != vk::Pipeline::null() {
            device.vk().destroy_pipeline(pipeline, None);
        }
        return Err(result.into());
    }
    device.allocations.push(Allocation::Pipeline(pipeline));
    Ok(PSO {
        pso: pipeline,
        bind_point: vk::PipelineBindPoint::GRAPHICS,
    })
}

unsafe fn record_image_barriers(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    barriers: &[vk::ImageMemoryBarrier2<'_>],
) {
    debug_assert!(!barriers.is_empty());
    device.vk().cmd_pipeline_barrier2(
        command_buffer,
        &vk::DependencyInfo::default().image_memory_barriers(barriers),
    );
}

unsafe fn submit_commands(
    mut commands: Vec<CommandBuffer>,
    device: &mut Device,
    completion: &TimelinePoint,
    wait_semaphore: vk::Semaphore,
    signal_semaphore: vk::Semaphore,
) {
    debug_assert!(!commands.is_empty() && device.active_command_buffers == commands.len());
    debug_assert!(device.pending_texture_initializations.is_empty());
    device
        .command_submit_infos
        .resize(commands.len(), vk::CommandBufferSubmitInfo::default());
    for (index, current) in commands.iter_mut().enumerate() {
        for timestamp in 0..current.timestamp_count {
            let destination = vk::StridedDeviceAddressRangeKHR {
                address: current.timestamp_destinations[timestamp as usize],
                size: 8,
                stride: 8,
            };
            (device.functions().cmd_copy_query_pool_results_to_memory)(
                current.command_buffer,
                current.timestamp_pool,
                timestamp,
                1,
                &destination,
                ADDRESS_FLAGS,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
        if current.timestamp_count != 0 {
            device.barrier(
                current,
                Stage::TRANSFER,
                Access::TRANSFER_WRITE,
                Stage::HOST,
                Access::HOST_READ,
            );
        }
        require(device.vk().end_command_buffer(current.command_buffer));
        device.active_command_buffers -= 1;
        device.command_submit_infos[index] = vk::CommandBufferSubmitInfo::default()
            .command_buffer(current.command_buffer)
            .device_mask(1);
    }
    let retirement = device.next_command_retirement();
    let wait = [vk::SemaphoreSubmitInfo::default()
        .semaphore(wait_semaphore)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let signals = [
        vk::SemaphoreSubmitInfo::default()
            .semaphore(completion.semaphore.semaphore)
            .value(completion.value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        vk::SemaphoreSubmitInfo::default()
            .semaphore(signal_semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
    ];
    let retirement_signal = [vk::SemaphoreSubmitInfo::default()
        .semaphore(device.command_retirement)
        .value(retirement)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    // A separate submission retires the command buffers after the public completion signal.
    let submissions = [
        vk::SubmitInfo2::default()
            .wait_semaphore_infos(if wait_semaphore == vk::Semaphore::null() {
                &[]
            } else {
                &wait
            })
            .command_buffer_infos(&device.command_submit_infos[..commands.len()])
            .signal_semaphore_infos(
                &signals[..if signal_semaphore == vk::Semaphore::null() {
                    1
                } else {
                    2
                }],
            ),
        vk::SubmitInfo2::default().signal_semaphore_infos(&retirement_signal),
    ];
    require(
        device
            .vk()
            .queue_submit2(device.queue, &submissions, vk::Fence::null()),
    );
    for mut current in commands {
        current.retire_value = retirement;
        current.swapchain = false;
        device.command_contexts.push_back(current);
    }
}

fn make_heap_bind_info(
    heap: GpuRange,
    reserved_alignment: u64,
    reserved_size: u64,
) -> vk::BindHeapInfoEXT<'static> {
    let offset = align_up(heap.size, reserved_alignment);
    vk::BindHeapInfoEXT::default()
        .heap_range(vk::DeviceAddressRangeEXT {
            address: heap.gpu,
            size: offset + reserved_size,
        })
        .reserved_range_offset(offset)
        .reserved_range_size(reserved_size)
}

unsafe fn emit_root_data(device: &Device, commands: &mut CommandBuffer, root: &[u8]) {
    if root.is_empty() {
        return;
    }
    let info = vk::PushDataInfoEXT::default().data(vk::HostAddressRangeConstEXT {
        address: root.as_ptr().cast(),
        size: root.len(),
        ..Default::default()
    });
    (device.functions().cmd_push_data)(commands.command_buffer, &info);
}

fn make_texture_copy_region(
    texture: &Texture,
    copy: &TextureCopyDesc,
    memory: GpuRange,
) -> vk::DeviceMemoryImageCopyKHR<'static> {
    let mip_width = (texture.width >> copy.mip_level).max(1);
    let mip_height = (texture.height >> copy.mip_level).max(1);
    let mip_depth = (texture.depth >> copy.mip_level).max(1);
    let width = if copy.extent.x == 0 {
        mip_width - copy.offset.x
    } else {
        copy.extent.x
    };
    let format = get_texture_format_info(texture.format);
    let row_pitch = if copy.row_pitch_bytes == 0 {
        (width as u64).div_ceil(format.block_extent.x as u64) * format.bytes_per_block as u64
    } else {
        copy.row_pitch_bytes
    };
    debug_assert!(copy.slice_pitch_bytes == 0 || row_pitch != 0);
    vk::DeviceMemoryImageCopyKHR::default()
        .address_range(vk::DeviceAddressRangeKHR {
            address: memory.gpu,
            size: memory.size,
        })
        .address_flags(ADDRESS_FLAGS)
        .address_row_length(
            (copy.row_pitch_bytes / format.bytes_per_block as u64 * format.block_extent.x as u64)
                as u32,
        )
        .address_image_height(if copy.slice_pitch_bytes == 0 {
            0
        } else {
            (copy.slice_pitch_bytes / row_pitch * format.block_extent.y as u64) as u32
        })
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(image_aspects(texture.format))
                .mip_level(copy.mip_level)
                .base_array_layer(copy.base_slice)
                .layer_count(if copy.slice_count == 0 {
                    texture.layer_count - copy.base_slice
                } else {
                    copy.slice_count
                }),
        )
        .image_layout(vk::ImageLayout::GENERAL)
        .image_offset(vk::Offset3D {
            x: copy.offset.x as i32,
            y: copy.offset.y as i32,
            z: copy.offset.z as i32,
        })
        .image_extent(vk::Extent3D {
            width,
            height: if copy.extent.y == 0 {
                mip_height - copy.offset.y
            } else {
                copy.extent.y
            },
            depth: if copy.extent.z == 0 {
                mip_depth - copy.offset.z
            } else {
                copy.extent.z
            },
        })
}

impl Device {
    /// # Safety
    /// `initial_value` must satisfy the device's timeline semaphore limits.
    pub unsafe fn create_timeline_semaphore(
        &mut self,
        initial_value: u64,
    ) -> Result<Arc<TimelineSemaphore>> {
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let semaphore = self.vk().create_semaphore(
            &vk::SemaphoreCreateInfo::default().push(&mut timeline),
            None,
        )?;
        self.allocations.push(Allocation::Semaphore(semaphore));
        Ok(Arc::new(TimelineSemaphore { semaphore }))
    }

    /// # Safety
    /// The semaphore must belong to this device and have no pending GPU uses.
    pub unsafe fn destroy_timeline_semaphore(&mut self, semaphore: Arc<TimelineSemaphore>) {
        let semaphore = Arc::try_unwrap(semaphore).expect("semaphore is still shared");
        self.release(Allocation::Semaphore(semaphore.semaphore));
    }

    /// # Safety
    /// The semaphore must belong to this device and must not have been destroyed.
    pub unsafe fn timeline_completed_value(&self, semaphore: &TimelineSemaphore) -> u64 {
        require(self.vk().get_semaphore_counter_value(semaphore.semaphore))
    }

    /// # Safety
    /// The semaphore must belong to this device; a submitted signal must be able to reach the value.
    pub unsafe fn wait_timeline(&mut self, point: &TimelinePoint) {
        require(
            self.vk().wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[point.semaphore.semaphore])
                    .values(&[point.value]),
                u64::MAX,
            ),
        );
        self.poll_command_retirement();
    }

    /// # Safety
    /// Every begun command buffer and acquired frame must have been submitted.
    pub unsafe fn wait_idle(&mut self) {
        debug_assert!(
            self.active_command_buffers == 0
                && !self
                    .swapchain
                    .as_ref()
                    .is_some_and(|swapchain| swapchain.acquired)
        );
        self.drain_contexts();
        self.next_present_context = 0;
    }

    pub fn get_device_caps(&self) -> &DeviceCaps {
        &self.caps
    }

    pub fn supports_texture_format(&self, format: Format, usage: TextureUsage) -> bool {
        debug_assert!(format != Format::Undefined && usage.0 != 0 && usage.0 & !0x3f == 0);
        let info = get_texture_format_info(format);
        if (info.depth || info.stencil) && usage.intersects(TextureUsage::COLOR_ATTACHMENT) {
            return false;
        }
        if !info.depth && !info.stencil && usage.intersects(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
        {
            return false;
        }
        if info.depth
            && info.stencil
            && usage.intersects(TextureUsage::TRANSFER_SOURCE | TextureUsage::TRANSFER_DESTINATION)
        {
            return false;
        }
        match format {
            Format::EacRg if !self.texture_compression_etc2 => return false,
            Format::Astc4x4Srgb | Format::Astc4x4Unorm if !self.caps.texture_compression_astc => {
                return false;
            }
            Format::Bc3Srgb
            | Format::Bc3Unorm
            | Format::Bc5Rg
            | Format::Bc7Srgb
            | Format::Bc7Unorm
                if !self.caps.texture_compression_bc =>
            {
                return false;
            }
            _ => (),
        }
        self.format_features[format as usize].contains(required_format_features(usage))
    }

    /// Allocates a raw GPU block. Descriptor heap ranges include exactly the requested usable
    /// bytes; implementation-reserved storage is appended outside the returned range.
    /// # Safety
    /// The allocation size must be nonzero and satisfy Vulkan buffer and memory limits.
    pub unsafe fn create_gpu_heap(
        &mut self,
        byte_count: u64,
        memory: MemoryType,
    ) -> Result<GpuHeap> {
        let descriptor = matches!(
            memory,
            MemoryType::TextureDescriptorHeap | MemoryType::SamplerDescriptorHeap
        );
        let mut allocation_alignment = 1;
        let (size, usage, required, preferred, avoided) = if descriptor {
            let texture = memory == MemoryType::TextureDescriptorHeap;
            let p = &self.heap_properties;
            let reserved_alignment = if texture {
                p.image_descriptor_alignment
                    .max(p.buffer_descriptor_alignment)
            } else {
                p.sampler_descriptor_alignment
            };
            let heap_alignment = if texture {
                p.resource_heap_alignment
            } else {
                p.sampler_heap_alignment
            };
            let reserved_size = if texture {
                p.min_resource_heap_reserved_range
            } else {
                p.min_sampler_heap_reserved_range
            };
            allocation_alignment = heap_alignment.max(GPU_ALLOCATION_ALIGNMENT);
            (
                align_up(byte_count, reserved_alignment) + reserved_size + allocation_alignment - 1,
                UNIVERSAL_BUFFER_USAGE | vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT,
                CPU_VISIBLE_MEMORY_PROPERTIES,
                vk::MemoryPropertyFlags::empty(),
                vk::MemoryPropertyFlags::empty(),
            )
        } else {
            let (required, preferred, avoided) = match memory {
                MemoryType::CpuVisible => (
                    CPU_VISIBLE_MEMORY_PROPERTIES,
                    vk::MemoryPropertyFlags::empty(),
                    vk::MemoryPropertyFlags::empty(),
                ),
                MemoryType::GpuOnly => (
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::empty(),
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                ),
                MemoryType::Readback => (
                    CPU_VISIBLE_MEMORY_PROPERTIES,
                    vk::MemoryPropertyFlags::HOST_CACHED,
                    vk::MemoryPropertyFlags::empty(),
                ),
                _ => unreachable!(),
            };
            (
                byte_count,
                UNIVERSAL_BUFFER_USAGE,
                required,
                preferred,
                avoided,
            )
        };
        let backing = self.create_backing_buffer(size, usage, required, preferred, avoided)?;
        let gpu = align_up(backing.address, allocation_alignment);
        let mapped_offset = (gpu - backing.address) as usize;
        self.allocations.push(Allocation::Buffer(
            backing.buffer,
            backing.memory,
            backing.mapped.is_some(),
        ));
        Ok(GpuHeap {
            range: GpuRange {
                gpu,
                size: byte_count,
            },
            backing,
            mapped_offset,
        })
    }

    /// # Safety
    /// The heap must belong to this device; all recorded and executing uses must have finished.
    pub unsafe fn destroy_gpu_heap(&mut self, heap: GpuHeap) {
        self.release(Allocation::Buffer(
            heap.backing.buffer,
            heap.backing.memory,
            heap.backing.mapped.is_some(),
        ));
    }

    /// # Safety
    /// The allocation size must be nonzero and satisfy the selected memory type's limits.
    pub unsafe fn create_texture_heap(&mut self, byte_count: u64) -> Result<TextureHeap> {
        let memory = self.vk().allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(byte_count)
                .memory_type_index(self.texture_memory_type),
            None,
        )?;
        self.allocations.push(Allocation::Memory(memory));
        Ok(TextureHeap {
            size: byte_count,
            memory,
        })
    }

    /// # Safety
    /// The heap must belong to this device; its images must be destroyed and GPU uses finished.
    pub unsafe fn destroy_texture_heap(&mut self, heap: TextureHeap) {
        self.release(Allocation::Memory(heap.memory));
    }

    /// # Safety
    /// The texture description must satisfy this device's format, extent, and usage limits.
    pub unsafe fn get_texture_size_align(&self, desc: &TextureDesc) -> SizeAlign {
        let mut texture = PreparedTexture::new(self, desc);
        let requirements = image_memory_requirements(self, texture.info());
        SizeAlign {
            size: requirements.size,
            align: requirements.alignment,
        }
    }

    /// # Safety
    /// The heap must belong to this device. The description and aligned offset must fit its allocation;
    /// no recording may be active, and overlapping allocations must be synchronized.
    pub unsafe fn create_texture(
        &mut self,
        desc: &TextureDesc,
        heap: &TextureHeap,
        offset: u64,
    ) -> Result<Texture> {
        debug_assert_eq!(self.active_command_buffers, 0);
        let mut prepared = PreparedTexture::new(self, desc);
        let image = self.vk().create_image(prepared.info(), None)?;
        if let Err(error) = self.vk().bind_image_memory(image, heap.memory, offset) {
            self.vk().destroy_image(image, None);
            return Err(error.into());
        }
        self.allocations.push(Allocation::Image(image));
        let result = Texture {
            image,
            width: desc.extent.x,
            height: desc.extent.y,
            depth: desc.extent.z,
            layer_count: desc.layer_count,
            r#type: desc.r#type,
            format: desc.format,
        };
        self.pending_texture_initializations
            .push_back(TextureInitialization {
                image,
                aspect_mask: image_aspects(desc.format),
                mip_levels: desc.mip_levels,
                array_layers: desc.layer_count,
            });
        Ok(result)
    }

    /// # Safety
    /// The texture must belong to this device. Destroy its views and finish all recorded and GPU uses first.
    pub unsafe fn destroy_texture(&mut self, texture: Texture) {
        self.pending_texture_initializations
            .retain(|initialization| initialization.image != texture.image);
        self.release(Allocation::Image(texture.image));
    }

    /// # Safety
    /// The texture must belong to this device, and the requested mip and slice must exist.
    pub unsafe fn create_render_view(
        &mut self,
        texture: &Texture,
        desc: &RenderViewDesc,
    ) -> Result<Arc<RenderView>> {
        let info = vk::ImageViewCreateInfo::default()
            .image(texture.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(texture.format.vk())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_aspects(texture.format))
                    .base_mip_level(desc.mip_level)
                    .level_count(1)
                    .base_array_layer(desc.slice)
                    .layer_count(1),
            );
        let view = self.vk().create_image_view(&info, None)?;
        self.allocations.push(Allocation::View(view));
        Ok(Arc::new(RenderView {
            view,
            width: (texture.width >> desc.mip_level).max(1),
            height: (texture.height >> desc.mip_level).max(1),
            swapchain_view: false,
        }))
    }

    /// # Safety
    /// The view must belong to this device and have no pending recorded or executing uses.
    pub unsafe fn destroy_render_view(&mut self, view: Arc<RenderView>) {
        let view = Arc::try_unwrap(view).expect("render view is still shared");
        assert!(!view.swapchain_view, "swapchain views belong to the device");
        self.release(Allocation::View(view.view));
    }

    /// # Safety
    /// The texture must belong to this device and support the requested view. The destination must
    /// satisfy descriptor alignment requirements and have no concurrent GPU or host accesses.
    pub unsafe fn write_texture_descriptor(
        &mut self,
        cpu_destination: &mut [std::mem::MaybeUninit<u8>],
        texture: &Texture,
        descriptor_type: TextureDescriptorType,
        desc: &TextureDescriptorDesc,
    ) {
        let sampled = descriptor_type == TextureDescriptorType::Sampled;
        let format_info = get_texture_format_info(texture.format);
        let aspect = match desc.aspect {
            TextureAspect::Automatic if format_info.depth => vk::ImageAspectFlags::DEPTH,
            TextureAspect::Automatic if format_info.stencil => vk::ImageAspectFlags::STENCIL,
            TextureAspect::Automatic | TextureAspect::Color => vk::ImageAspectFlags::COLOR,
            TextureAspect::Depth => vk::ImageAspectFlags::DEPTH,
            TextureAspect::Stencil => vk::ImageAspectFlags::STENCIL,
        };
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(if sampled {
            vk::ImageUsageFlags::SAMPLED
        } else {
            vk::ImageUsageFlags::STORAGE
        });
        let view = vk::ImageViewCreateInfo::default()
            .push(&mut usage)
            .image(texture.image)
            .view_type(texture.r#type.vk_view())
            .format(
                if desc.format == Format::Undefined {
                    texture.format
                } else {
                    desc.format
                }
                .vk(),
            )
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .base_mip_level(desc.base_mip)
                    .level_count(if desc.mip_count == 0 {
                        vk::REMAINING_MIP_LEVELS
                    } else {
                        desc.mip_count
                    })
                    .base_array_layer(desc.base_layer)
                    .layer_count(if desc.layer_count == 0 {
                        vk::REMAINING_ARRAY_LAYERS
                    } else {
                        desc.layer_count
                    }),
            );
        let image = vk::ImageDescriptorInfoEXT::default()
            .view(&view)
            .layout(vk::ImageLayout::GENERAL);
        let descriptor = vk::ResourceDescriptorInfoEXT::default()
            .ty(if sampled {
                vk::DescriptorType::SAMPLED_IMAGE
            } else {
                vk::DescriptorType::STORAGE_IMAGE
            })
            .data(vk::ResourceDescriptorDataEXT { p_image: &image });
        assert!(cpu_destination.len() >= self.heap_properties.image_descriptor_size as usize);
        let destination = vk::HostAddressRangeEXT {
            address: cpu_destination.as_mut_ptr().cast(),
            size: self.heap_properties.image_descriptor_size as usize,
            ..Default::default()
        };
        require(
            (self.functions().write_resource_descriptors)(
                self.vk().handle(),
                1,
                &descriptor,
                &destination,
            )
            .result(),
        );
    }

    /// # Safety
    /// The sampler description and destination alignment must satisfy the device limits.
    /// The destination must have no concurrent GPU or host accesses.
    pub unsafe fn write_sampler_descriptor(
        &mut self,
        cpu_destination: &mut [std::mem::MaybeUninit<u8>],
        desc: &SamplerDesc,
    ) {
        let sampler = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::from_raw(desc.mag_filter as i32))
            .min_filter(vk::Filter::from_raw(desc.min_filter as i32))
            .mipmap_mode(vk::SamplerMipmapMode::from_raw(desc.mip_filter as i32))
            .address_mode_u(vk::SamplerAddressMode::from_raw(desc.address_u as i32))
            .address_mode_v(vk::SamplerAddressMode::from_raw(desc.address_v as i32))
            .address_mode_w(vk::SamplerAddressMode::from_raw(desc.address_w as i32))
            .anisotropy_enable(desc.anisotropic)
            .max_anisotropy(if desc.anisotropic { 4.0 } else { 1.0 })
            .compare_enable(desc.compare_enabled)
            .compare_op(vk::CompareOp::from_raw(desc.compare as i32))
            .max_lod(vk::LOD_CLAMP_NONE);
        assert!(cpu_destination.len() >= self.heap_properties.sampler_descriptor_size as usize);
        let destination = vk::HostAddressRangeEXT {
            address: cpu_destination.as_mut_ptr().cast(),
            size: self.heap_properties.sampler_descriptor_size as usize,
            ..Default::default()
        };
        require(
            (self.functions().write_sampler_descriptors)(
                self.vk().handle(),
                1,
                &sampler,
                &destination,
            )
            .result(),
        );
    }

    /// # Safety
    /// No command recording or acquired frame may be outstanding.
    pub unsafe fn get_drawable_extent(&mut self) -> Uint32x2 {
        debug_assert!(
            self.active_command_buffers == 0
                && !self
                    .swapchain
                    .as_ref()
                    .is_some_and(|swapchain| swapchain.acquired)
        );
        if self.swapchain.is_none() {
            return Uint32x2::default();
        }

        let capabilities = require(
            self.surface_api
                .get_physical_device_surface_capabilities(self.physical_device, self.surface),
        );
        let extent = drawable_extent(self, &capabilities);
        let swapchain = self.swapchain.as_mut().unwrap();
        if swapchain.handle != vk::SwapchainKHR::null()
            && (swapchain.width != extent.width || swapchain.height != extent.height)
        {
            swapchain.recreate_required = true;
        }
        Uint32x2 {
            x: extent.width,
            y: extent.height,
        }
    }

    /// Returns `None` while the drawable extent is zero. Call from winit's redraw handling;
    /// use `get_drawable_extent` after resize events before beginning command recording.
    /// # Safety
    /// Call on the window's event-loop thread with no outstanding recording or acquired frame.
    pub unsafe fn acquire(&mut self) -> Option<SwapchainFrame> {
        debug_assert!(
            self.swapchain.is_some()
                && !self
                    .swapchain
                    .as_ref()
                    .is_some_and(|swapchain| swapchain.acquired)
                && self.active_command_buffers == 0
        );
        self.with_swapchain(|device, swapchain| {
            debug_assert!(!swapchain.acquired);
            let mut context = None;
            loop {
                if swapchain.handle == vk::SwapchainKHR::null() || swapchain.recreate_required {
                    require_error(recreate_swapchain(device, swapchain));
                    if swapchain.handle == vk::SwapchainKHR::null()
                        || swapchain.width == 0
                        || swapchain.height == 0
                    {
                        return None;
                    }
                }
                let index = *context.get_or_insert_with(|| {
                    let index = device.next_present_context;
                    device.wait_present_context(index);
                    index
                });
                let (image_index, suboptimal) = match device.swapchains().acquire_next_image(
                    swapchain.handle,
                    u64::MAX,
                    device.present_contexts[index].acquired,
                    vk::Fence::null(),
                ) {
                    Ok(result) => result,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        swapchain.recreate_required = true;
                        continue;
                    }
                    Err(error) => require(Err(error)),
                };
                debug_assert!((image_index as usize) < swapchain.image_count);
                swapchain.image_index = image_index;
                swapchain.present_context = Some(index);
                swapchain.acquired = true;
                swapchain.recreate_required =
                    suboptimal && swapchain_surface_configuration_changed(device, swapchain);
                let frame = SwapchainFrame {
                    render_view: Arc::clone(
                        swapchain.render_views[image_index as usize]
                            .as_ref()
                            .unwrap(),
                    ),
                    extent: Uint32x2 {
                        x: swapchain.width,
                        y: swapchain.height,
                    },
                };

                return Some(frame);
            }
        })
    }

    /// # Safety
    /// The shader modules, interfaces, attachment formats, and rasterization state must satisfy Vulkan requirements.
    pub unsafe fn create_graphics_pso(&mut self, desc: &GraphicsPSODesc<'_>) -> Result<PSO> {
        create_raster_pso(
            self,
            desc.vertex_spirv,
            desc.fragment_spirv,
            desc.color_targets,
            desc.depth_format,
            desc.stencil_format,
            &desc.rasterization,
            false,
        )
    }

    /// # Safety
    /// The mesh and fragment shaders, interfaces, attachment formats, and state must satisfy Vulkan requirements.
    pub unsafe fn create_mesh_pso(&mut self, desc: &MeshPSODesc<'_>) -> Result<PSO> {
        create_raster_pso(
            self,
            desc.mesh_spirv,
            desc.fragment_spirv,
            desc.color_targets,
            desc.depth_format,
            desc.stencil_format,
            &desc.rasterization,
            true,
        )
    }

    /// # Safety
    /// The SPIR-V module and its compute entry point must satisfy this device's Vulkan requirements.
    pub unsafe fn create_compute_pso(&mut self, compute_spirv: &[u32]) -> Result<PSO> {
        let mut module = vk::ShaderModuleCreateInfo::default().code(compute_spirv);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .push(&mut module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(c"computeMain");
        let mut flags = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
        let info = vk::ComputePipelineCreateInfo::default()
            .push(&mut flags)
            .stage(stage)
            .base_pipeline_index(-1);
        let mut pipeline = vk::Pipeline::null();
        let result = (self.vk().fp_v1_0().create_compute_pipelines)(
            self.vk().handle(),
            vk::PipelineCache::null(),
            1,
            &info,
            ptr::null(),
            &mut pipeline,
        );
        if result != vk::Result::SUCCESS {
            if pipeline != vk::Pipeline::null() {
                self.vk().destroy_pipeline(pipeline, None);
            }
            return Err(result.into());
        }
        self.allocations.push(Allocation::Pipeline(pipeline));
        Ok(PSO {
            pso: pipeline,
            bind_point: vk::PipelineBindPoint::COMPUTE,
        })
    }

    /// # Safety
    /// The pipeline must belong to this device and have no pending recorded or executing uses.
    pub unsafe fn destroy_pso(&mut self, pso: PSO) {
        self.release(Allocation::Pipeline(pso.pso));
    }

    /// # Safety
    /// Both values must belong to this device. The command buffer must be recording, and the
    /// pipeline must remain allocated until all uses finish.
    pub unsafe fn bind_pso(&self, commands: &mut CommandBuffer, pso: &PSO) {
        self.vk()
            .cmd_bind_pipeline(commands.command_buffer, pso.bind_point, pso.pso);
    }

    /// The first begun command buffer initializes pending textures and the acquired swapchain
    /// image. It must be first in the next submission, which consumes every begun buffer.
    /// # Safety
    /// Submit the returned recording in the next batch. The first begun buffer must be first
    /// in that batch; pending texture allocations must remain alive through completion.
    pub unsafe fn begin_commands(&mut self) -> CommandBuffer {
        let mut result = self.acquire_command_context();
        require(
            self.vk().begin_command_buffer(
                result.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ),
        );
        result.timestamp_count = 0;
        if result.timestamp_pool != vk::QueryPool::null() {
            self.vk().cmd_reset_query_pool(
                result.command_buffer,
                result.timestamp_pool,
                0,
                self.timestamp_query_count,
            );
        }
        let mut barriers = [vk::ImageMemoryBarrier2::default(); IMAGE_BARRIER_BATCH_SIZE];
        let mut count = 0;
        while let Some(initialization) = self.pending_texture_initializations.pop_front() {
            barriers[count] = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(initialization.image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(initialization.aspect_mask)
                        .level_count(initialization.mip_levels)
                        .layer_count(initialization.array_layers),
                );
            count += 1;
            if count == IMAGE_BARRIER_BATCH_SIZE {
                record_image_barriers(self, result.command_buffer, &barriers);
                count = 0;
            }
        }
        if count != 0 {
            record_image_barriers(self, result.command_buffer, &barriers[..count]);
        }
        if let Some(swapchain) = self
            .swapchain
            .as_ref()
            .filter(|swapchain| swapchain.acquired && swapchain.transition_commands.is_none())
        {
            debug_assert!(swapchain.acquired && swapchain.present_context.is_some());
            let index = swapchain.image_index as usize;
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(if swapchain.initialized[index] {
                    vk::ImageLayout::PRESENT_SRC_KHR
                } else {
                    vk::ImageLayout::UNDEFINED
                })
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(swapchain.images[index])
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            record_image_barriers(self, result.command_buffer, &[barrier]);
            result.swapchain = true;
            self.swapchain.as_mut().unwrap().transition_commands = Some(result.command_buffer);
        }
        self.active_command_buffers += 1;
        result
    }

    /// # Safety
    /// The batch must contain every begun recording from this device, in initialization order,
    /// with no acquired frame. Resources must remain alive and synchronized until completion;
    /// the completion semaphore must belong to this device and its signal value must be valid.
    pub unsafe fn submit(&mut self, commands: Vec<CommandBuffer>, completion: &TimelinePoint) {
        assert!(!commands.is_empty());
        debug_assert!(
            !self
                .swapchain
                .as_ref()
                .is_some_and(|swapchain| swapchain.acquired)
        );
        debug_assert!(commands.iter().all(|command| !command.swapchain));
        submit_commands(
            commands,
            self,
            completion,
            vk::Semaphore::null(),
            vk::Semaphore::null(),
        );
    }

    /// # Safety
    /// The batch must contain every begun recording from this device, with the acquired image
    /// transition first. Call on the window's event-loop thread. Resources must remain alive and
    /// synchronized until completion, and the completion semaphore and value must be valid.
    pub unsafe fn submit_and_present(
        &mut self,
        commands: Vec<CommandBuffer>,
        completion: &TimelinePoint,
    ) {
        debug_assert!(self.swapchain.is_some() && !commands.is_empty());
        self.with_swapchain(|device, swapchain| {
            debug_assert!(
                swapchain.acquired
                    && commands[0].swapchain
                    && swapchain.transition_commands == Some(commands[0].command_buffer)
            );
            debug_assert!(commands[1..].iter().all(|command| !command.swapchain));
            let context_index = swapchain.present_context.unwrap();
            let context = device.present_contexts[context_index];
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(swapchain.images[swapchain.image_index as usize])
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            record_image_barriers(device, commands.last().unwrap().command_buffer, &[barrier]);
            debug_assert!(
                !context.present_pending && context.swapchain == vk::SwapchainKHR::null()
            );
            require(device.vk().reset_fences(&[context.presented]));
            swapchain.transition_commands = None;
            submit_commands(
                commands,
                device,
                completion,
                context.acquired,
                context.rendered,
            );
            swapchain.initialized[swapchain.image_index as usize] = true;
            let fences = [context.presented];
            let mut fence_info = vk::SwapchainPresentFenceInfoKHR::default().fences(&fences);
            let rendered = [context.rendered];
            let handles = [swapchain.handle];
            let indices = [swapchain.image_index];
            let present = vk::PresentInfoKHR::default()
                .push(&mut fence_info)
                .wait_semaphores(&rendered)
                .swapchains(&handles)
                .image_indices(&indices);
            if let Some(window) = &device.window {
                window.pre_present_notify();
            }
            let result = device.swapchains().queue_present(device.queue, &present);
            device.present_contexts[context_index].present_pending = true;
            device.present_contexts[context_index].swapchain = swapchain.handle;
            swapchain.present_context = None;
            swapchain.acquired = false;
            device.next_present_context =
                (device.next_present_context + 1) % device.present_context_count;
            match result {
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => swapchain.recreate_required = true,
                Ok(true) => {
                    swapchain.recreate_required |=
                        swapchain_surface_configuration_changed(device, swapchain)
                }
                result => {
                    require(result);
                }
            }
        })
    }

    /// # Safety
    /// The command buffer must be recording on this device with an available query slot.
    /// The destination must be an aligned, writable GPU range kept alive through completion.
    pub unsafe fn write_timestamp(
        &self,
        commands: &mut CommandBuffer,
        gpu_destination: u64,
        stage: Stage,
    ) {
        debug_assert!(commands.timestamp_count < self.timestamp_query_count);
        commands.timestamp_destinations[commands.timestamp_count as usize] = gpu_destination;
        self.vk().cmd_write_timestamp2(
            commands.command_buffer,
            stage.vk(),
            commands.timestamp_pool,
            commands.timestamp_count,
        );
        commands.timestamp_count += 1;
    }

    /// # Safety
    /// The command buffer must be recording on this device. The range must identify a texture
    /// descriptor heap with its required alignment and reserved storage, alive through completion.
    pub unsafe fn set_texture_descriptor_heap(&self, commands: &mut CommandBuffer, heap: GpuRange) {
        let properties = &self.heap_properties;
        let info = make_heap_bind_info(
            heap,
            properties
                .image_descriptor_alignment
                .max(properties.buffer_descriptor_alignment),
            properties.min_resource_heap_reserved_range,
        );
        (self.functions().cmd_bind_texture_heap)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be recording on this device. The range must identify a sampler
    /// descriptor heap with its required alignment and reserved storage, alive through completion.
    pub unsafe fn set_sampler_descriptor_heap(&self, commands: &mut CommandBuffer, heap: GpuRange) {
        let info = make_heap_bind_info(
            heap,
            self.heap_properties.sampler_descriptor_alignment,
            self.heap_properties.min_sampler_heap_reserved_range,
        );
        (self.functions().cmd_bind_sampler_heap)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be recording on this device, and the viewport must satisfy device limits.
    pub unsafe fn set_viewport(&self, commands: &mut CommandBuffer, viewport: &Viewport) {
        let viewport = vk::Viewport {
            x: viewport.x,
            y: viewport.y,
            width: viewport.width,
            height: viewport.height,
            min_depth: viewport.min_depth,
            max_depth: viewport.max_depth,
        };
        self.vk()
            .cmd_set_viewport_with_count(commands.command_buffer, &[viewport]);
    }

    /// # Safety
    /// The command buffer must be recording on this device, and the scissor must satisfy device limits.
    pub unsafe fn set_scissor(&self, commands: &mut CommandBuffer, scissor: &Scissor) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: scissor.x,
                y: scissor.y,
            },
            extent: vk::Extent2D {
                width: scissor.width,
                height: scissor.height,
            },
        };
        self.vk()
            .cmd_set_scissor_with_count(commands.command_buffer, &[scissor]);
    }

    /// # Safety
    /// The command buffer must be recording on this device, and the state must be valid for its pipeline.
    pub unsafe fn set_depth_stencil(
        &self,
        commands: &mut CommandBuffer,
        state: &DepthStencilState,
    ) {
        let vk_device = self.vk();
        let command = commands.command_buffer;
        vk_device.cmd_set_depth_test_enable(command, state.depth_test);
        if state.depth_test {
            vk_device.cmd_set_depth_write_enable(command, state.depth_write);
            vk_device.cmd_set_depth_compare_op(
                command,
                vk::CompareOp::from_raw(state.depth_compare as i32),
            );
        }
        vk_device.cmd_set_stencil_test_enable(command, state.stencil_test);
        if !state.stencil_test {
            return;
        }
        vk_device.cmd_set_stencil_op(
            command,
            vk::StencilFaceFlags::FRONT,
            vk::StencilOp::from_raw(state.front.fail as i32),
            vk::StencilOp::from_raw(state.front.pass as i32),
            vk::StencilOp::from_raw(state.front.depth_fail as i32),
            vk::CompareOp::from_raw(state.front.compare as i32),
        );
        vk_device.cmd_set_stencil_op(
            command,
            vk::StencilFaceFlags::BACK,
            vk::StencilOp::from_raw(state.back.fail as i32),
            vk::StencilOp::from_raw(state.back.pass as i32),
            vk::StencilOp::from_raw(state.back.depth_fail as i32),
            vk::CompareOp::from_raw(state.back.compare as i32),
        );
        vk_device.cmd_set_stencil_compare_mask(
            command,
            vk::StencilFaceFlags::FRONT_AND_BACK,
            state.stencil_read_mask as u32,
        );
        vk_device.cmd_set_stencil_write_mask(
            command,
            vk::StencilFaceFlags::FRONT_AND_BACK,
            state.stencil_write_mask as u32,
        );
        vk_device.cmd_set_stencil_reference(
            command,
            vk::StencilFaceFlags::FRONT,
            state.front.reference as u32,
        );
        vk_device.cmd_set_stencil_reference(
            command,
            vk::StencilFaceFlags::BACK,
            state.back.reference as u32,
        );
    }

    /// # Safety
    /// The command buffer must be recording outside a render pass on this device. Attachments
    /// must be live, compatible views with synchronized images in GENERAL layout.
    pub unsafe fn begin_render_pass(&self, commands: &mut CommandBuffer, desc: &RenderingDesc<'_>) {
        debug_assert!(desc.colors.len() <= MAX_COLOR_ATTACHMENTS);
        let area_view = if !desc.colors.is_empty() {
            desc.colors[0].render_view.as_ref()
        } else if desc.depth.render_view.is_some() {
            desc.depth.render_view.as_ref()
        } else {
            desc.stencil.render_view.as_ref()
        };
        let area_view = area_view.expect("render pass requires an attachment");
        let mut colors = [vk::RenderingAttachmentInfo::default(); MAX_COLOR_ATTACHMENTS];
        for (index, attachment) in desc.colors.iter().enumerate() {
            debug_assert!(!attachment.render_view.is_none());
            colors[index] = vk::RenderingAttachmentInfo::default()
                .image_view(attachment.render_view.as_ref().unwrap().view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::from_raw(attachment.load as i32))
                .store_op(vk::AttachmentStoreOp::from_raw(attachment.store as i32))
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [
                            attachment.clear.x,
                            attachment.clear.y,
                            attachment.clear.z,
                            attachment.clear.w,
                        ],
                    },
                });
        }
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(
                desc.depth
                    .render_view
                    .as_ref()
                    .map_or(vk::ImageView::null(), |view| view.view),
            )
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::from_raw(desc.depth.load as i32))
            .store_op(vk::AttachmentStoreOp::from_raw(desc.depth.store as i32))
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: desc.depth.clear,
                    stencil: 0,
                },
            });
        let stencil = vk::RenderingAttachmentInfo::default()
            .image_view(
                desc.stencil
                    .render_view
                    .as_ref()
                    .map_or(vk::ImageView::null(), |view| view.view),
            )
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::from_raw(desc.stencil.load as i32))
            .store_op(vk::AttachmentStoreOp::from_raw(desc.stencil.store as i32))
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: desc.stencil.clear as u32,
                },
            });
        let extent = vk::Extent2D {
            width: area_view.width,
            height: area_view.height,
        };
        let mut rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&colors[..desc.colors.len()]);
        if desc.depth.render_view.is_some() {
            rendering = rendering.depth_attachment(&depth);
        }
        if desc.stencil.render_view.is_some() {
            rendering = rendering.stencil_attachment(&stencil);
        }
        self.vk()
            .cmd_begin_rendering(commands.command_buffer, &rendering);
        self.set_viewport(
            commands,
            &Viewport {
                width: extent.width as f32,
                height: extent.height as f32,
                ..Default::default()
            },
        );
        self.set_scissor(
            commands,
            &Scissor {
                width: extent.width,
                height: extent.height,
                ..Default::default()
            },
        );
        self.set_depth_stencil(commands, &DepthStencilState::default());
    }

    /// # Safety
    /// The command buffer must be recording inside a render pass on this device.
    pub unsafe fn end_render_pass(&self, commands: &mut CommandBuffer) {
        self.vk().cmd_end_rendering(commands.command_buffer);
    }

    /// # Safety
    /// The command buffer must be inside a compatible render pass on this device. Pipeline,
    /// root data, descriptor bindings, and resource synchronization must satisfy the shaders and Vulkan.
    pub unsafe fn draw(
        &self,
        commands: &mut CommandBuffer,
        root: &[u8],
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        emit_root_data(self, commands, root);
        self.vk().cmd_draw(
            commands.command_buffer,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }

    /// # Safety
    /// The command buffer must be inside a compatible render pass on this device. Index ranges,
    /// root data, bindings, and synchronization must be valid and remain alive through completion.
    pub unsafe fn draw_indexed(
        &self,
        commands: &mut CommandBuffer,
        root: &[u8],
        indices: GpuRange,
        index_type: IndexType,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        emit_root_data(self, commands, root);
        let bind = vk::BindIndexBuffer3InfoKHR::default()
            .address_range(vk::DeviceAddressRangeKHR {
                address: indices.gpu,
                size: indices.size,
            })
            .address_flags(ADDRESS_FLAGS)
            .index_type(vk::IndexType::from_raw(index_type as i32));
        (self.functions().cmd_bind_index_buffer)(commands.command_buffer, &bind);
        self.vk().cmd_draw_indexed(
            commands.command_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    /// # Safety
    /// The command buffer must be inside a compatible render pass on this device. Indirect
    /// arguments, stride, count, root data, bindings, and synchronization must satisfy Vulkan requirements.
    pub unsafe fn draw_indirect(
        &self,
        commands: &mut CommandBuffer,
        root: &[u8],
        arguments: GpuRange,
        draw_count: u32,
        stride: u32,
    ) {
        emit_root_data(self, commands, root);
        let info = vk::DrawIndirect2InfoKHR::default()
            .address_range(vk::StridedDeviceAddressRangeKHR {
                address: arguments.gpu,
                size: arguments.size,
                stride: if stride == 0 {
                    size_of::<vk::DrawIndirectCommand>() as u64
                } else {
                    stride as u64
                },
            })
            .address_flags(ADDRESS_FLAGS)
            .draw_count(draw_count);
        (self.functions().cmd_draw_indirect)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be inside a compatible render pass on this device. Index and
    /// indirect ranges, root data, bindings, and synchronization must satisfy Vulkan requirements.
    pub unsafe fn draw_indexed_indirect(
        &self,
        commands: &mut CommandBuffer,
        root: &[u8],
        indices: GpuRange,
        index_type: IndexType,
        arguments: GpuRange,
        draw_count: u32,
        stride: u32,
    ) {
        emit_root_data(self, commands, root);
        let bind = vk::BindIndexBuffer3InfoKHR::default()
            .address_range(vk::DeviceAddressRangeKHR {
                address: indices.gpu,
                size: indices.size,
            })
            .address_flags(ADDRESS_FLAGS)
            .index_type(vk::IndexType::from_raw(index_type as i32));
        (self.functions().cmd_bind_index_buffer)(commands.command_buffer, &bind);
        let info = vk::DrawIndirect2InfoKHR::default()
            .address_range(vk::StridedDeviceAddressRangeKHR {
                address: arguments.gpu,
                size: arguments.size,
                stride: if stride == 0 {
                    size_of::<vk::DrawIndexedIndirectCommand>() as u64
                } else {
                    stride as u64
                },
            })
            .address_flags(ADDRESS_FLAGS)
            .draw_count(draw_count);
        (self.functions().cmd_draw_indexed_indirect)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be recording outside a render pass on this device with a compute
    /// pipeline. Group counts, root data, bindings, and synchronization must satisfy Vulkan requirements.
    pub unsafe fn dispatch(
        &self,
        commands: &mut CommandBuffer,
        root: &[u8],
        group_count: Uint32x3,
    ) {
        emit_root_data(self, commands, root);
        self.vk().cmd_dispatch(
            commands.command_buffer,
            group_count.x,
            group_count.y,
            group_count.z,
        );
    }

    /// # Safety
    /// The command buffer must be recording outside a render pass on this device with a compute
    /// pipeline. The indirect range, root data, bindings, and synchronization must be valid.
    pub unsafe fn dispatch_indirect(
        &self,
        commands: &mut CommandBuffer,
        root: &[u8],
        arguments: GpuRange,
    ) {
        emit_root_data(self, commands, root);
        let info = vk::DispatchIndirect2InfoKHR::default()
            .address_range(vk::DeviceAddressRangeKHR {
                address: arguments.gpu,
                size: arguments.size,
            })
            .address_flags(ADDRESS_FLAGS);
        (self.functions().cmd_dispatch_indirect)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be inside a compatible render pass on this device with a mesh
    /// pipeline. Group counts, root data, bindings, and synchronization must be valid.
    pub unsafe fn draw_meshlets(
        &self,
        commands: &mut CommandBuffer,
        root: &[u8],
        group_count: Uint32x3,
    ) {
        emit_root_data(self, commands, root);
        (self.functions().cmd_draw_mesh_tasks)(
            commands.command_buffer,
            group_count.x,
            group_count.y,
            group_count.z,
        );
    }

    /// # Safety
    /// The command buffer must be inside a compatible render pass on this device with a mesh
    /// pipeline. Indirect arguments, stride, count, root data, and resource accesses must be valid.
    pub unsafe fn draw_meshlets_indirect(
        &self,
        commands: &mut CommandBuffer,
        root: &[u8],
        arguments: GpuRange,
        draw_count: u32,
        stride: u32,
    ) {
        emit_root_data(self, commands, root);
        let info = vk::DrawIndirect2InfoKHR::default()
            .address_range(vk::StridedDeviceAddressRangeKHR {
                address: arguments.gpu,
                size: arguments.size,
                stride: if stride == 0 {
                    size_of::<vk::DrawMeshTasksIndirectCommandEXT>() as u64
                } else {
                    stride as u64
                },
            })
            .address_flags(ADDRESS_FLAGS)
            .draw_count(draw_count);
        (self.functions().cmd_draw_mesh_tasks_indirect)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be recording outside a render pass on this device. Source and
    /// destination ranges must be valid, appropriately sized, synchronized, and nonoverlapping.
    pub unsafe fn copy_memory(
        &self,
        commands: &mut CommandBuffer,
        source: GpuRange,
        destination: GpuRange,
    ) {
        let regions = [vk::DeviceMemoryCopyKHR::default()
            .src_range(vk::DeviceAddressRangeKHR {
                address: source.gpu,
                size: source.size,
            })
            .src_flags(ADDRESS_FLAGS)
            .dst_range(vk::DeviceAddressRangeKHR {
                address: destination.gpu,
                size: destination.size,
            })
            .dst_flags(ADDRESS_FLAGS)];
        let info = vk::CopyDeviceMemoryInfoKHR::default().regions(&regions);
        (self.functions().cmd_copy_memory)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be recording outside a render pass on this device. The image
    /// and source range must be live and synchronized; regions, pitches, and layout must be valid.
    pub unsafe fn copy_memory_to_texture(
        &self,
        commands: &mut CommandBuffer,
        source: GpuRange,
        destination: &Texture,
        copy: &TextureCopyDesc,
    ) {
        let regions = [make_texture_copy_region(destination, copy, source)];
        let info = vk::CopyDeviceMemoryImageInfoKHR::default()
            .image(destination.image)
            .regions(&regions);
        (self.functions().cmd_copy_memory_to_image)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be recording outside a render pass on this device. The image
    /// and destination range must be live and synchronized; regions, pitches, and layout must be valid.
    pub unsafe fn copy_texture_to_memory(
        &self,
        commands: &mut CommandBuffer,
        source: &Texture,
        destination: GpuRange,
        copy: &TextureCopyDesc,
    ) {
        let regions = [make_texture_copy_region(source, copy, destination)];
        let info = vk::CopyDeviceMemoryImageInfoKHR::default()
            .image(source.image)
            .regions(&regions);
        (self.functions().cmd_copy_image_to_memory)(commands.command_buffer, &info);
    }

    /// # Safety
    /// The command buffer must be recording on this device. Stage and access masks must be
    /// supported and valid for its current rendering scope and the resource dependencies.
    pub unsafe fn barrier(
        &self,
        commands: &mut CommandBuffer,
        before: Stage,
        before_access: Access,
        after: Stage,
        after_access: Access,
    ) {
        let barriers = [vk::MemoryBarrier2::default()
            .src_stage_mask(before.vk())
            .src_access_mask(before_access.vk())
            .dst_stage_mask(after.vk())
            .dst_access_mask(after_access.vk())];
        self.vk().cmd_pipeline_barrier2(
            commands.command_buffer,
            &vk::DependencyInfo::default().memory_barriers(&barriers),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::MaybeUninit;
    use std::sync::Mutex;

    #[test]
    fn owned_and_shared_resources_support_thread_transfer() {
        fn send<T: Send>() {}
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Arc<Mutex<Device>>>();
        send_sync::<Arc<Mutex<CommandBuffer>>>();
        send_sync::<Arc<Mutex<GpuHeap>>>();
        send_sync::<Arc<RenderView>>();
        send_sync::<Arc<TimelineSemaphore>>();
        send::<Texture>();
        send::<TextureHeap>();
        send::<PSO>();
    }

    #[test]
    fn mapping_exposes_only_usable_bytes_after_alignment_padding() {
        let mut allocation = [MaybeUninit::new(0xa5_u8); 32];
        let mut heap = GpuHeap {
            range: GpuRange { gpu: 16, size: 8 },
            backing: BackingBuffer {
                mapped: ptr::NonNull::new(allocation.as_mut_ptr().cast()),
                ..Default::default()
            },
            mapped_offset: 16,
        };
        let bytes = unsafe { heap.mapped_bytes().unwrap() };
        assert_eq!(bytes.len(), 8);
        for byte in bytes {
            byte.write(0x42);
        }
        let actual = allocation.map(|byte| unsafe { byte.assume_init() });
        assert_eq!(&actual[..16], &[0xa5; 16]);
        assert_eq!(&actual[16..24], &[0x42; 8]);
        assert_eq!(&actual[24..], &[0xa5; 8]);
    }

    #[test]
    fn gpu_only_heap_has_no_cpu_mapping() {
        let mut heap = GpuHeap {
            range: GpuRange { gpu: 16, size: 8 },
            backing: BackingBuffer::default(),
            mapped_offset: 0,
        };
        assert!(unsafe { heap.mapped_bytes() }.is_none());
    }
}
