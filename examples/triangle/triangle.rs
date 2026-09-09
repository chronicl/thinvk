mod example_support;

use example_support::{Example, read_spirv, run_example};
use no_graphics::{
    ColorAttachment, ColorTargetDesc, Device, DeviceDesc, Format, GraphicsPSODesc, LoadOp, PSO,
    RenderingDesc, TimelinePoint,
};
use std::{error::Error, path::PathBuf, sync::Arc};
use winit::window::Window;

struct Triangle {
    device: Device,
    pso: PSO,
    completion: TimelinePoint,
}

impl Example for Triangle {
    fn new(window: Arc<Window>) -> Result<Self, Box<dyn Error>> {
        let shader_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/shaders/compiled/triangle.spv");
        let spirv = read_spirv(&shader_path)?;
        let mut device = Device::new(&DeviceDesc {
            window: Some(window),
            swapchain_format: Format::Bgra8Srgb,
            ..Default::default()
        })?;

        println!("Using {}", device.get_device_caps().device_name);

        let pso = unsafe {
            device.create_graphics_pso(&GraphicsPSODesc {
                vertex_spirv: &spirv,
                fragment_spirv: &spirv,
                color_targets: &[ColorTargetDesc {
                    format: Format::Bgra8Srgb,
                    ..Default::default()
                }],
                ..Default::default()
            })?
        };
        let semaphore = unsafe { device.create_timeline_semaphore(0)? };

        Ok(Self {
            device,
            pso,
            completion: TimelinePoint {
                semaphore,
                value: 0,
            },
        })
    }

    fn render(&mut self) {
        unsafe {
            let Some(frame) = self.device.acquire() else {
                return;
            };
            let mut commands = self.device.begin_commands();
            self.device.begin_render_pass(
                &mut commands,
                &RenderingDesc {
                    colors: &[ColorAttachment {
                        render_view: Some(frame.render_view),
                        load: LoadOp::Clear,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            );
            self.device.bind_pso(&mut commands, &self.pso);
            self.device.draw(&mut commands, &[], 3, 1, 0, 0);
            self.device.end_render_pass(&mut commands);
            self.completion.value += 1;
            self.device
                .submit_and_present(vec![commands], &self.completion);
        }
    }

    fn shutdown(self) {
        // Device::drop waits for submissions and releases all remaining allocations.
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    run_example::<Triangle>("NoGraphicsAPI triangle", 512, 512)
}
