mod example_support;

use example_support::{Example, read_spirv, run_example};
use no_graphics::{
    ColorAttachment, ColorTargetDesc, Device, DeviceDesc, Format, GraphicsPSODesc, LoadOp, PSO,
    RenderingDesc, TimelinePoint, acquire, begin_commands, begin_render_pass, bind_pso,
    create_device, create_graphics_pso, create_timeline_semaphore, destroy_device, destroy_pso,
    destroy_timeline_semaphore, draw, end_render_pass, get_device_caps, submit_and_present,
    wait_idle,
};
use std::{error::Error, path::PathBuf, rc::Rc};
use winit::window::Window;

struct Triangle {
    device: *mut Device,
    pso: *mut PSO,
    completion: TimelinePoint,
}

impl Example for Triangle {
    fn new(window: Rc<Window>) -> Result<Self, Box<dyn Error>> {
        let shader_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/shaders/compiled/triangle.spv");
        let spirv = read_spirv(&shader_path)?;
        let device = create_device(&DeviceDesc {
            window: Some(window),
            swapchain_format: Format::Bgra8Srgb,
            ..Default::default()
        })?;

        println!("Using {}", get_device_caps(unsafe { &*device }).device_name);

        let pso = match unsafe {
            create_graphics_pso(
                device,
                &GraphicsPSODesc {
                    vertex_spirv: &spirv,
                    fragment_spirv: &spirv,
                    color_targets: &[ColorTargetDesc {
                        format: Format::Bgra8Srgb,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            )
        } {
            Ok(pso) => pso,
            Err(error) => {
                unsafe { destroy_device(device) };
                return Err(Box::new(error));
            }
        };
        let semaphore = match unsafe { create_timeline_semaphore(device, 0) } {
            Ok(semaphore) => semaphore,
            Err(error) => {
                unsafe {
                    destroy_pso(pso);
                    destroy_device(device);
                }
                return Err(Box::new(error));
            }
        };

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
            let Some(frame) = acquire(self.device) else {
                return;
            };
            let commands = begin_commands(self.device);
            begin_render_pass(
                commands,
                &RenderingDesc {
                    colors: &[ColorAttachment {
                        render_view: frame.render_view,
                        load: LoadOp::Clear,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            );
            bind_pso(commands, self.pso);
            draw(commands, &[], 3, 1, 0, 0);
            end_render_pass(commands);
            self.completion.value += 1;
            submit_and_present(self.device, &[commands], self.completion);
        }
    }

    fn shutdown(self) {
        unsafe {
            wait_idle(self.device);
            destroy_timeline_semaphore(self.completion.semaphore);
            destroy_pso(self.pso);
            destroy_device(self.device);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    run_example::<Triangle>("NoGraphicsAPI triangle", 512, 512)
}
