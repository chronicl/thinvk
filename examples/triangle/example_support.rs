use std::{error::Error, fs, path::Path, rc::Rc};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

pub trait Example: Sized {
    fn new(window: Rc<Window>) -> Result<Self, Box<dyn Error>>;
    fn render(&mut self);
    fn shutdown(self);
}

struct ExampleRunner<E> {
    title: String,
    width: u32,
    height: u32,
    example: Option<E>,
    window: Option<Rc<Window>>,
    error: Option<Box<dyn Error>>,
}

impl<E: Example> ExampleRunner<E> {
    fn shutdown(&mut self) {
        if let Some(example) = self.example.take() {
            example.shutdown();
        }
    }
}

impl<E: Example> ApplicationHandler for ExampleRunner<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(self.width, self.height));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Rc::new(window),
            Err(error) => {
                self.error = Some(Box::new(error));
                event_loop.exit();
                return;
            }
        };
        match E::new(window.clone()) {
            Ok(example) => {
                self.example = Some(example);
                self.window = Some(window);
            }
            Err(error) => {
                self.error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self
            .window
            .as_ref()
            .is_none_or(|window| window.id() != window_id)
        {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
            {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(example) = &mut self.example {
                    example.render();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.shutdown();
    }
}

pub fn run_example<E: Example>(title: &str, width: u32, height: u32) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut runner: ExampleRunner<E> = ExampleRunner {
        title: title.to_owned(),
        width,
        height,
        example: None,
        window: None,
        error: None,
    };
    let run_result = event_loop.run_app(&mut runner);
    runner.shutdown();
    run_result?;
    match runner.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub fn read_spirv(path: &Path) -> Result<Vec<u32>, Box<dyn Error>> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if bytes.len() < 5 * size_of::<u32>() || bytes.len() % size_of::<u32>() != 0 {
        return Err(format!("invalid SPIR-V file size: {}", path.display()).into());
    }
    let words: Vec<u32> = bytes
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    if words[0] != 0x0723_0203 {
        return Err(format!("invalid SPIR-V file: {}", path.display()).into());
    }
    Ok(words)
}
