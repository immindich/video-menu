//! Virtual camera daemon.
//!
//! Exposes a PipeWire `Video/Source` node that mirrors a real webcam. Frames
//! are converted to planar RGB, run through effects (currently a vertical
//! flip as a placeholder), and offered to applications as YUY2. The real
//! camera is only opened while some client is actually streaming from the
//! virtual node; otherwise it stays idle (and suspended).
//!
//! Usage: `video-menu [CAMERA_NODE_NAME]`. Without an argument, the first
//! camera found in the graph is used; if there is none, the daemon starts
//! with no camera selected. When run in a terminal, cameras can be switched
//! with commands on stdin (see [`commands`]).
//!
//! Frames are processed on the GPU (Vulkan via wgpu) when one is available,
//! otherwise on the CPU; set `VIDEO_MENU_BACKEND=cpu` or `=gpu` to force one.

mod commands;
mod formats;
mod gpu;
mod model;
mod pipeline;
mod processor;
mod relay;

use std::error::Error;

use pipewire as pw;
use pw::{context::ContextRc, main_loop::MainLoopRc};

use processor::Processor;
use relay::{CameraRelay, OUTPUT_NODE_NAME, VideoSources, roundtrip};

fn main() -> Result<(), Box<dyn Error>> {
    pw::init();

    let target = std::env::args().nth(1);

    let mainloop = MainLoopRc::new(None)?;
    let context = ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    let sources = VideoSources::watch(&core)?;
    roundtrip(&mainloop, &core)?; // the initial sources
    roundtrip(&mainloop, &core)?; // their formats
    let initial = sources.select(target.as_deref())?;

    let processor = Processor::from_env()?;
    eprintln!("processing frames on the {}", processor.describe());
    let relay = CameraRelay::new(&core, sources, processor)?;
    let _listeners = relay.listen()?;
    match &initial {
        Some(camera) => relay.set_camera(Some(camera))?,
        None => eprintln!("no camera found; starting unconnected"),
    }
    relay.publish()?;
    let _commands = commands::attach(&mainloop, relay.clone());

    eprintln!("virtual camera {OUTPUT_NODE_NAME} ready");
    mainloop.run();
    Ok(())
}
