//! Virtual camera daemon.
//!
//! Exposes a PipeWire `Video/Source` node that mirrors a real webcam. Frames
//! are converted to planar RGB, their background is replaced with green
//! using a matting model (RVM, on the NPU), and they are offered to
//! applications as YUY2. The real camera is only opened while some client
//! is actually streaming from the virtual node; otherwise it stays idle (and
//! suspended).
//!
//! Usage: `video-menu [CAMERA_NODE_NAME]`. Without an argument, the first
//! camera found in the graph is used; if there is none, the daemon starts
//! with no camera selected. When run in a terminal, cameras can be switched
//! with commands on stdin (see [`commands`]).
//!
//! Frames are processed on the GPU (Vulkan via wgpu) when one is available,
//! otherwise on the CPU; set `VIDEO_MENU_BACKEND=cpu` or `=gpu` to force one.
//!
//! Other settings:
//! - `VIDEO_MENU_EFFECT=flip` flips frames instead of running the model.
//! - `VIDEO_MENU_MODEL` is the model file; if it is unset or doesn't exist,
//!   `rvm_mobilenetv3_fp16_core.onnx` in the current directory is used.
//! - `VIDEO_MENU_MODEL_DEVICE` is the device the model runs on (default NPU).
//! - `VIDEO_MENU_NPU_TURBO=1` trades CPU time for slightly lower model
//!   latency when the camera is slow (see `model.rs`).
//! - `VIDEO_MENU_REUSE_MATTE=1` composites each frame with the previous
//!   frame's matte while the model works on the current one, taking the
//!   model off the path from camera to output. The matte then lags the
//!   image by a frame.

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

use model::{Matting, RvmMatte};
use processor::{Effects, Processor};
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

    let effects = match std::env::var("VIDEO_MENU_EFFECT").as_deref() {
        Ok("flip") => Effects::Flip,
        Ok("green") | Ok("") | Err(_) => {
            let matte = RvmMatte::from_env()?;
            let reuse = std::env::var("VIDEO_MENU_REUSE_MATTE").as_deref() == Ok("1");
            eprintln!("matting model running on the {}", matte.device());
            if reuse {
                eprintln!("reusing each frame's matte for the next frame");
            }
            Effects::GreenScreen(Matting::new(Box::new(matte), reuse))
        }
        Ok(other) => return Err(format!("VIDEO_MENU_EFFECT must be green or flip, not {other:?}").into()),
    };
    let processor = Processor::from_env(effects)?;
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
