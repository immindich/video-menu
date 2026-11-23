//! Virtual camera daemon.
//!
//! Exposes a PipeWire `Video/Source` node that mirrors a real webcam, flipped
//! vertically. The real camera is only opened while some client is actually
//! streaming from the virtual node; otherwise it stays idle (and suspended).
//!
//! Usage: `video-menu [CAMERA_NODE_NAME]`. Without an argument, the first
//! camera found in the graph is used.

mod model;
mod relay;

use std::error::Error;

use pipewire as pw;
use pw::{context::ContextRc, main_loop::MainLoopRc};

use relay::{CameraRelay, OUTPUT_NODE_NAME, VideoSources, probe_formats, roundtrip};

fn main() -> Result<(), Box<dyn Error>> {
    pw::init();

    let target = std::env::args().nth(1);

    let mainloop = MainLoopRc::new(None)?;
    let context = ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    let sources = VideoSources::watch(&core)?;
    roundtrip(&mainloop, &core)?;
    let (id, camera) = sources.select(target.as_deref())?;
    let formats = probe_formats(&mainloop, &core, &sources, id)?;
    eprintln!("using camera {camera} ({} formats)", formats.len());

    let relay = CameraRelay::new(&core, sources, &camera)?;
    let _listeners = relay.listen()?;
    relay.publish(&formats)?;

    eprintln!("virtual camera {OUTPUT_NODE_NAME} ready");
    mainloop.run();
    Ok(())
}
