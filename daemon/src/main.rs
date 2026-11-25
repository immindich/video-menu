//! Virtual camera daemon.
//!
//! Exposes a PipeWire `Video/Source` node that mirrors a real webcam. Frames
//! are converted to planar RGB, optionally have their background blurred or
//! replaced with a colour using a matting model (on the NPU), and are
//! offered to applications as YUY2. The real camera is only opened while some client
//! is actually streaming from the virtual node; otherwise it stays idle (and
//! suspended).
//!
//! Usage: `video-menu [CAMERA_NODE_NAME]`. Without an argument, the first
//! camera found in the graph is used; if there is none, the daemon starts
//! with no camera selected.
//!
//! The daemon is controlled over D-Bus (see [`dbus`] and the
//! `video-menu-protocol` crate): it owns `io.github.immindich.VideoMenu` on
//! the session bus, so only one can run at a time. The environment variables
//! below set its starting settings.
//!
//! Frames are processed on the GPU (Vulkan via wgpu) when one is available,
//! otherwise on the CPU; set `VIDEO_MENU_BACKEND=cpu` or `=gpu` to force one.
//!
//! Other settings:
//! - `VIDEO_MENU_EFFECT` is `none` (default: frames pass through), `blur`
//!   (blur the background) or `color` (replace it with a solid colour).
//! - `VIDEO_MENU_BACKGROUND_COLOR` (`#rrggbb`, default `#00b140`) is the
//!   `color` effect's colour.
//! - `VIDEO_MENU_BLUR_RADIUS` (default 2) is the `blur` effect's strength:
//!   the blur's standard deviation as a percentage of the frame height.
//! - `VIDEO_MENU_MATTE_MODEL` chooses the matting model: `rvm` (default,
//!   GPL-3.0) or `selfie` (MediaPipe's selfie segmenter, Apache-2.0; see
//!   `model.rs`).
//! - `VIDEO_MENU_SELFIE_SMOOTHING` (0..1, default 0.5) is how strongly the
//!   selfie model's mask is smoothed over time where the image is still; 0
//!   disables it.
//! - `VIDEO_MENU_MODEL` / `VIDEO_MENU_SELFIE_MODEL` are the model files; if
//!   unset or missing, `rvm_mobilenetv3_fp16_core.onnx` /
//!   `selfie_segmenter_landscape.xml` in the current directory are used.
//! - `VIDEO_MENU_MODEL_DEVICE` is the device the model runs on (default NPU).
//! - `VIDEO_MENU_NPU_TURBO=1` trades CPU time for slightly lower model
//!   latency when the camera is slow (see `model.rs`).
//! - `VIDEO_MENU_GPU_PROFILE=FILE` records a timeline of each frame's GPU
//!   work to `FILE` (see `profile.rs`).
//! - `VIDEO_MENU_REUSE_MATTE=1` composites each frame with the previous
//!   frame's matte while the model works on the current one, taking the
//!   model off the path from camera to output. The matte then lags the
//!   image by a frame.

mod dbus;
mod formats;
mod gpu;
mod model;
mod pipeline;
mod processor;
mod profile;
mod relay;
// The relay starts using these in the next commit.
#[allow(dead_code)]
mod scheduler;
#[allow(dead_code)]
mod worker;

use std::{error::Error, rc::Rc};

use pipewire as pw;
use pw::{context::ContextRc, main_loop::MainLoopRc};

use processor::{Effects, Processor};
use relay::{CameraRelay, OUTPUT_NODE_NAME, Settings, VideoSources, roundtrip};

fn main() -> Result<(), Box<dyn Error>> {
    pw::init();

    // Claim the bus name first, so a second daemon stops before it loads a
    // model or opens a camera. Requests wait in the channel until the relay
    // exists.
    let settings = Settings::from_env()?;
    let (calls, calls_rx) = pw::channel::channel::<dbus::Call>();
    let service = dbus::start(
        calls,
        dbus::Snapshot {
            model: settings.model,
            effect: settings.effect,
            reuse_matte: settings.reuse_matte,
            selfie_smoothing: settings.selfie_smoothing as f64,
            background_color: settings.background_color.into(),
            blur_radius: settings.blur_radius as f64,
            ..Default::default()
        },
    )?;

    let target = std::env::args().nth(1);

    let mainloop = MainLoopRc::new(None)?;
    let context = ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    let sources = VideoSources::watch(&core)?;
    roundtrip(&mainloop, &core)?; // the initial sources
    roundtrip(&mainloop, &core)?; // their formats
    let initial = sources.select(target.as_deref())?;

    let (effects, description) = Effects::load(
        settings.background(),
        settings.model,
        settings.reuse_matte,
        settings.selfie_smoothing,
    )?;
    eprintln!("effect: {description}");
    let processor = Processor::from_env(effects)?;
    eprintln!("processing frames on the {}", processor.describe());
    let relay = CameraRelay::new(&core, sources, processor, settings)?;
    let _listeners = relay.listen()?;
    match &initial {
        Some(camera) => relay.set_camera(Some(camera))?,
        None => eprintln!("no camera found; starting unconnected"),
    }
    relay.publish()?;

    let service = Rc::new(service);
    service.refresh(snapshot(&relay));
    let _calls = calls_rx.attach(mainloop.loop_(), {
        let (relay, service) = (relay.clone(), service.clone());
        move |call: dbus::Call| {
            let result = handle(&relay, &call.request);
            if let Err(e) = &result {
                eprintln!("dbus: {:?} failed: {e}", call.request);
            }
            service.refresh(snapshot(&relay));
            call.reply(result);
        }
    });
    // Sources come and go and statistics change on their own; publish them.
    let refresh = mainloop.loop_().add_timer({
        let relay = relay.clone();
        move |_| service.refresh(snapshot(&relay))
    });
    let second = Some(std::time::Duration::from_secs(1));
    refresh.update_timer(second, second).into_sync_result()?;

    eprintln!("virtual camera {OUTPUT_NODE_NAME} ready");
    mainloop.run();
    Ok(())
}

/// What D-Bus clients should see of the relay right now.
fn snapshot(relay: &CameraRelay) -> dbus::Snapshot {
    let settings = relay.settings();
    dbus::Snapshot {
        camera: relay.camera().unwrap_or_default(),
        sources: relay.sources().list(),
        model: settings.model,
        effect: settings.effect,
        reuse_matte: settings.reuse_matte,
        selfie_smoothing: settings.selfie_smoothing as f64,
        background_color: settings.background_color.into(),
        blur_radius: settings.blur_radius as f64,
        stats: relay.stats(),
    }
}

/// Carry out a D-Bus client's request on the relay.
fn handle(relay: &CameraRelay, request: &dbus::Request) -> Result<(), String> {
    use dbus::Request::*;
    match request {
        Camera(name) => relay
            .set_camera(Some(name.as_str()).filter(|n| !n.is_empty()))
            .map_err(|e| e.to_string()),
        Model(model) => relay.set_model(*model),
        Effect(effect) => relay.set_effect(*effect),
        BackgroundColor(color) => relay.set_background_color(*color),
        BlurRadius(radius) => relay.set_blur_radius(*radius),
        ReuseMatte(reuse) => {
            relay.set_reuse_matte(*reuse);
            Ok(())
        }
        SelfieSmoothing(strength) => relay.set_selfie_smoothing(*strength),
    }
}
