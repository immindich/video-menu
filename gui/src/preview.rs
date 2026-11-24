//! Live preview: reads the virtual camera like any other application would,
//! on a PipeWire thread of its own, and hands frames to the UI.
//!
//! Like any consumer, a running preview keeps the real camera on, so it only
//! runs while the user has it switched on ([`Preview::start`] /
//! [`Preview::stop`]).

use std::{
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use eframe::egui::{Color32, ColorImage};
use pipewire as pw;
use pw::{
    properties::properties,
    spa::{
        self,
        param::{
            ParamType,
            format::{FormatProperties, MediaSubtype, MediaType},
            video::{VideoFormat, VideoInfoRaw},
        },
        pod::{ChoiceValue, Object, Pod, Property, PropertyFlags, Value, serialize::PodSerializer},
        utils::{Choice, ChoiceEnum, ChoiceFlags, Direction, Id, SpaTypes},
    },
    stream::{StreamFlags, StreamRc, StreamState},
};

/// Node name of the daemon's virtual camera.
const TARGET: &str = "video-menu.camera";

/// Where the preview thread leaves frames for the UI.
#[derive(Default)]
pub struct Shared {
    /// The newest frame not yet shown.
    pub frame: Option<ColorImage>,
    /// What the stream is doing, for the UI to show while there's no frame.
    pub status: String,
}

/// A running preview; stops when dropped.
pub struct Preview {
    pub shared: Arc<Mutex<Shared>>,
    quit: pw::channel::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Preview {
    /// Start reading the virtual camera. `repaint` is called for each frame.
    pub fn start(repaint: impl Fn() + Send + Sync + 'static) -> Self {
        let repaint: Repaint = Arc::new(repaint);
        let shared = Arc::new(Mutex::new(Shared {
            frame: None,
            status: "Connecting…".into(),
        }));
        let (quit, quit_rx) = pw::channel::channel();
        let thread_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("preview".into())
            .spawn(move || {
                if let Err(e) = run(&thread_shared, quit_rx, repaint.clone()) {
                    thread_shared.lock().unwrap().status = format!("Preview failed: {e}");
                    repaint();
                }
            })
            .expect("starting the preview thread");
        Self { shared, quit, thread: Some(thread) }
    }

    /// Disconnect from the virtual camera and wait for the thread to end.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        let _ = self.quit.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        self.shutdown();
    }
}

type Repaint = Arc<dyn Fn() + Send + Sync>;

fn run(
    shared: &Arc<Mutex<Shared>>,
    quit: pw::channel::Receiver<()>,
    repaint: Repaint,
) -> Result<(), Box<dyn std::error::Error>> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let _quit = quit.attach(mainloop.loop_(), {
        let mainloop = mainloop.clone();
        move |()| mainloop.quit()
    });

    let stream = StreamRc::new(
        core,
        "video-menu-preview",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Camera",
            "target.object" => TARGET,
            // Without the daemon, don't fall back to some other camera: the
            // preview would show unprocessed video as if it were ours.
            "node.dont-fallback" => "true",
        },
    )?;

    struct Data {
        shared: Arc<Mutex<Shared>>,
        repaint: Repaint,
        size: Option<(usize, usize)>,
    }
    let data = Data { shared: shared.clone(), repaint, size: None };

    let _listener = stream
        .add_local_listener_with_user_data(data)
        .state_changed(|_, data, _, new| {
            let status = match new {
                StreamState::Streaming => "Waiting for frames…".to_string(),
                StreamState::Error(e) => format!("Stream error: {e}"),
                StreamState::Unconnected => "Not connected to the virtual camera".into(),
                _ => "Connecting…".into(),
            };
            data.shared.lock().unwrap().status = status;
            (data.repaint)();
        })
        .param_changed(|_, data, id, param| {
            let Some(param) = param else { return };
            if id != ParamType::Format.as_raw() {
                return;
            }
            let mut info = VideoInfoRaw::new();
            if info.parse(param).is_ok() && info.format() == VideoFormat::YUY2 {
                let size = info.size();
                data.size = Some((size.width as usize, size.height as usize));
            }
        })
        .process(|stream, data| {
            let Some(mut buffer) = stream.dequeue_buffer() else { return };
            let Some((w, h)) = data.size else { return };
            let Some(d) = buffer.datas_mut().first_mut() else { return };
            let chunk = d.chunk();
            let (offset, size) = (chunk.offset() as usize, chunk.size() as usize);
            let stride = match chunk.stride() {
                s if s > 0 => s as usize,
                _ => w * 2,
            };
            let Some(bytes) = d.data().and_then(|b| b.get(offset..offset + size)) else { return };
            if size == 0 || bytes.len() < stride * (h - 1) + w * 2 {
                return; // the daemon has no frame yet
            }
            data.shared.lock().unwrap().frame = Some(yuy2_half(bytes, stride, w, h));
            (data.repaint)();
        })
        .register()?;

    let format = serialize(Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties: vec![
            id_prop(FormatProperties::MediaType.as_raw(), MediaType::Video.as_raw()),
            id_prop(FormatProperties::MediaSubtype.as_raw(), MediaSubtype::Raw.as_raw()),
            id_prop(FormatProperties::VideoFormat.as_raw(), VideoFormat::YUY2.as_raw()),
        ],
    })?;
    // Only memory we can map (V4L2-style DMA-BUFs fail with -22 otherwise).
    let types = (1 << spa::sys::SPA_DATA_MemPtr) | (1 << spa::sys::SPA_DATA_MemFd);
    let buffers = serialize(Object {
        type_: SpaTypes::ObjectParamBuffers.as_raw(),
        id: ParamType::Buffers.as_raw(),
        properties: vec![Property {
            key: spa::sys::SPA_PARAM_BUFFERS_dataType,
            flags: PropertyFlags::empty(),
            value: Value::Choice(ChoiceValue::Int(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Flags { default: types, flags: vec![types] },
            ))),
        }],
    })?;
    let mut params = [
        Pod::from_bytes(&format).ok_or("bad format pod")?,
        Pod::from_bytes(&buffers).ok_or("bad buffers pod")?,
    ];
    stream.connect(
        Direction::Input,
        None,
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
        &mut params,
    )?;

    mainloop.run();
    let _ = stream.disconnect();
    Ok(())
}

/// Convert a YUY2 frame (BT.601, limited range) to RGBA at half its width
/// and height, by taking every other pixel and row: plenty for a preview,
/// and a quarter of the work.
pub fn yuy2_half(src: &[u8], stride: usize, width: usize, height: usize) -> ColorImage {
    let (w, h) = (width / 2, height / 2);
    let mut pixels = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &src[2 * y * stride..][..width * 2];
        // Each 4-byte group is Y0 U Y1 V for two pixels; keep the first.
        for group in row.chunks_exact(4).take(w) {
            let (y0, cb, cr) = (group[0] as f32, group[1] as f32 - 128.0, group[3] as f32 - 128.0);
            let luma = (y0 - 16.0) * (255.0 / 219.0);
            let chroma = 255.0 / 224.0;
            let r = luma + 1.402 * chroma * cr;
            let g = luma - (0.344136 * cb + 0.714136 * cr) * chroma;
            let b = luma + 1.772 * chroma * cb;
            let byte = |v: f32| v.clamp(0.0, 255.0) as u8;
            pixels.push(Color32::from_rgb(byte(r), byte(g), byte(b)));
        }
    }
    ColorImage::new([w, h], pixels)
}

fn id_prop(key: u32, id: u32) -> Property {
    Property { key, flags: PropertyFlags::empty(), value: Value::Id(Id(id)) }
}

fn serialize(object: Object) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Ok(PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(object))?
        .0
        .into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuy2_half_converts_bt601_and_halves() {
        // 4x2 YUY2: two pixel pairs per row. Top-left pair is BT.601 red,
        // top-right white; the second row is skipped by the halving.
        let red = [81, 90, 81, 240];
        let white = [235, 128, 235, 128];
        let black = [16, 128, 16, 128];
        let src: Vec<u8> = [red, white, black, black].concat();
        let image = yuy2_half(&src, 8, 4, 2);
        assert_eq!(image.size, [2, 1]);
        let close = |c: Color32, rgb: [u8; 3]| {
            let d = c.r().abs_diff(rgb[0]).max(c.g().abs_diff(rgb[1])).max(c.b().abs_diff(rgb[2]));
            assert!(d <= 2, "{c:?} vs {rgb:?}");
        };
        close(image.pixels[0], [255, 0, 0]);
        close(image.pixels[1], [255, 255, 255]);
    }
}
