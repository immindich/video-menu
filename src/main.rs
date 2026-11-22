//! Virtual camera daemon.
//!
//! Exposes a PipeWire `Video/Source` node that mirrors a real webcam, flipped
//! vertically. The real camera is only opened while some client is actually
//! streaming from the virtual node; otherwise it stays idle (and suspended).
//!
//! Usage: `video-menu [CAMERA_NODE_NAME]`. Without an argument, the first
//! camera found in the graph is used.

mod model;

use std::{
    cell::{Cell, RefCell},
    error::Error,
    rc::Rc,
    time::Instant,
};

use pipewire as pw;
use pw::{
    context::ContextRc,
    core::CoreRc,
    main_loop::MainLoopRc,
    node::Node,
    properties::properties,
    spa,
    stream::{StreamFlags, StreamListener, StreamRc, StreamState},
    types::ObjectType,
};
use spa::{
    buffer::meta::MetaHeader,
    param::{
        ParamType,
        video::{VideoFormat, VideoInfoRaw},
    },
    pod::{ChoiceValue, Object, Pod, Property, PropertyFlags, Value, serialize::PodSerializer},
    utils::{Choice, ChoiceEnum, ChoiceFlags, Direction, SpaTypes},
};

const OUTPUT_NODE_NAME: &str = "video-menu.camera";
const OUTPUT_DESCRIPTION: &str = "Video Menu Camera";
const INPUT_NODE_NAME: &str = "video-menu.capture";

/// Mutable state of a [`CameraRelay`], shared by its stream callbacks.
#[derive(Default)]
struct State {
    /// Format negotiated with the consumer of the virtual camera.
    format: Option<VideoInfoRaw>,
    /// That same format as an EnumFormat pod, used to request it from the camera.
    format_pod: Option<Vec<u8>>,
    /// Row stride of frames we produce.
    stride: usize,
    /// Bytes per row that carry pixel data.
    row_bytes: usize,
    /// Most recent flipped frame, ready to hand to the consumer.
    frame: Vec<u8>,
    /// Presentation timestamp (ns, CLOCK_MONOTONIC) of `frame`.
    pts: i64,
    /// Frame sequence number of `frame`.
    seq: u64,
    /// Whether the capture stream is currently connected to the camera.
    capturing: bool,
}

/// Moves frames from a real camera to a virtual one, flipping them on the way.
///
/// `input` captures from the real camera; `output` is the virtual camera node
/// that applications read from. The camera is only captured from while an
/// application is streaming from `output`.
struct CameraRelay {
    input: StreamRc,
    output: StreamRc,
    state: RefCell<State>,
    /// Origin for fallback timestamps when the camera provides none.
    start: Instant,
}

/// Keeps a relay's stream callbacks registered; they are removed when dropped.
struct RelayListeners {
    _input: StreamListener<Rc<CameraRelay>>,
    _output: StreamListener<Rc<CameraRelay>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    pw::init();

    let target = std::env::args().nth(1);

    let mainloop = MainLoopRc::new(None)?;
    let context = ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    let (camera, formats) = probe_camera(&mainloop, &core, target.as_deref())?;
    eprintln!("using camera {camera} ({} formats)", formats.len());

    let relay = CameraRelay::new(&core, &camera)?;
    let _listeners = relay.listen()?;
    relay.publish(&formats)?;

    eprintln!("virtual camera {OUTPUT_NODE_NAME} ready");
    mainloop.run();
    Ok(())
}

impl CameraRelay {
    /// Create the capture and virtual camera streams. Nothing appears in the
    /// graph until [`publish`](Self::publish) is called.
    fn new(core: &CoreRc, camera: &str) -> Result<Rc<Self>, pw::Error> {
        let input = StreamRc::new(
            core.clone(),
            "video-menu-capture",
            properties! {
                *pw::keys::NODE_NAME => INPUT_NODE_NAME,
                *pw::keys::MEDIA_TYPE => "Video",
                *pw::keys::MEDIA_CATEGORY => "Capture",
                *pw::keys::MEDIA_ROLE => "Camera",
                "target.object" => camera,
            },
        )?;

        let output = StreamRc::new(
            core.clone(),
            "video-menu-camera",
            properties! {
                *pw::keys::NODE_NAME => OUTPUT_NODE_NAME,
                *pw::keys::NODE_DESCRIPTION => OUTPUT_DESCRIPTION,
                *pw::keys::MEDIA_CLASS => "Video/Source",
                *pw::keys::MEDIA_TYPE => "Video",
                *pw::keys::MEDIA_ROLE => "Camera",
            },
        )?;

        Ok(Rc::new(Self {
            input,
            output,
            state: RefCell::default(),
            start: Instant::now(),
        }))
    }

    /// Register the stream callbacks. Each listener holds its own `Rc` to the
    /// relay as user data, which the callbacks receive as `relay`.
    fn listen(self: &Rc<Self>) -> Result<RelayListeners, pw::Error> {
        let format = ParamType::Format.as_raw();

        let input = self
            .input
            .add_local_listener_with_user_data(self.clone())
            .state_changed(|_, _, old, new| eprintln!("capture: {old:?} -> {new:?}"))
            .param_changed(move |_, relay, id, param| {
                if id == format
                    && let Some(param) = param
                {
                    relay.on_input_format(param);
                }
            })
            .process(|_, relay| relay.on_input_frame())
            .register()?;

        let output = self
            .output
            .add_local_listener_with_user_data(self.clone())
            .state_changed(|_, relay, old, new| relay.on_output_state(old, new))
            .param_changed(move |_, relay, id, param| {
                if id == format {
                    relay.on_output_format(param);
                }
            })
            .process(|_, relay| relay.on_output_frame())
            .register()?;

        Ok(RelayListeners {
            _input: input,
            _output: output,
        })
    }

    /// Add the virtual camera to the graph, offering the camera's formats.
    fn publish(&self, formats: &[Vec<u8>]) -> Result<(), pw::Error> {
        let mut params: Vec<&Pod> = formats
            .iter()
            .filter_map(|bytes| Pod::from_bytes(bytes))
            .collect();
        self.output.connect(
            Direction::Output,
            None,
            StreamFlags::DRIVER | StreamFlags::MAP_BUFFERS,
            &mut params,
        )
    }

    // ---- input (capture) side ----

    /// The camera settled on a format; check it is the one we asked for.
    fn on_input_format(&self, param: &Pod) {
        let mut info = VideoInfoRaw::new();
        if info.parse(param).is_err() {
            return;
        }
        let (fmt, size) = (info.format(), info.size());
        eprintln!("capture: camera format {fmt:?} {}x{}", size.width, size.height);
        if let Some(want) = self.state.borrow().format
            && (want.format() != fmt || want.size() != size)
        {
            eprintln!("capture: camera format does not match requested format");
        }
    }

    /// A camera frame arrived: flip it into `state.frame`, then drive the
    /// virtual camera's graph so the consumer gets it.
    fn on_input_frame(&self) {
        let Some(mut buffer) = self.input.dequeue_buffer() else {
            return;
        };
        // Consumers (e.g. GStreamer's pipewiresrc) timestamp frames from the
        // header meta; without advancing pts they show a still image.
        let header = buffer.find_meta::<MetaHeader>().map(|h| (h.pts(), h.seq()));
        let datas = buffer.datas_mut();
        let Some(data) = datas.first_mut() else { return };
        let chunk = data.chunk();
        let (offset, size, src_stride) =
            (chunk.offset() as usize, chunk.size() as usize, chunk.stride());
        let Some(src) = data.data() else { return };
        let Some(src) = src.get(offset..offset + size) else {
            return;
        };

        {
            let st = &mut *self.state.borrow_mut();
            let Some(format) = st.format else { return };
            let height = format.size().height as usize;
            let src_stride = if src_stride > 0 {
                src_stride as usize
            } else {
                st.row_bytes
            };
            st.frame.resize(st.stride * height, 0);
            if !flip_vertical(src, src_stride, &mut st.frame, st.stride, st.row_bytes, height) {
                eprintln!("capture: short frame ({size} bytes), dropping");
                return;
            }
            (st.pts, st.seq) = match header {
                Some((pts, seq)) if pts > 0 => (pts, seq),
                _ => (self.start.elapsed().as_nanos() as i64, st.seq.wrapping_add(1)),
            };
        }
        drop(buffer);

        if let Err(e) = self.output.trigger_process() {
            eprintln!("camera: trigger_process failed: {e}");
        }
    }

    /// Connect the capture stream to the camera with the consumer's format.
    fn start_capture(&self) {
        let pod = {
            let st = self.state.borrow();
            if st.capturing {
                return;
            }
            match &st.format_pod {
                Some(pod) => pod.clone(),
                None => return,
            }
        };
        let mut params = [Pod::from_bytes(&pod).unwrap()];
        match self.input.connect(
            Direction::Input,
            None,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
            &mut params,
        ) {
            Ok(()) => {
                eprintln!("capture: started");
                self.state.borrow_mut().capturing = true;
            }
            Err(e) => eprintln!("capture: connect failed: {e}"),
        }
    }

    /// Disconnect from the camera so it can go idle.
    fn stop_capture(&self) {
        {
            let mut st = self.state.borrow_mut();
            if !st.capturing {
                return;
            }
            st.capturing = false;
            st.frame.clear();
        }
        if let Err(e) = self.input.disconnect() {
            eprintln!("capture: disconnect failed: {e}");
        }
        eprintln!("capture: stopped");
    }

    // ---- output (virtual camera) side ----

    /// Capture only while an application is actually streaming from us.
    fn on_output_state(&self, old: StreamState, new: StreamState) {
        eprintln!("camera: {old:?} -> {new:?}");
        match new {
            StreamState::Streaming => self.start_capture(),
            _ => self.stop_capture(),
        }
    }

    /// The consumer settled on a format (or unlinked, if `param` is `None`).
    fn on_output_format(&self, param: Option<&Pod>) {
        // Any format change invalidates the running capture.
        self.stop_capture();

        let mut st = self.state.borrow_mut();
        st.format = None;
        st.format_pod = None;
        let Some(param) = param else { return };

        let mut info = VideoInfoRaw::new();
        if info.parse(param).is_err() {
            return;
        }
        let size = info.size();
        let Some(bpp) = bytes_per_pixel(info.format()) else {
            eprintln!("camera: unsupported format {:?}", info.format());
            return;
        };
        eprintln!(
            "camera: consumer format {:?} {}x{}",
            info.format(),
            size.width,
            size.height
        );

        st.row_bytes = size.width as usize * bpp;
        st.stride = (st.row_bytes + 3) & !3;
        st.format = Some(info);
        st.format_pod = Some(as_enum_format(param));

        let buffers = serialize(buffers_param(
            st.stride as i32,
            (st.stride * size.height as usize) as i32,
        ));
        drop(st);
        let meta = serialize(header_meta_param());
        let mut params = [
            Pod::from_bytes(&buffers).unwrap(),
            Pod::from_bytes(&meta).unwrap(),
        ];
        if let Err(e) = self.output.update_params(&mut params) {
            eprintln!("camera: update_params failed: {e}");
        }
    }

    /// The virtual camera's graph is running a cycle: hand out the latest frame.
    fn on_output_frame(&self) {
        // The safe `Buffer` wrapper only exposes metadata read-only, so fill
        // the buffer through the raw API to set the header meta.
        unsafe {
            let buffer = self.output.dequeue_raw_buffer();
            if buffer.is_null() {
                return;
            }
            fill_output_buffer((*buffer).buffer, &self.state.borrow());
            self.output.queue_raw_buffer(buffer);
        }
    }
}

/// Copy `height` rows of `row_bytes` from `src` into `dst` in reverse order.
/// Returns false if `src` is too short to hold the frame.
fn flip_vertical(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    row_bytes: usize,
    height: usize,
) -> bool {
    if height == 0 || src.len() < (height - 1) * src_stride + row_bytes {
        return false;
    }
    for (y, src_row) in src.chunks(src_stride).take(height).enumerate() {
        let dst_row = &mut dst[(height - 1 - y) * dst_stride..][..row_bytes];
        dst_row.copy_from_slice(&src_row[..row_bytes]);
    }
    true
}

/// Bytes per pixel for single-plane packed formats, which can be flipped by
/// reversing rows. Planar formats are not supported.
fn bytes_per_pixel(format: VideoFormat) -> Option<usize> {
    use VideoFormat as F;
    let packed: [(&[VideoFormat], usize); 4] = [
        (
            &[F::RGBx, F::BGRx, F::xRGB, F::xBGR, F::RGBA, F::BGRA, F::ARGB, F::ABGR],
            4,
        ),
        (&[F::RGB, F::BGR], 3),
        (&[F::YUY2, F::UYVY, F::YVYU, F::VYUY], 2),
        (&[F::GRAY8], 1),
    ];
    packed
        .iter()
        .find(|(formats, _)| formats.contains(&format))
        .map(|&(_, bpp)| bpp)
}

/// Re-tag a negotiated Format pod as an EnumFormat pod so it can be offered to
/// `pw_stream_connect`.
fn as_enum_format(format: &Pod) -> Vec<u8> {
    let mut bytes = format.as_bytes().to_vec();
    // struct spa_pod_object { struct spa_pod pod; struct spa_pod_object_body { u32 type; u32 id; } }
    bytes[12..16].copy_from_slice(&ParamType::EnumFormat.as_raw().to_ne_bytes());
    bytes
}

fn buffers_param(stride: i32, size: i32) -> Object {
    let int = |key, value| Property {
        key,
        flags: PropertyFlags::empty(),
        value: Value::Int(value),
    };
    Object {
        type_: SpaTypes::ObjectParamBuffers.as_raw(),
        id: ParamType::Buffers.as_raw(),
        properties: vec![
            Property {
                key: spa::sys::SPA_PARAM_BUFFERS_buffers,
                flags: PropertyFlags::empty(),
                value: Value::Choice(ChoiceValue::Int(Choice(
                    ChoiceFlags::empty(),
                    ChoiceEnum::Range {
                        default: 4,
                        min: 2,
                        max: 16,
                    },
                ))),
            },
            int(spa::sys::SPA_PARAM_BUFFERS_blocks, 1),
            int(spa::sys::SPA_PARAM_BUFFERS_size, size),
            int(spa::sys::SPA_PARAM_BUFFERS_stride, stride),
        ],
    }
}

/// Ask for a `spa_meta_header` on each output buffer, which carries the
/// frame's timestamp and sequence number.
fn header_meta_param() -> Object {
    Object {
        type_: SpaTypes::ObjectParamMeta.as_raw(),
        id: ParamType::Meta.as_raw(),
        properties: vec![
            Property {
                key: spa::sys::SPA_PARAM_META_type,
                flags: PropertyFlags::empty(),
                value: Value::Id(spa::utils::Id(spa::sys::SPA_META_Header)),
            },
            Property {
                key: spa::sys::SPA_PARAM_META_size,
                flags: PropertyFlags::empty(),
                value: Value::Int(size_of::<spa::sys::spa_meta_header>() as i32),
            },
        ],
    }
}

/// Copy the latest flipped frame and its timestamp into an output buffer.
///
/// # Safety
/// `buffer` must be a valid, mapped buffer dequeued from the output stream.
unsafe fn fill_output_buffer(buffer: *mut spa::sys::spa_buffer, st: &State) {
    unsafe {
        let header = spa::sys::spa_buffer_find_meta_data(
            buffer,
            spa::sys::SPA_META_Header,
            size_of::<spa::sys::spa_meta_header>(),
        ) as *mut spa::sys::spa_meta_header;
        if !header.is_null() {
            header.write(spa::sys::spa_meta_header {
                flags: 0,
                offset: 0,
                pts: st.pts,
                dts_offset: 0,
                seq: st.seq,
            });
        }

        if (*buffer).n_datas == 0 || (*buffer).datas.is_null() {
            return;
        }
        let data = &mut *(*buffer).datas;
        let len = if !data.data.is_null()
            && !st.frame.is_empty()
            && data.maxsize as usize >= st.frame.len()
        {
            std::ptr::copy_nonoverlapping(st.frame.as_ptr(), data.data.cast(), st.frame.len());
            st.frame.len()
        } else {
            0
        };
        if let Some(chunk) = data.chunk.as_mut() {
            chunk.offset = 0;
            chunk.stride = st.stride as i32;
            chunk.size = len as u32;
        }
    }
}

fn serialize(object: Object) -> Vec<u8> {
    PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(object))
        .expect("pod serialization")
        .0
        .into_inner()
}

/// Run the loop until the server has processed everything sent so far.
fn roundtrip(mainloop: &MainLoopRc, core: &CoreRc) -> Result<(), pw::Error> {
    let done = Rc::new(Cell::new(false));
    let pending = core.sync(0)?;
    let _listener = core
        .add_listener_local()
        .done({
            let done = done.clone();
            let mainloop = mainloop.clone();
            move |id, seq| {
                if id == pw::core::PW_ID_CORE && seq == pending {
                    done.set(true);
                    mainloop.quit();
                }
            }
        })
        .register();
    while !done.get() {
        mainloop.run();
    }
    Ok(())
}

/// Find the camera node and collect the formats it supports, so the virtual
/// camera can advertise the same ones without opening the device.
fn probe_camera(
    mainloop: &MainLoopRc,
    core: &CoreRc,
    target: Option<&str>,
) -> Result<(String, Vec<Vec<u8>>), Box<dyn Error>> {
    struct Candidate {
        name: String,
        is_camera: bool,
        global: pw::registry::GlobalObject<pw::properties::PropertiesBox>,
    }

    let registry = core.get_registry()?;
    let candidates = Rc::new(RefCell::new(Vec::new()));
    let _registry_listener = registry
        .add_listener_local()
        .global({
            let candidates = candidates.clone();
            move |global| {
                if global.type_ != ObjectType::Node {
                    return;
                }
                let Some(props) = global.props else { return };
                if props.get(*pw::keys::MEDIA_CLASS) != Some("Video/Source") {
                    return;
                }
                let Some(name) = props.get(*pw::keys::NODE_NAME) else {
                    return;
                };
                if name == OUTPUT_NODE_NAME {
                    return;
                }
                candidates.borrow_mut().push(Candidate {
                    name: name.to_owned(),
                    is_camera: props.get(*pw::keys::MEDIA_ROLE) == Some("Camera"),
                    global: global.to_owned(),
                });
            }
        })
        .register();
    roundtrip(mainloop, core)?;

    let candidates = candidates.borrow();
    let camera = match target {
        Some(target) => candidates.iter().find(|c| c.name == target),
        None => candidates
            .iter()
            .find(|c| c.is_camera)
            .or(candidates.first()),
    };
    let Some(camera) = camera else {
        let names: Vec<&str> = candidates.iter().map(|c| c.name.as_str()).collect();
        return Err(format!("camera not found; available video sources: {names:?}").into());
    };

    let node: Node = registry.bind(&camera.global)?;
    let formats = Rc::new(RefCell::new(Vec::new()));
    let _node_listener = node
        .add_listener_local()
        .param({
            let formats = formats.clone();
            move |_, id, _, _, param| {
                if let (ParamType::EnumFormat, Some(param)) = (id, param) {
                    formats.borrow_mut().push(param.as_bytes().to_vec());
                }
            }
        })
        .register();
    node.enum_params(0, Some(ParamType::EnumFormat), 0, u32::MAX);
    roundtrip(mainloop, core)?;

    let formats = formats.take();
    if formats.is_empty() {
        return Err(format!("camera {} reported no formats", camera.name).into());
    }
    Ok((camera.name.clone(), formats))
}
