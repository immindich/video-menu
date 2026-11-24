//! Relay that mirrors a real camera onto a virtual PipeWire camera node.
//!
//! Capture and output negotiate formats independently: the camera delivers
//! whatever supported format it prefers (decoding MJPG if it must), each
//! frame is unpacked to planar RGB
//! (see [`crate::pipeline`]), run through the effects, and packed as YUY2
//! by a [`Processor`] on the GPU or CPU
//! (BT.601, limited range) for applications.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    error::Error,
    rc::Rc,
    time::{Duration, Instant},
};

use pipewire as pw;
use pw::{
    core::CoreRc,
    main_loop::MainLoopRc,
    node::Node,
    node::NodeListener,
    properties::properties,
    registry::RegistryRc,
    spa,
    stream::{StreamFlags, StreamListener, StreamRc, StreamState},
    types::ObjectType,
};
use spa::{
    buffer::meta::MetaHeader,
    param::{
        ParamType,
        format::{MediaSubtype, MediaType},
        video::{VideoFormat, VideoInfoRaw},
    },
    pod::{ChoiceValue, Object, Pod, Property, PropertyFlags, Value},
    utils::{Choice, ChoiceEnum, ChoiceFlags, Direction, SpaTypes, result::SpaResult},
};

use crate::{
    formats::{self, CameraMode, serialize},
    formats::CaptureFormat,
    pipeline::InputFormat,
    processor::{FrameFormat, Processor},
};

pub const OUTPUT_NODE_NAME: &str = "video-menu.camera";
const OUTPUT_DESCRIPTION: &str = "Video Menu Camera";
const INPUT_NODE_NAME: &str = "video-menu.capture";

/// How often per-stage frame timings are logged.
const TIMING_INTERVAL: Duration = Duration::from_secs(5);

/// Mutable state of a [`CameraRelay`], shared by its stream callbacks.
#[derive(Default)]
struct State {
    /// Format negotiated with the consumer of the virtual camera (YUY2).
    format: Option<VideoInfoRaw>,
    /// Row stride of frames we produce.
    stride: usize,
    /// Format the camera is delivering, once capture has negotiated.
    input: Option<FrameFormat>,
    /// Most recent processed frame as YUY2, ready to hand to the consumer.
    frame: Vec<u8>,
    /// Presentation timestamp (ns, CLOCK_MONOTONIC) of `frame`.
    pts: i64,
    /// Frame sequence number of `frame`.
    seq: u64,
    /// Whether the capture stream is currently connected to the camera.
    capturing: bool,
    timing: Timing,
}

/// Accumulates per-stage processing times and logs their averages.
#[derive(Default)]
struct Timing {
    frames: u32,
    stages: [Duration; 4],
    since: Option<Instant>,
}

impl Timing {
    fn record(&mut self, stages: [Duration; 4], names: [&str; 4]) {
        self.frames += 1;
        for (total, t) in self.stages.iter_mut().zip(stages) {
            *total += t;
        }
        let since = *self.since.get_or_insert_with(Instant::now);
        if since.elapsed() >= TIMING_INTERVAL {
            let ms = |d: Duration| d.as_secs_f64() * 1e3 / self.frames as f64;
            let total: Duration = self.stages.iter().sum();
            // Skip stages that took no time at all (decode, for raw frames).
            let stages: Vec<String> = names
                .iter()
                .zip(self.stages)
                .filter(|(_, t)| !t.is_zero())
                .map(|(n, t)| format!("{n} {:.2}", ms(t)))
                .collect();
            eprintln!(
                "pipeline: {} frames, {:.2} ms/frame ({})",
                self.frames,
                ms(total),
                stages.join(", ")
            );
            *self = Self { since: Some(Instant::now()), ..Self::default() };
        }
    }
}

/// Moves frames from a real camera to a virtual one, flipping them on the way.
///
/// `input` captures from the real camera; `output` is the virtual camera node
/// that applications read from. The camera is only captured from while an
/// application is streaming from `output`. The relay may have no camera
/// selected, in which case `output` advertises generic formats and produces
/// no frames; see [`set_camera`](Self::set_camera).
pub struct CameraRelay {
    input: StreamRc,
    output: StreamRc,
    state: RefCell<State>,
    /// Turns camera frames into processed YUY2 frames.
    processor: RefCell<Processor>,
    /// Name of the video source being relayed, if any.
    camera: RefCell<Option<String>>,
    /// Live list of video sources the relay could capture from.
    sources: VideoSources,
    /// Origin for fallback timestamps when the camera provides none.
    start: Instant,
}

/// Keeps a relay's stream callbacks registered; they are removed when dropped.
pub struct RelayListeners {
    _input: StreamListener<Rc<CameraRelay>>,
    _output: StreamListener<Rc<CameraRelay>>,
}

impl CameraRelay {
    /// Create the capture and virtual camera streams, with no camera
    /// selected. Nothing appears in the graph until
    /// [`publish`](Self::publish) is called.
    pub fn new(
        core: &CoreRc,
        sources: VideoSources,
        processor: Processor,
    ) -> Result<Rc<Self>, pw::Error> {
        let input = StreamRc::new(
            core.clone(),
            "video-menu-capture",
            properties! {
                *pw::keys::NODE_NAME => INPUT_NODE_NAME,
                *pw::keys::MEDIA_TYPE => "Video",
                *pw::keys::MEDIA_CATEGORY => "Capture",
                *pw::keys::MEDIA_ROLE => "Camera",
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
            processor: RefCell::new(processor),
            camera: RefCell::default(),
            sources,
            start: Instant::now(),
        }))
    }

    /// The video sources currently in the graph.
    pub fn sources(&self) -> &VideoSources {
        &self.sources
    }

    /// Name of the video source being relayed, if any.
    pub fn camera(&self) -> Option<String> {
        self.camera.borrow().clone()
    }

    /// Relay from the video source called `name`, or from nothing if `None`.
    ///
    /// The virtual camera's advertised formats are replaced with ones the new
    /// source can supply (or generic ones), so a consumer that is already
    /// streaming renegotiates if its format is no longer offered. Fails,
    /// leaving the current selection alone, if `name` is unknown or offers no
    /// format we can capture (including when its formats haven't arrived yet).
    pub fn set_camera(&self, name: Option<&str>) -> Result<(), Box<dyn Error>> {
        let formats = match name {
            Some(name) => {
                let pods = self
                    .sources
                    .formats(name)
                    .ok_or_else(|| format!("no video source named {name}"))?;
                let modes = formats::camera_modes(&pods);
                if modes.is_empty() {
                    return Err(
                        format!("video source {name} offers no format we can capture").into(),
                    );
                }
                formats::output_formats(&modes)
            }
            None => formats::generic_output_formats(),
        };

        self.stop_capture();
        self.set_target(name.unwrap_or(""))?;
        *self.camera.borrow_mut() = name.map(str::to_owned);
        match name {
            Some(name) => eprintln!("relay: using {name} ({} output sizes)", formats.len()),
            None => eprintln!("relay: no camera"),
        }

        let mut params: Vec<&Pod> = formats
            .iter()
            .filter_map(|bytes| Pod::from_bytes(bytes))
            .collect();
        self.output.update_params(&mut params)?;

        self.resume_capture();
        Ok(())
    }

    /// Capturable modes of the selected camera; empty if there is none.
    fn camera_modes(&self) -> Vec<CameraMode> {
        self.camera()
            .and_then(|name| self.sources.formats(&name))
            .map(|pods| formats::camera_modes(&pods))
            .unwrap_or_default()
    }

    /// Point the capture stream at `target`. Takes effect on the next connect.
    fn set_target(&self, target: &str) -> Result<(), pw::Error> {
        let props = properties! { "target.object" => target };
        // The safe stream wrapper does not expose pw_stream_update_properties.
        let res = unsafe {
            pw::sys::pw_stream_update_properties(self.input.as_raw_ptr(), props.dict().as_raw_ptr())
        };
        SpaResult::from_c(res).into_sync_result()?;
        Ok(())
    }

    /// Register the stream callbacks. Each listener holds its own `Rc` to the
    /// relay as user data, which the callbacks receive as `relay`.
    pub fn listen(self: &Rc<Self>) -> Result<RelayListeners, pw::Error> {
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

    /// Add the virtual camera to the graph, offering YUY2 at the sizes the
    /// selected camera supports, or generic sizes if no camera is selected.
    pub fn publish(&self) -> Result<(), pw::Error> {
        let modes = self.camera_modes();
        let formats = if modes.is_empty() {
            formats::generic_output_formats()
        } else {
            formats::output_formats(&modes)
        };
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

    /// The camera settled on a format: remember how to unpack it.
    fn on_input_format(&self, param: &Pod) {
        let mut st = self.state.borrow_mut();
        st.input = None;
        let Some((capture, width, height)) = formats::negotiated(param) else {
            eprintln!("capture: camera chose a format we cannot capture");
            return;
        };
        eprintln!("capture: camera format {capture:?} {width}x{height}");
        let input = match capture {
            CaptureFormat::Mjpg => FrameFormat::Mjpg,
            CaptureFormat::Raw(fmt) => {
                let Some(input) = InputFormat::from_video_format(fmt) else {
                    eprintln!("capture: cannot unpack {fmt:?}");
                    return;
                };
                let mut info = VideoInfoRaw::new();
                let matrix = match info.parse(param) {
                    Ok(_) => info.color_matrix(),
                    Err(_) => spa::sys::SPA_VIDEO_COLOR_MATRIX_UNKNOWN,
                };
                if input == InputFormat::Yuy2
                    && matrix != spa::sys::SPA_VIDEO_COLOR_MATRIX_UNKNOWN
                    && matrix != spa::sys::SPA_VIDEO_COLOR_MATRIX_BT601
                {
                    eprintln!("capture: camera uses color matrix {matrix}; decoding as BT.601");
                }
                FrameFormat::Raw(input)
            }
        };
        if let Some(out) = st.format
            && (out.size().width, out.size().height) != (width, height)
        {
            eprintln!(
                "capture: camera size does not match consumer's {}x{}",
                out.size().width,
                out.size().height
            );
            return;
        }
        st.input = Some(input);
    }

    /// A camera frame arrived: decode or unpack it to planar RGB, apply the effects,
    /// pack it into `state.frame` as YUY2, then drive the virtual camera's
    /// graph so the consumer gets it.
    fn on_input_frame(&self) {
        let Some(mut buffer) = self.input.dequeue_buffer() else {
            return;
        };
        // Consumers (e.g. GStreamer's pipewiresrc) timestamp frames from the
        // header meta; without advancing pts they show a still image.
        let header = buffer.find_meta::<MetaHeader>().map(|h| (h.pts(), h.seq()));
        let datas = buffer.datas_mut();
        let Some(data) = datas.first_mut() else {
            return;
        };
        let chunk = data.chunk();
        let (offset, size, src_stride) = (
            chunk.offset() as usize,
            chunk.size() as usize,
            chunk.stride(),
        );
        let Some(src) = data.data() else { return };
        let Some(src) = src.get(offset..offset + size) else {
            return;
        };

        {
            let st = &mut *self.state.borrow_mut();
            let (Some(format), Some(input)) = (st.format, st.input) else {
                return;
            };
            let (width, height) = (format.size().width as usize, format.size().height as usize);
            let src_stride = match input {
                _ if src_stride > 0 => src_stride as usize,
                FrameFormat::Raw(raw) => raw.row_bytes(width),
                FrameFormat::Mjpg => 0,
            };

            st.frame.resize(st.stride * height, 0);
            let mut processor = self.processor.borrow_mut();
            match processor.process(
                input,
                src,
                src_stride,
                width,
                height,
                &mut st.frame,
                st.stride,
            ) {
                Ok(stages) => st.timing.record(stages, processor.stage_names()),
                Err(e) => {
                    eprintln!("capture: dropping frame: {e}");
                    return;
                }
            }

            (st.pts, st.seq) = match header {
                Some((pts, seq)) if pts > 0 => (pts, seq),
                _ => (
                    self.start.elapsed().as_nanos() as i64,
                    st.seq.wrapping_add(1),
                ),
            };
        }
        drop(buffer);

        if let Err(e) = self.output.trigger_process() {
            eprintln!("camera: trigger_process failed: {e}");
        }
    }

    /// Connect the capture stream to the camera, asking for a format we can
    /// unpack at the consumer's size and frame rate.
    fn start_capture(&self) {
        let format = {
            let st = self.state.borrow();
            if st.capturing {
                return;
            }
            match st.format {
                Some(format) => format,
                None => return,
            }
        };
        let modes = self.camera_modes();
        if modes.is_empty() {
            return;
        }
        let (size, rate) = (format.size(), format.framerate());
        let pods = formats::capture_formats(&modes, size.width, size.height, (rate.num, rate.denom));
        if pods.is_empty() {
            eprintln!(
                "capture: camera offers nothing at {}x{} {}/{} fps",
                size.width, size.height, rate.num, rate.denom
            );
            return;
        }
        let buffers = serialize(input_buffers_param());
        let mut params: Vec<&Pod> = pods.iter().filter_map(|p| Pod::from_bytes(p)).collect();
        params.extend(Pod::from_bytes(&buffers));
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

    /// Start capturing if a consumer is streaming from us. Covers cases where
    /// no `Streaming` state transition follows, such as a camera switch or a
    /// renegotiation while already streaming.
    fn resume_capture(&self) {
        if self.output.state() == StreamState::Streaming {
            self.start_capture();
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
            st.input = None;
            st.frame.clear();
            st.timing = Timing::default();
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
        let Some(param) = param else { return };

        let mut info = VideoInfoRaw::new();
        if info.parse(param).is_err() {
            return;
        }
        let (size, rate) = (info.size(), info.framerate());
        eprintln!(
            "camera: consumer format {:?} {}x{} {}/{} fps",
            info.format(),
            size.width,
            size.height,
            rate.num,
            rate.denom
        );
        // We only ever offer YUY2, at even widths.
        if info.format() != VideoFormat::YUY2 || size.width % 2 != 0 {
            eprintln!("camera: unexpected consumer format, not producing frames");
            return;
        }

        st.stride = (size.width as usize * 2 + 3) & !3;
        st.format = Some(info);

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
        self.resume_capture();
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

/// Buffer memory the capture stream accepts: memory it can map. Without this,
/// V4L2 cameras offer DMA-BUFs and linking fails with "mix use buffers
/// failed: -22".
fn input_buffers_param() -> Object {
    let types = (1 << spa::sys::SPA_DATA_MemPtr) | (1 << spa::sys::SPA_DATA_MemFd);
    Object {
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

/// Copy the latest processed frame and its timestamp into an output buffer.
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

/// Run the loop until the server has processed everything sent so far.
pub fn roundtrip(mainloop: &MainLoopRc, core: &CoreRc) -> Result<(), pw::Error> {
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

/// A `Video/Source` node in the graph. `is_camera` marks nodes with the
/// Camera media role, which are preferred when picking what to relay.
pub struct VideoSource {
    pub name: String,
    pub is_camera: bool,
    /// EnumFormat pods the node supports, filled in asynchronously after the
    /// node appears.
    formats: Rc<RefCell<Vec<Vec<u8>>>>,
    _node: Node,
    _node_listener: NodeListener,
}

/// Live list of the graph's `Video/Source` nodes, excluding our own virtual
/// camera, with the formats each supports. Kept up to date by registry
/// events for as long as it is alive.
pub struct VideoSources {
    /// Owned here; the registry callback only holds a weak reference, which
    /// avoids a cycle through the listener.
    _registry: RegistryRc,
    list: Rc<RefCell<BTreeMap<u32, VideoSource>>>,
    _listener: pw::registry::Listener,
}

impl VideoSources {
    /// Start tracking video sources. Sources and their formats are filled in
    /// as the main loop runs: after one [`roundtrip`] the initial sources are
    /// known, after a second their formats are too.
    pub fn watch(core: &CoreRc) -> Result<Self, pw::Error> {
        let registry = core.get_registry_rc()?;
        let list = Rc::new(RefCell::new(BTreeMap::new()));
        let listener = registry
            .add_listener_local()
            .global({
                let list = list.clone();
                let registry = registry.downgrade();
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
                    let Some(registry) = registry.upgrade() else {
                        return;
                    };
                    let node: Node = match registry.bind(global) {
                        Ok(node) => node,
                        Err(e) => {
                            eprintln!("sources: cannot bind {name}: {e}");
                            return;
                        }
                    };
                    let formats = Rc::new(RefCell::new(Vec::new()));
                    let node_listener = node
                        .add_listener_local()
                        .param({
                            let formats = formats.clone();
                            move |_, id, _, _, param| {
                                let Some(param) = param else { return };
                                let is_video = matches!(
                                    spa::param::format_utils::parse_format(param),
                                    Ok((MediaType::Video, MediaSubtype::Raw | MediaSubtype::Mjpg))
                                );
                                if id == ParamType::EnumFormat && is_video {
                                    formats.borrow_mut().push(param.as_bytes().to_vec());
                                }
                            }
                        })
                        .register();
                    node.enum_params(0, Some(ParamType::EnumFormat), 0, u32::MAX);

                    eprintln!("sources: added {name}");
                    list.borrow_mut().insert(
                        global.id,
                        VideoSource {
                            name: name.to_owned(),
                            is_camera: props.get(*pw::keys::MEDIA_ROLE) == Some("Camera"),
                            formats,
                            _node: node,
                            _node_listener: node_listener,
                        },
                    );
                }
            })
            .global_remove({
                let list = list.clone();
                move |id| {
                    if let Some(source) = list.borrow_mut().remove(&id) {
                        eprintln!("sources: removed {}", source.name);
                    }
                }
            })
            .register();
        Ok(Self {
            _registry: registry,
            list,
            _listener: listener,
        })
    }

    /// Names of the currently known video sources and whether each has the
    /// Camera role.
    pub fn list(&self) -> Vec<(String, bool)> {
        self.list
            .borrow()
            .values()
            .map(|s| (s.name.clone(), s.is_camera))
            .collect()
    }

    /// Formats supported by the source called `name`, if it exists. Empty if
    /// they have not been received yet.
    pub fn formats(&self, name: &str) -> Option<Vec<Vec<u8>>> {
        self.list
            .borrow()
            .values()
            .find(|s| s.name == name)
            .map(|s| s.formats.borrow().clone())
    }

    /// Pick the source to relay at startup: `target` by name if given,
    /// otherwise the first camera, otherwise the first video source.
    /// Returns `None` if there are no sources and no target was asked for.
    pub fn select(&self, target: Option<&str>) -> Result<Option<String>, Box<dyn Error>> {
        let list = self.list.borrow();
        let found = match target {
            Some(target) => match list.values().find(|s| s.name == target) {
                Some(source) => Some(source),
                None => {
                    let names: Vec<&str> = list.values().map(|s| s.name.as_str()).collect();
                    return Err(format!(
                        "camera {target} not found; available video sources: {names:?}"
                    )
                    .into());
                }
            },
            None => list
                .values()
                .find(|s| s.is_camera)
                .or(list.values().next()),
        };
        Ok(found.map(|s| s.name.clone()))
    }
}
