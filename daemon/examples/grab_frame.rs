//! Grab one frame from a PipeWire video source and write it to a file as raw
//! bytes, for checking what the virtual camera produces.
//!
//! Usage: `cargo run --example grab_frame -- OUT [TARGET] [SKIP]`
//!
//! - `TARGET`: node name to capture from (default `video-menu.camera`).
//! - `SKIP`: frames to discard first so the camera's exposure settles
//!   (default 30).
//!
//! The negotiated format and size are printed; for YUY2 convert with e.g.
//! `ffmpeg -f rawvideo -pix_fmt yuyv422 -s WxH -i OUT out.png`.

use std::{cell::RefCell, error::Error, rc::Rc};

use pipewire as pw;
use pw::{
    context::ContextRc,
    main_loop::MainLoopRc,
    properties::properties,
    spa::{
        self,
        param::{
            ParamType,
            format::{FormatProperties, MediaSubtype, MediaType},
            video::VideoInfoRaw,
        },
        pod::{ChoiceValue, Object, Pod, Property, PropertyFlags, Value, serialize::PodSerializer},
        utils::{Choice, ChoiceEnum, ChoiceFlags, Direction, Id, SpaTypes},
    },
    stream::{StreamFlags, StreamRc},
};

#[derive(Default)]
struct Grab {
    info: Option<VideoInfoRaw>,
    seen: u32,
    done: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let out = args.next().ok_or("usage: grab_frame OUT [TARGET] [SKIP]")?;
    let target = args.next().unwrap_or_else(|| "video-menu.camera".into());
    let skip: u32 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(30);

    pw::init();
    let mainloop = MainLoopRc::new(None)?;
    let context = ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    let stream = StreamRc::new(
        core,
        "video-menu-grab",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Camera",
            "target.object" => target.as_str(),
        },
    )?;

    let grab = Rc::new(RefCell::new(Grab::default()));
    let _listener = stream
        .add_local_listener_with_user_data(grab.clone())
        .state_changed(|_, _, old, new| eprintln!("grab: {old:?} -> {new:?}"))
        .param_changed(|_, grab, id, param| {
            let Some(param) = param else { return };
            if id != ParamType::Format.as_raw() {
                return;
            }
            let mut info = VideoInfoRaw::new();
            if info.parse(param).is_ok() {
                let (size, rate) = (info.size(), info.framerate());
                println!(
                    "format {:?} {}x{} {}/{} fps",
                    info.format(),
                    size.width,
                    size.height,
                    rate.num,
                    rate.denom
                );
                grab.borrow_mut().info = Some(info);
            }
        })
        .process({
            let mainloop = mainloop.clone();
            move |stream, grab| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                let mut g = grab.borrow_mut();
                if g.done {
                    return;
                }
                let Some(data) = buffer.datas_mut().first_mut() else {
                    return;
                };
                let chunk = data.chunk();
                let (offset, size, stride) =
                    (chunk.offset() as usize, chunk.size() as usize, chunk.stride());
                if size == 0 {
                    return; // the relay hasn't produced a frame yet
                }
                g.seen += 1;
                if g.seen <= skip {
                    return;
                }
                let Some(bytes) = data.data().and_then(|d| d.get(offset..offset + size)) else {
                    return;
                };
                match std::fs::write(&out, bytes) {
                    Ok(()) => println!("wrote {size} bytes (stride {stride}) to {out}"),
                    Err(e) => eprintln!("grab: writing {out}: {e}"),
                }
                g.done = true;
                mainloop.quit();
            }
        })
        .register()?;

    // Accept any raw video format; the source decides.
    let any_raw = Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties: vec![
            Property {
                key: FormatProperties::MediaType.as_raw(),
                flags: PropertyFlags::empty(),
                value: Value::Id(Id(MediaType::Video.as_raw())),
            },
            Property {
                key: FormatProperties::MediaSubtype.as_raw(),
                flags: PropertyFlags::empty(),
                value: Value::Id(Id(MediaSubtype::Raw.as_raw())),
            },
        ],
    };
    let pod = PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(any_raw))?
        .0
        .into_inner();
    // Only accept buffer memory we can map (V4L2 sources default to DMA-BUF).
    let types = (1 << spa::sys::SPA_DATA_MemPtr) | (1 << spa::sys::SPA_DATA_MemFd);
    let buffers = Object {
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
    };
    let buffers =
        PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(buffers))?
            .0
            .into_inner();
    let mut params = [
        Pod::from_bytes(&pod).ok_or("bad pod")?,
        Pod::from_bytes(&buffers).ok_or("bad pod")?,
    ];
    stream.connect(
        Direction::Input,
        None,
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
        &mut params,
    )?;

    mainloop.run();
    if !grab.borrow().done {
        return Err("no frame captured".into());
    }
    Ok(())
}
