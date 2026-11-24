//! Format negotiation helpers: reading the modes a camera offers, and
//! building the EnumFormat pods for the virtual camera (always YUY2) and for
//! the capture stream (whatever camera format we can unpack or decode).

use pipewire::spa::{
    self,
    param::{
        ParamType,
        format::{FormatProperties, MediaSubtype, MediaType},
        video::VideoFormat,
    },
    pod::Pod,
    pod::{
        ChoiceValue, Object, Property, PropertyFlags, Value, deserialize::PodDeserializer,
        serialize::PodSerializer,
    },
    utils::{Choice, ChoiceEnum, ChoiceFlags, Fraction, Id, Rectangle, SpaTypes},
};

use crate::pipeline::InputFormat;

/// Sizes the virtual camera offers when no camera is selected.
const GENERIC_SIZES: [(u32, u32); 3] = [(1920, 1080), (1280, 720), (640, 360)];

/// A camera format we can capture from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureFormat {
    /// Uncompressed, in one of [`InputFormat::SUPPORTED`].
    Raw(VideoFormat),
    /// Motion JPEG, which has to be decoded first.
    Mjpg,
}

impl CaptureFormat {
    /// Lower is preferred: raw formats in [`InputFormat::SUPPORTED`] order,
    /// then MJPG, which costs a decode per frame.
    fn rank(self) -> usize {
        match self {
            Self::Raw(f) => InputFormat::SUPPORTED.iter().position(|&s| s == f).unwrap_or(usize::MAX),
            Self::Mjpg => InputFormat::SUPPORTED.len(),
        }
    }
}

/// One format/size combination a camera supports.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraMode {
    pub format: CaptureFormat,
    pub width: u32,
    pub height: u32,
    /// Frame rates offered at this format and size, or `None` if the camera
    /// doesn't say (libcamera), meaning any rate.
    pub rates: Option<Vec<(u32, u32)>>,
}

impl CameraMode {
    fn offers(&self, rate: (u32, u32)) -> bool {
        self.rates
            .as_ref()
            .is_some_and(|rates| rates.iter().any(|&r| same_rate(r, rate)))
    }
}

/// The contents of a video format pod that we care about.
struct ParsedFormat {
    /// Formats we can capture, in the order listed.
    formats: Vec<CaptureFormat>,
    sizes: Vec<(u32, u32)>,
    rates: Option<Vec<(u32, u32)>>,
}

fn parse_format(pod: &[u8]) -> Option<ParsedFormat> {
    let Ok((_, Value::Object(object))) = PodDeserializer::deserialize_any_from(pod) else {
        return None;
    };
    let prop = |key: FormatProperties| {
        object
            .properties
            .iter()
            .find(|p| p.key == key.as_raw())
            .map(|p| &p.value)
    };
    let subtype = prop(FormatProperties::MediaSubtype).map(ids).unwrap_or_default();
    let formats = match subtype.as_slice() {
        [s] if *s == MediaSubtype::Raw.as_raw() => prop(FormatProperties::VideoFormat)
            .map(ids)
            .unwrap_or_default()
            .into_iter()
            .map(VideoFormat::from_raw)
            .filter(|f| InputFormat::SUPPORTED.contains(f))
            .map(CaptureFormat::Raw)
            .collect(),
        [s] if *s == MediaSubtype::Mjpg.as_raw() => vec![CaptureFormat::Mjpg],
        _ => Vec::new(),
    };
    Some(ParsedFormat {
        formats,
        sizes: prop(FormatProperties::VideoSize)
            .map(rectangles)
            .unwrap_or_default(),
        rates: prop(FormatProperties::VideoFramerate).and_then(fractions),
    })
}

/// Expand a camera's EnumFormat pods into the modes we can capture from.
/// Formats we can't unpack or decode, and sizes given as ranges, are skipped.
pub fn camera_modes(pods: &[Vec<u8>]) -> Vec<CameraMode> {
    let mut modes = Vec::new();
    for parsed in pods.iter().filter_map(|pod| parse_format(pod)) {
        for &format in &parsed.formats {
            for &(width, height) in &parsed.sizes {
                modes.push(CameraMode { format, width, height, rates: parsed.rates.clone() });
            }
        }
    }
    modes
}

/// The format and size a camera settled on, from its negotiated Format pod.
/// `None` if it isn't one we can capture.
pub fn negotiated(pod: &Pod) -> Option<(CaptureFormat, u32, u32)> {
    let parsed = parse_format(pod.as_bytes())?;
    let (&format, &(width, height)) = (parsed.formats.first()?, parsed.sizes.first()?);
    Some((format, width, height))
}

/// The virtual camera's formats for a camera with `modes`: YUY2 at every size
/// the camera can supply, largest first, at the rates it supports there.
pub fn output_formats(modes: &[CameraMode]) -> Vec<Vec<u8>> {
    let mut sizes: Vec<(u32, u32)> = modes.iter().map(|m| (m.width, m.height)).collect();
    // Sizes with equal area (e.g. 960x640 and 1024x600) must still end up
    // adjacent to their duplicates, so break ties on the size itself.
    sizes.sort_by_key(|&(w, h)| std::cmp::Reverse((w as u64 * h as u64, w, h)));
    sizes.dedup();
    sizes
        .into_iter()
        .filter(|(w, _)| w % 2 == 0)
        .map(|(w, h)| {
            let at_size: Vec<&CameraMode> =
                modes.iter().filter(|m| (m.width, m.height) == (w, h)).collect();
            let rates = if at_size.iter().any(|m| m.rates.is_none()) {
                None
            } else {
                let mut rates: Vec<(u32, u32)> = at_size
                    .iter()
                    .flat_map(|m| m.rates.iter().flatten().copied())
                    .collect();
                rates.sort_by(|&a, &b| rate_value(b).total_cmp(&rate_value(a)));
                rates.dedup_by(|a, b| same_rate(*a, *b));
                Some(rates)
            };
            serialize(yuy2_format(w, h, rates.as_deref()))
        })
        .collect()
}

/// The virtual camera's formats when no camera is selected.
pub fn generic_output_formats() -> Vec<Vec<u8>> {
    GENERIC_SIZES
        .iter()
        .map(|&(w, h)| serialize(yuy2_format(w, h, None)))
        .collect()
}

/// Formats to request from a camera with `modes` to produce `width`x`height`
/// frames at `rate`, most preferred first: modes that list `rate` before
/// those that don't say, raw formats (in [`InputFormat::SUPPORTED`] order)
/// before MJPG.
pub fn capture_formats(
    modes: &[CameraMode],
    width: u32,
    height: u32,
    rate: (u32, u32),
) -> Vec<Vec<u8>> {
    let mut candidates: Vec<&CameraMode> = modes
        .iter()
        .filter(|m| (m.width, m.height) == (width, height))
        .filter(|m| m.rates.is_none() || m.offers(rate))
        .collect();
    let preference = |m: &CameraMode| (!m.offers(rate), m.format.rank());
    candidates.sort_by_key(|m| preference(m));
    candidates.dedup_by(|a, b| a.format == b.format);
    candidates
        .into_iter()
        .map(|m| {
            let mut props = vec![id_prop(FormatProperties::MediaType, MediaType::Video.as_raw())];
            match m.format {
                CaptureFormat::Raw(format) => {
                    props.push(id_prop(FormatProperties::MediaSubtype, MediaSubtype::Raw.as_raw()));
                    props.push(id_prop(FormatProperties::VideoFormat, format.as_raw()));
                }
                CaptureFormat::Mjpg => {
                    props.push(id_prop(FormatProperties::MediaSubtype, MediaSubtype::Mjpg.as_raw()));
                }
            }
            props.push(prop(
                FormatProperties::VideoSize,
                Value::Rectangle(Rectangle { width, height }),
            ));
            if m.rates.is_some() {
                let (num, denom) = rate;
                props.push(prop(
                    FormatProperties::VideoFramerate,
                    Value::Fraction(Fraction { num, denom }),
                ));
            }
            serialize(enum_format(props))
        })
        .collect()
}

pub fn serialize(object: Object) -> Vec<u8> {
    PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(object))
        .expect("pod serialization")
        .0
        .into_inner()
}

// ---- pod building ----

fn yuy2_format(width: u32, height: u32, rates: Option<&[(u32, u32)]>) -> Object {
    let frac = |(num, denom): (u32, u32)| Fraction { num, denom };
    let framerate = match rates {
        Some([only]) => Value::Fraction(frac(*only)),
        Some(rates @ [first, ..]) => Value::Choice(ChoiceValue::Fraction(Choice(
            ChoiceFlags::empty(),
            // SPA lists the default first among the alternatives too.
            ChoiceEnum::Enum {
                default: frac(*first),
                alternatives: rates.iter().copied().map(frac).collect(),
            },
        ))),
        Some([]) | None => Value::Choice(ChoiceValue::Fraction(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Range {
                default: frac((30, 1)),
                min: frac((1, 1)),
                max: frac((120, 1)),
            },
        ))),
    };
    enum_format(vec![
        id_prop(FormatProperties::MediaType, MediaType::Video.as_raw()),
        id_prop(FormatProperties::MediaSubtype, MediaSubtype::Raw.as_raw()),
        id_prop(FormatProperties::VideoFormat, VideoFormat::YUY2.as_raw()),
        prop(FormatProperties::VideoSize, Value::Rectangle(Rectangle { width, height })),
        prop(FormatProperties::VideoFramerate, framerate),
        id_prop(FormatProperties::VideoColorMatrix, spa::sys::SPA_VIDEO_COLOR_MATRIX_BT601),
        id_prop(FormatProperties::VideoColorRange, spa::sys::SPA_VIDEO_COLOR_RANGE_16_235),
    ])
}

fn enum_format(properties: Vec<Property>) -> Object {
    Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties,
    }
}

fn prop(key: FormatProperties, value: Value) -> Property {
    Property { key: key.as_raw(), flags: PropertyFlags::empty(), value }
}

fn id_prop(key: FormatProperties, id: u32) -> Property {
    prop(key, Value::Id(Id(id)))
}

// ---- pod reading ----

/// All values of a choice: the alternatives of an enum, or the single value.
fn choice_values<T>(choice: &ChoiceEnum<T>) -> Option<Vec<T>>
where
    T: Copy + spa::pod::CanonicalFixedSizedPod,
{
    match choice {
        ChoiceEnum::None(v) => Some(vec![*v]),
        ChoiceEnum::Enum { default, alternatives } => {
            Some(if alternatives.is_empty() { vec![*default] } else { alternatives.clone() })
        }
        _ => None,
    }
}

fn ids(value: &Value) -> Vec<u32> {
    match value {
        Value::Id(Id(id)) => vec![*id],
        Value::Choice(ChoiceValue::Id(Choice(_, choice))) => choice_values(choice)
            .unwrap_or_default()
            .into_iter()
            .map(|Id(id)| id)
            .collect(),
        _ => Vec::new(),
    }
}

fn rectangles(value: &Value) -> Vec<(u32, u32)> {
    let list = match value {
        Value::Rectangle(r) => vec![*r],
        Value::Choice(ChoiceValue::Rectangle(Choice(_, choice))) => {
            choice_values(choice).unwrap_or_default()
        }
        _ => Vec::new(),
    };
    list.into_iter().map(|r| (r.width, r.height)).collect()
}

/// Frame rates listed by a framerate value; `None` for ranges or unknown
/// shapes, which we treat as "any rate".
fn fractions(value: &Value) -> Option<Vec<(u32, u32)>> {
    let list = match value {
        Value::Fraction(f) => vec![*f],
        Value::Choice(ChoiceValue::Fraction(Choice(_, choice))) => choice_values(choice)?,
        _ => return None,
    };
    Some(list.into_iter().map(|f| (f.num, f.denom)).collect())
}

fn rate_value((num, denom): (u32, u32)) -> f64 {
    if denom == 0 { 0.0 } else { num as f64 / denom as f64 }
}

fn same_rate(a: (u32, u32), b: (u32, u32)) -> bool {
    a.0 as u64 * b.1 as u64 == b.0 as u64 * a.1 as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mjpg_pod(size: Value, rate: Option<Value>) -> Vec<u8> {
        let mut props = vec![
            id_prop(FormatProperties::MediaType, MediaType::Video.as_raw()),
            id_prop(FormatProperties::MediaSubtype, MediaSubtype::Mjpg.as_raw()),
            prop(FormatProperties::VideoSize, size),
        ];
        if let Some(rate) = rate {
            props.push(prop(FormatProperties::VideoFramerate, rate));
        }
        serialize(enum_format(props))
    }

    fn camera_pod(format: Value, size: Value, rate: Option<Value>) -> Vec<u8> {
        let mut props = vec![
            id_prop(FormatProperties::MediaType, MediaType::Video.as_raw()),
            id_prop(FormatProperties::MediaSubtype, MediaSubtype::Raw.as_raw()),
            prop(FormatProperties::VideoFormat, format),
            prop(FormatProperties::VideoSize, size),
        ];
        if let Some(rate) = rate {
            props.push(prop(FormatProperties::VideoFramerate, rate));
        }
        serialize(enum_format(props))
    }

    fn rect(width: u32, height: u32) -> Value {
        Value::Rectangle(Rectangle { width, height })
    }

    fn fmt(f: VideoFormat) -> Value {
        Value::Id(Id(f.as_raw()))
    }

    fn rate_enum(rates: &[(u32, u32)]) -> Value {
        let frac = |&(num, denom): &(u32, u32)| Fraction { num, denom };
        Value::Choice(ChoiceValue::Fraction(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Enum {
                default: frac(&rates[0]),
                alternatives: rates.iter().map(frac).collect(),
            },
        )))
    }

    fn mode(format: VideoFormat, w: u32, h: u32, rates: Option<&[(u32, u32)]>) -> CameraMode {
        let format = CaptureFormat::Raw(format);
        CameraMode { format, width: w, height: h, rates: rates.map(<[_]>::to_vec) }
    }

    fn mjpg(w: u32, h: u32, rates: &[(u32, u32)]) -> CameraMode {
        CameraMode { format: CaptureFormat::Mjpg, width: w, height: h, rates: Some(rates.to_vec()) }
    }

    #[test]
    fn parses_fixed_and_choice_values() {
        let pods = [
            // libcamera style: no framerate.
            camera_pod(fmt(VideoFormat::BGRx), rect(1920, 1080), None),
            // V4L2 style: enumerated rates.
            camera_pod(fmt(VideoFormat::YUY2), rect(640, 360), Some(rate_enum(&[(60, 1), (30, 1)]))),
            // A choice of formats, one of which we can't unpack.
            camera_pod(
                Value::Choice(ChoiceValue::Id(Choice(
                    ChoiceFlags::empty(),
                    ChoiceEnum::Enum {
                        default: Id(VideoFormat::NV12.as_raw()),
                        alternatives: vec![
                            Id(VideoFormat::NV12.as_raw()),
                            Id(VideoFormat::RGBx.as_raw()),
                        ],
                    },
                ))),
                rect(320, 240),
                Some(Value::Fraction(Fraction { num: 15, denom: 1 })),
            ),
        ];
        assert_eq!(
            camera_modes(&pods),
            [
                mode(VideoFormat::BGRx, 1920, 1080, None),
                mode(VideoFormat::YUY2, 640, 360, Some(&[(60, 1), (30, 1)])),
                mode(VideoFormat::RGBx, 320, 240, Some(&[(15, 1)])),
            ]
        );
    }

    #[test]
    fn output_formats_cover_camera_sizes() {
        let modes = [
            mode(VideoFormat::YUY2, 640, 360, Some(&[(60, 1)])),
            mode(VideoFormat::YUY2, 1920, 1080, Some(&[(5, 1)])),
            mode(VideoFormat::BGRx, 640, 360, Some(&[(30, 1)])),
        ];
        let formats = output_formats(&modes);
        assert_eq!(formats.len(), 2);
        // Round-trip through the parser (YUY2 is a supported input too).
        let parsed = camera_modes(&formats);
        assert_eq!(
            parsed,
            [
                mode(VideoFormat::YUY2, 1920, 1080, Some(&[(5, 1)])),
                mode(VideoFormat::YUY2, 640, 360, Some(&[(60, 1), (30, 1)])),
            ]
        );
    }

    #[test]
    fn output_formats_dedup_equal_area_sizes() {
        let modes: Vec<CameraMode> = [VideoFormat::RGBx, VideoFormat::BGRx]
            .into_iter()
            .flat_map(|f| [mode(f, 960, 640, None), mode(f, 1024, 600, None)])
            .collect();
        assert_eq!(output_formats(&modes).len(), 2);
    }

    #[test]
    fn capture_formats_prefer_listed_rates_then_format_order() {
        let modes = [
            mode(VideoFormat::YUY2, 640, 360, Some(&[(30, 1)])),
            mode(VideoFormat::RGBx, 640, 360, None),
            mode(VideoFormat::BGRx, 640, 360, Some(&[(60, 1)])),
            mode(VideoFormat::BGRx, 1280, 720, Some(&[(30, 1)])),
        ];
        let parsed = camera_modes(&capture_formats(&modes, 640, 360, (30, 1)));
        assert_eq!(
            parsed,
            [
                mode(VideoFormat::YUY2, 640, 360, Some(&[(30, 1)])),
                mode(VideoFormat::RGBx, 640, 360, None),
            ]
        );
    }

    #[test]
    fn parses_mjpg_modes() {
        let pods = [
            mjpg_pod(rect(1920, 1080), Some(rate_enum(&[(60, 1), (30, 1)]))),
            // Some other compressed format we can't decode.
            {
                let props = vec![
                    id_prop(FormatProperties::MediaType, MediaType::Video.as_raw()),
                    id_prop(FormatProperties::MediaSubtype, MediaSubtype::H264.as_raw()),
                    prop(FormatProperties::VideoSize, rect(1920, 1080)),
                ];
                serialize(enum_format(props))
            },
        ];
        assert_eq!(camera_modes(&pods), [mjpg(1920, 1080, &[(60, 1), (30, 1)])]);
    }

    #[test]
    fn capture_formats_use_mjpg_only_when_raw_cannot() {
        let modes = [
            mjpg(1920, 1080, &[(60, 1), (30, 1), (5, 1)]),
            mode(VideoFormat::YUY2, 1920, 1080, Some(&[(5, 1)])),
        ];
        // 30 fps: only MJPG can do it.
        assert_eq!(
            camera_modes(&capture_formats(&modes, 1920, 1080, (30, 1))),
            [mjpg(1920, 1080, &[(30, 1)])]
        );
        // 5 fps: both can, raw first.
        assert_eq!(
            camera_modes(&capture_formats(&modes, 1920, 1080, (5, 1))),
            [mode(VideoFormat::YUY2, 1920, 1080, Some(&[(5, 1)])), mjpg(1920, 1080, &[(5, 1)])]
        );
        // The output offers the union of rates at that size.
        assert_eq!(
            camera_modes(&output_formats(&modes)),
            [mode(VideoFormat::YUY2, 1920, 1080, Some(&[(60, 1), (30, 1), (5, 1)]))]
        );
    }

    #[test]
    fn reads_negotiated_format() {
        let pod = mjpg_pod(rect(1280, 720), Some(Value::Fraction(Fraction { num: 30, denom: 1 })));
        let pod = Pod::from_bytes(&pod).unwrap();
        assert_eq!(negotiated(pod), Some((CaptureFormat::Mjpg, 1280, 720)));
        let pod = camera_pod(fmt(VideoFormat::BGRx), rect(640, 480), None);
        let pod = Pod::from_bytes(&pod).unwrap();
        assert_eq!(negotiated(pod), Some((CaptureFormat::Raw(VideoFormat::BGRx), 640, 480)));
    }
}
