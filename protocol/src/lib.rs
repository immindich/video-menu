//! What the video-menu daemon and its clients share: the D-Bus names, the
//! setting types, and a client proxy for the daemon's interface.
//!
//! The daemon owns [`BUS_NAME`] on the session bus (only one daemon can run
//! at a time) and serves the [`Relay`] interface at [`OBJECT_PATH`]. From a
//! shell:
//!
//! ```text
//! busctl --user introspect computer.isaiah.VideoMenu /computer/isaiah/VideoMenu/Relay
//! busctl --user set-property computer.isaiah.VideoMenu /computer/isaiah/VideoMenu/Relay \
//!     computer.isaiah.VideoMenu.Relay1 Model s selfie
//! ```

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

/// Well-known name the daemon owns on the session bus.
pub const BUS_NAME: &str = "computer.isaiah.VideoMenu";
/// Object path of the relay.
pub const OBJECT_PATH: &str = "/computer/isaiah/VideoMenu/Relay";
/// The relay's interface. The version suffix changes on incompatible changes.
pub const INTERFACE: &str = "computer.isaiah.VideoMenu.Relay1";

/// How the relay has been performing, averaged over about a second.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, zbus::Type)]
pub struct Stats {
    /// Frames processed per second (0 when nothing is streaming).
    pub fps: f64,
    /// Average time from the camera to the virtual camera, in
    /// milliseconds.
    pub latency_ms: f64,
    /// Average time per pipeline stage (name, milliseconds), in order.
    pub stages: Vec<(String, f64)>,
    /// Where frames are processed, e.g. "GPU (Intel(R) Graphics (PTL))".
    pub backend: String,
}

/// Client side of the relay's D-Bus interface.
///
/// `Model` and `Effect` are the names from [`ModelChoice::as_str`] and
/// [`EffectChoice::as_str`]. `BackgroundColor` is the `color` effect's
/// sRGB colour; `BlurRadius` is the `blur` effect's Gaussian standard
/// deviation as a percentage of the frame height, within
/// [`BLUR_RADIUS_RANGE`]. `Camera` is the PipeWire node name of the camera
/// being relayed, or empty for none; `Sources` lists (node name, has the
/// Camera role) for every video source the daemon could use.
#[zbus::proxy(
    interface = "computer.isaiah.VideoMenu.Relay1",
    default_service = "computer.isaiah.VideoMenu",
    default_path = "/computer/isaiah/VideoMenu/Relay"
)]
pub trait Relay {
    /// Relay from the named source, or from none if `name` is empty.
    fn set_camera(&self, name: &str) -> zbus::Result<()>;

    #[zbus(property)]
    fn camera(&self) -> zbus::Result<String>;

    #[zbus(property)]
    fn sources(&self) -> zbus::Result<Vec<(String, bool)>>;

    #[zbus(property)]
    fn model(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn set_model(&self, value: &str) -> zbus::Result<()>;

    #[zbus(property)]
    fn effect(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn set_effect(&self, value: &str) -> zbus::Result<()>;

    #[zbus(property)]
    fn background_color(&self) -> zbus::Result<(u8, u8, u8)>;
    #[zbus(property)]
    fn set_background_color(&self, value: (u8, u8, u8)) -> zbus::Result<()>;

    #[zbus(property)]
    fn blur_radius(&self) -> zbus::Result<f64>;
    #[zbus(property)]
    fn set_blur_radius(&self, value: f64) -> zbus::Result<()>;

    #[zbus(property)]
    fn reuse_matte(&self) -> zbus::Result<bool>;
    #[zbus(property)]
    fn set_reuse_matte(&self, value: bool) -> zbus::Result<()>;

    #[zbus(property)]
    fn selfie_smoothing(&self) -> zbus::Result<f64>;
    #[zbus(property)]
    fn set_selfie_smoothing(&self, value: f64) -> zbus::Result<()>;

    #[zbus(property)]
    fn stats(&self) -> zbus::Result<Stats>;
}

/// The matting model the background effects use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelChoice {
    /// RobustVideoMatting: best edges and temporal stability (GPL-3.0).
    #[default]
    Rvm,
    /// MediaPipe's selfie segmenter: much faster, more flicker (Apache-2.0).
    Selfie,
}

impl ModelChoice {
    pub const ALL: [Self; 2] = [Self::Rvm, Self::Selfie];

    /// The name used in settings and environment variables.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rvm => "rvm",
            Self::Selfie => "selfie",
        }
    }
}

/// What the relay does to each frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EffectChoice {
    /// Pass the camera's picture through unchanged.
    #[default]
    None,
    /// Blur the background, found with the matting model.
    Blur,
    /// Replace the background with a solid colour, found with the matting
    /// model.
    Color,
}

impl EffectChoice {
    pub const ALL: [Self; 3] = [Self::None, Self::Blur, Self::Color];

    /// The name used in settings and environment variables.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Blur => "blur",
            Self::Color => "color",
        }
    }

    /// Whether the effect runs the matting model.
    pub fn needs_matte(self) -> bool {
        self != Self::None
    }
}

/// Colour the `color` effect fills the background with, by default:
/// chroma-key green.
pub const DEFAULT_BACKGROUND_COLOR: (u8, u8, u8) = (0, 177, 64);

/// How strongly the `blur` effect blurs, by default (see `BlurRadius`).
pub const DEFAULT_BLUR_RADIUS: f64 = 2.0;

/// The range of `BlurRadius` the daemon accepts.
pub const BLUR_RADIUS_RANGE: std::ops::RangeInclusive<f64> = 0.25..=8.0;

macro_rules! string_enum {
    ($ty:ty, $what:literal) => {
        impl FromStr for $ty {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, String> {
                Self::ALL.into_iter().find(|v| v.as_str() == s).ok_or_else(|| {
                    let names: Vec<_> = Self::ALL.iter().map(|v| v.as_str()).collect();
                    format!("{} must be one of {}, not {s:?}", $what, names.join(", "))
                })
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

string_enum!(ModelChoice, "model");
string_enum!(EffectChoice, "effect");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for m in ModelChoice::ALL {
            assert_eq!(m.as_str().parse::<ModelChoice>(), Ok(m));
        }
        for e in EffectChoice::ALL {
            assert_eq!(e.to_string().parse::<EffectChoice>(), Ok(e));
        }
        assert_eq!("rvm".parse(), Ok(ModelChoice::Rvm));
        assert_eq!("blur".parse(), Ok(EffectChoice::Blur));
        assert!(!EffectChoice::default().needs_matte());
        let err = "modnet".parse::<ModelChoice>().unwrap_err();
        assert!(err.contains("rvm, selfie"), "{err}");
    }
}
