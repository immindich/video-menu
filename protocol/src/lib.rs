//! What the video-menu daemon and its clients share: setting types now, and
//! the D-Bus interface definition.

use std::{fmt, str::FromStr};

/// The matting model the green screen uses.
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
    /// Replace the background with green using the matting model.
    #[default]
    GreenScreen,
    /// Flip upside down; a cheap stand-in that needs no model.
    Flip,
}

impl EffectChoice {
    pub const ALL: [Self; 2] = [Self::GreenScreen, Self::Flip];

    /// The name used in settings and environment variables.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GreenScreen => "green",
            Self::Flip => "flip",
        }
    }
}

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
        assert_eq!("green".parse(), Ok(EffectChoice::GreenScreen));
        let err = "modnet".parse::<ModelChoice>().unwrap_err();
        assert!(err.contains("rvm, selfie"), "{err}");
    }
}
