use crate::AppResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandBackend {
    Off,
    Replay,
    Sidecar,
}

impl HandBackend {
    pub fn parse(raw: &str) -> AppResult<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "" => Ok(Self::Off),
            "replay" => Ok(Self::Replay),
            "sidecar" => Ok(Self::Sidecar),
            _ => Err(
                format!("Invalid --hand-backend '{raw}'. Expected off, replay, or sidecar").into(),
            ),
        }
    }

    #[cfg_attr(not(feature = "hands"), allow(dead_code))]
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Replay => "replay",
            Self::Sidecar => "sidecar",
        }
    }
}

#[derive(Debug, Clone)]
pub struct HandConfig {
    pub enabled: bool,
    pub backend: HandBackend,
    pub debug: bool,
    pub target_fps: u32,
    pub timeout_ms: u64,
    pub sensitivity: f32,
}

impl HandConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            backend: HandBackend::Off,
            debug: false,
            target_fps: 30,
            timeout_ms: 200,
            sensitivity: 1.0,
        }
    }

    pub fn from_parts(
        hands: bool,
        backend_raw: Option<&str>,
        debug: bool,
        target_fps: Option<u32>,
        timeout_ms: Option<u64>,
        sensitivity: Option<f32>,
    ) -> AppResult<Self> {
        let mut backend = backend_raw
            .map(HandBackend::parse)
            .transpose()?
            .unwrap_or(if hands {
                HandBackend::Replay
            } else {
                HandBackend::Off
            });
        let target_fps = target_fps.unwrap_or(30);
        if !(1..=60).contains(&target_fps) {
            return Err("--hand-target-fps must be in 1..=60".into());
        }
        let timeout_ms = timeout_ms.unwrap_or(200);
        if !(10..=5000).contains(&timeout_ms) {
            return Err("--hand-timeout-ms must be in 10..=5000".into());
        }
        let sensitivity = sensitivity.unwrap_or(1.0);
        if !sensitivity.is_finite() || sensitivity <= 0.0 || sensitivity > 20.0 {
            return Err("--hand-sensitivity must be finite and in 0..=20".into());
        }

        let enabled = hands || backend != HandBackend::Off || debug;
        if !enabled {
            backend = HandBackend::Off;
        }

        Ok(Self {
            enabled,
            backend,
            debug,
            target_fps,
            timeout_ms,
            sensitivity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_to_replay_when_hands_enabled() {
        let config = HandConfig::from_parts(true, None, true, None, None, None).unwrap();
        assert!(config.enabled);
        assert_eq!(config.backend, HandBackend::Replay);
        assert_eq!(config.target_fps, 30);
        assert_eq!(config.timeout_ms, 200);
    }

    #[test]
    fn config_rejects_invalid_ranges() {
        assert!(HandConfig::from_parts(true, None, false, Some(0), None, None).is_err());
        assert!(HandConfig::from_parts(true, None, false, None, Some(5), None).is_err());
        assert!(HandConfig::from_parts(true, None, false, None, None, Some(-1.0)).is_err());
    }
}
