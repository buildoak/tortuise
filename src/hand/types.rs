use std::time::Instant;

use super::{config::HandBackend, controller::GestureController, HandConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandLandmark {
    pub x: f32,
    pub y: f32,
    pub z: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct TrackedHand {
    #[allow(dead_code)]
    pub id: u8,
    pub x: f32,
    pub y: f32,
    pub pinch: f32,
    pub confidence: f32,
    #[allow(dead_code)]
    pub handedness: Option<Handedness>,
    #[allow(dead_code)]
    pub landmarks: Option<[HandLandmark; 21]>,
}

#[derive(Debug, Clone)]
pub struct CameraPreviewFrame {
    #[allow(dead_code)]
    pub sequence: u64,
    pub captured_at: Instant,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct HandPoseFrame {
    #[allow(dead_code)]
    pub sequence: u64,
    pub captured_at: Instant,
    pub detect_ms: f32,
    pub hands: Vec<TrackedHand>,
}

impl HandPoseFrame {
    pub fn visible_count(&self) -> usize {
        self.hands
            .iter()
            .filter(|hand| hand.confidence >= 0.25)
            .count()
    }

    pub fn pinched_count(&self) -> usize {
        self.hands
            .iter()
            .filter(|hand| hand.pinch >= 0.72 && hand.confidence >= 0.25)
            .count()
    }
}

#[derive(Debug, Clone)]
pub enum HandInputMessage {
    Sample(HandPoseFrame),
    Error(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandStatus {
    Off,
    Idle,
    Tracking,
    DebugTracking,
    Stale,
    Error(&'static str),
}

impl HandStatus {
    #[cfg_attr(not(feature = "hands"), allow(dead_code))]
    pub fn code(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Idle => "idle",
            Self::Tracking => "track",
            Self::DebugTracking => "debug_track",
            Self::Stale => "stale",
            Self::Error(code) => code,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HandDrainStats {
    pub messages: usize,
    pub samples: usize,
    #[cfg_attr(not(feature = "hands"), allow(dead_code))]
    pub dropped_or_superseded: u64,
    pub oldest_age_ms: f64,
    pub newest_age_ms: f64,
    pub drain_ms: f64,
    pub sample_latency_ms: f64,
    pub detect_ms: f64,
}

impl Default for HandDrainStats {
    fn default() -> Self {
        Self {
            messages: 0,
            samples: 0,
            dropped_or_superseded: 0,
            oldest_age_ms: 0.0,
            newest_age_ms: 0.0,
            drain_ms: 0.0,
            sample_latency_ms: 0.0,
            detect_ms: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct HandControlState {
    pub enabled: bool,
    pub backend: HandBackend,
    pub debug: bool,
    #[cfg_attr(not(feature = "hands"), allow(dead_code))]
    pub target_fps: u32,
    #[allow(dead_code)]
    pub timeout_ms: u64,
    pub status: HandStatus,
    pub hands_visible: usize,
    pub pinched_hands: usize,
    pub engaged: bool,
    pub applied_this_frame: bool,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub pan_x_delta: f32,
    pub pan_y_delta: f32,
    pub zoom_delta: f32,
    pub roll_delta: f32,
    pub control_age_ms: f64,
    pub detect_ewma_ms: f64,
    pub last_drain: HandDrainStats,
    pub controller: GestureController,
    pub camera_preview_enabled: bool,
    pub camera_preview_scale: f32,
    pub camera_preview_age_ms: f64,
    pub latest_preview: Option<CameraPreviewFrame>,
    pub latest_hands: Vec<TrackedHand>,
}

impl HandControlState {
    pub fn disabled() -> Self {
        Self::new(HandConfig::disabled())
    }

    pub fn new(config: HandConfig) -> Self {
        let status = if config.enabled {
            HandStatus::Idle
        } else {
            HandStatus::Off
        };
        Self {
            enabled: config.enabled,
            backend: config.backend,
            debug: config.debug,
            target_fps: config.target_fps,
            timeout_ms: config.timeout_ms,
            status,
            hands_visible: 0,
            pinched_hands: 0,
            engaged: false,
            applied_this_frame: false,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
            pan_x_delta: 0.0,
            pan_y_delta: 0.0,
            zoom_delta: 0.0,
            roll_delta: 0.0,
            control_age_ms: 0.0,
            detect_ewma_ms: 0.0,
            last_drain: HandDrainStats::default(),
            controller: GestureController::new(
                std::time::Duration::from_millis(config.timeout_ms),
                config.sensitivity,
            ),
            camera_preview_enabled: config.camera_preview,
            camera_preview_scale: config.camera_preview_scale,
            camera_preview_age_ms: 0.0,
            latest_preview: None,
            latest_hands: Vec::new(),
        }
    }

    pub fn reset_tracking(&mut self, status: HandStatus) {
        self.status = status;
        self.hands_visible = 0;
        self.pinched_hands = 0;
        self.engaged = false;
        self.applied_this_frame = false;
        self.yaw_delta = 0.0;
        self.pitch_delta = 0.0;
        self.pan_x_delta = 0.0;
        self.pan_y_delta = 0.0;
        self.zoom_delta = 0.0;
        self.roll_delta = 0.0;
        self.control_age_ms = 0.0;
        self.latest_hands.clear();
        self.controller.reset();
    }

    pub fn toggle_enabled(&mut self) {
        if self.enabled {
            self.enabled = false;
            self.reset_tracking(HandStatus::Off);
        } else {
            self.enabled = true;
            self.status = HandStatus::Idle;
        }
    }

    pub fn observe(&mut self, frame: &HandPoseFrame, now: Instant, mut stats: HandDrainStats) {
        let output = self.controller.observe(frame, now);
        self.hands_visible = frame.visible_count();
        self.pinched_hands = frame.pinched_count();
        self.latest_hands = frame.hands.clone();
        self.engaged = output.engaged;
        self.applied_this_frame = false;
        self.yaw_delta = output.yaw_delta;
        self.pitch_delta = output.pitch_delta;
        self.pan_x_delta = output.pan_x_delta;
        self.pan_y_delta = output.pan_y_delta;
        self.zoom_delta = output.zoom_delta;
        self.roll_delta = output.roll_delta;
        self.control_age_ms = output.age_ms;
        stats.sample_latency_ms = output.age_ms;
        stats.detect_ms = frame.detect_ms as f64;
        self.detect_ewma_ms = if self.detect_ewma_ms <= 0.0 {
            frame.detect_ms as f64
        } else {
            0.85 * self.detect_ewma_ms + 0.15 * frame.detect_ms as f64
        };
        self.status = if output.stale {
            HandStatus::Stale
        } else if self.debug {
            HandStatus::DebugTracking
        } else {
            HandStatus::Tracking
        };
        self.last_drain = stats;
    }

    pub fn observe_preview(&mut self, frame: CameraPreviewFrame, now: Instant) {
        self.camera_preview_age_ms = now
            .saturating_duration_since(frame.captured_at)
            .as_secs_f64()
            * 1000.0;
        self.latest_preview = Some(frame);
    }

    pub fn update_preview_age(&mut self, now: Instant) {
        if let Some(frame) = self.latest_preview.as_ref() {
            self.camera_preview_age_ms = now
                .saturating_duration_since(frame.captured_at)
                .as_secs_f64()
                * 1000.0;
        }
    }

    pub fn set_error(&mut self, code: &'static str, stats: HandDrainStats) {
        self.reset_tracking(HandStatus::Error(code));
        self.last_drain = stats;
    }
}

impl Default for HandControlState {
    fn default() -> Self {
        Self::disabled()
    }
}
