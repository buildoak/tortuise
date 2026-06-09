use std::time::{Duration, Instant};

use super::types::HandPoseFrame;

const PINCH_ENTER: f32 = 0.72;
const PINCH_EXIT: f32 = 0.58;
const DEADZONE: f32 = 0.015;
const DELTA_CAP: f32 = 0.12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandControlOutput {
    pub engaged: bool,
    pub stale: bool,
    pub age_ms: f64,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
}

impl HandControlOutput {
    fn idle(age_ms: f64, stale: bool) -> Self {
        Self {
            engaged: false,
            stale,
            age_ms,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GestureController {
    timeout: Duration,
    sensitivity: f32,
    engaged: bool,
    last_position: Option<(f32, f32)>,
}

impl GestureController {
    pub fn new(timeout: Duration, sensitivity: f32) -> Self {
        Self {
            timeout,
            sensitivity,
            engaged: false,
            last_position: None,
        }
    }

    pub fn reset(&mut self) {
        self.engaged = false;
        self.last_position = None;
    }

    pub fn observe(&mut self, frame: &HandPoseFrame, now: Instant) -> HandControlOutput {
        let age = now.saturating_duration_since(frame.captured_at);
        let age_ms = age.as_secs_f64() * 1000.0;
        if age > self.timeout {
            self.reset();
            return HandControlOutput::idle(age_ms, true);
        }

        let Some(hand) = frame
            .hands
            .iter()
            .filter(|hand| hand.confidence >= 0.25)
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
        else {
            self.reset();
            return HandControlOutput::idle(age_ms, false);
        };

        if self.engaged {
            if hand.pinch <= PINCH_EXIT {
                self.reset();
                return HandControlOutput::idle(age_ms, false);
            }
        } else if hand.pinch >= PINCH_ENTER {
            self.engaged = true;
            self.last_position = Some((hand.x, hand.y));
            return HandControlOutput {
                engaged: true,
                stale: false,
                age_ms,
                yaw_delta: 0.0,
                pitch_delta: 0.0,
            };
        } else {
            self.last_position = Some((hand.x, hand.y));
            return HandControlOutput::idle(age_ms, false);
        }

        let (last_x, last_y) = self.last_position.unwrap_or((hand.x, hand.y));
        self.last_position = Some((hand.x, hand.y));
        let yaw_delta = shape_delta((hand.x - last_x) * self.sensitivity);
        let pitch_delta = shape_delta((hand.y - last_y) * self.sensitivity);

        HandControlOutput {
            engaged: true,
            stale: false,
            age_ms,
            yaw_delta,
            pitch_delta,
        }
    }
}

fn shape_delta(delta: f32) -> f32 {
    if delta.abs() < DEADZONE {
        0.0
    } else {
        delta.clamp(-DELTA_CAP, DELTA_CAP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::types::{HandPoseFrame, TrackedHand};

    fn frame(at: Instant, pinch: f32, x: f32, y: f32) -> HandPoseFrame {
        HandPoseFrame {
            sequence: 1,
            captured_at: at,
            detect_ms: 3.0,
            hands: vec![TrackedHand {
                id: 0,
                x,
                y,
                pinch,
                confidence: 0.95,
            }],
        }
    }

    #[test]
    fn pinch_hysteresis_does_not_flap_between_thresholds() {
        let now = Instant::now();
        let mut controller = GestureController::new(Duration::from_millis(200), 1.0);

        assert!(controller.observe(&frame(now, 0.73, 0.5, 0.5), now).engaged);
        assert!(
            controller
                .observe(&frame(now, 0.62, 0.52, 0.5), now)
                .engaged
        );
        assert!(
            !controller
                .observe(&frame(now, 0.57, 0.53, 0.5), now)
                .engaged
        );
    }

    #[test]
    fn stale_sample_resets_continuity() {
        let now = Instant::now();
        let mut controller = GestureController::new(Duration::from_millis(50), 1.0);

        assert!(controller.observe(&frame(now, 0.9, 0.5, 0.5), now).engaged);
        let output =
            controller.observe(&frame(now - Duration::from_millis(80), 0.9, 0.8, 0.5), now);
        assert!(output.stale);
        assert!(!output.engaged);

        let fresh = controller.observe(&frame(now, 0.9, 0.8, 0.5), now);
        assert!(fresh.engaged);
        assert_eq!(fresh.yaw_delta, 0.0);
    }

    #[test]
    fn deadzone_and_delta_cap_bound_motion() {
        let now = Instant::now();
        let mut controller = GestureController::new(Duration::from_millis(200), 4.0);

        controller.observe(&frame(now, 0.9, 0.5, 0.5), now);
        let tiny = controller.observe(&frame(now, 0.9, 0.503, 0.5), now);
        assert_eq!(tiny.yaw_delta, 0.0);

        let large = controller.observe(&frame(now, 0.9, 0.9, 0.1), now);
        assert_eq!(large.yaw_delta, DELTA_CAP);
        assert_eq!(large.pitch_delta, -DELTA_CAP);
    }
}
