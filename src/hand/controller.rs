use std::time::{Duration, Instant};

use super::types::{HandPoseFrame, TrackedHand};

const PINCH_ENTER: f32 = 0.72;
const PINCH_EXIT: f32 = 0.58;
const DEADZONE: f32 = 0.015;
const DELTA_CAP: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandControlOutput {
    pub engaged: bool,
    pub stale: bool,
    pub age_ms: f64,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub pan_x_delta: f32,
    pub pan_y_delta: f32,
    pub zoom_delta: f32,
    pub roll_delta: f32,
}

impl HandControlOutput {
    fn idle(age_ms: f64, stale: bool) -> Self {
        Self {
            engaged: false,
            stale,
            age_ms,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
            pan_x_delta: 0.0,
            pan_y_delta: 0.0,
            zoom_delta: 0.0,
            roll_delta: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GestureController {
    timeout: Duration,
    sensitivity: f32,
    engaged: bool,
    last_position: Option<(f32, f32)>,
    two_hand_engaged: bool,
    last_two_center: Option<(f32, f32)>,
    last_two_distance: Option<f32>,
    last_two_angle: Option<f32>,
}

impl GestureController {
    pub fn new(timeout: Duration, sensitivity: f32) -> Self {
        Self {
            timeout,
            sensitivity,
            engaged: false,
            last_position: None,
            two_hand_engaged: false,
            last_two_center: None,
            last_two_distance: None,
            last_two_angle: None,
        }
    }

    pub fn reset(&mut self) {
        self.engaged = false;
        self.last_position = None;
        self.two_hand_engaged = false;
        self.last_two_center = None;
        self.last_two_distance = None;
        self.last_two_angle = None;
    }

    pub fn observe(&mut self, frame: &HandPoseFrame, now: Instant) -> HandControlOutput {
        let age = now.saturating_duration_since(frame.captured_at);
        let age_ms = age.as_secs_f64() * 1000.0;
        if age > self.timeout {
            self.reset();
            return HandControlOutput::idle(age_ms, true);
        }

        let mut pinched = frame
            .hands
            .iter()
            .filter(|hand| hand.confidence >= 0.25)
            .filter(|hand| {
                if self.two_hand_engaged {
                    hand.pinch >= PINCH_EXIT
                } else {
                    hand.pinch >= PINCH_ENTER
                }
            })
            .collect::<Vec<_>>();
        pinched.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        if pinched.len() >= 2 {
            self.engaged = false;
            self.last_position = None;
            return self.observe_two_hand(pinched[0], pinched[1], age_ms);
        }
        self.two_hand_engaged = false;
        self.last_two_center = None;
        self.last_two_distance = None;
        self.last_two_angle = None;

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
                pan_x_delta: 0.0,
                pan_y_delta: 0.0,
                zoom_delta: 0.0,
                roll_delta: 0.0,
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
            pan_x_delta: 0.0,
            pan_y_delta: 0.0,
            zoom_delta: 0.0,
            roll_delta: 0.0,
        }
    }

    fn observe_two_hand(
        &mut self,
        first: &TrackedHand,
        second: &TrackedHand,
        age_ms: f64,
    ) -> HandControlOutput {
        let center = ((first.x + second.x) * 0.5, (first.y + second.y) * 0.5);
        let dx = second.x - first.x;
        let dy = second.y - first.y;
        let distance = (dx * dx + dy * dy).sqrt();
        let angle = dy.atan2(dx);

        if !self.two_hand_engaged {
            self.two_hand_engaged = true;
            self.last_two_center = Some(center);
            self.last_two_distance = Some(distance);
            self.last_two_angle = Some(angle);
            return HandControlOutput {
                engaged: true,
                stale: false,
                age_ms,
                yaw_delta: 0.0,
                pitch_delta: 0.0,
                pan_x_delta: 0.0,
                pan_y_delta: 0.0,
                zoom_delta: 0.0,
                roll_delta: 0.0,
            };
        }

        let last_center = self.last_two_center.unwrap_or(center);
        let last_distance = self.last_two_distance.unwrap_or(distance);
        let last_angle = self.last_two_angle.unwrap_or(angle);
        self.last_two_center = Some(center);
        self.last_two_distance = Some(distance);
        self.last_two_angle = Some(angle);

        HandControlOutput {
            engaged: true,
            stale: false,
            age_ms,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
            pan_x_delta: shape_delta((center.0 - last_center.0) * self.sensitivity),
            pan_y_delta: shape_delta((center.1 - last_center.1) * self.sensitivity),
            zoom_delta: shape_delta((distance - last_distance) * self.sensitivity),
            roll_delta: shape_delta(wrap_angle_delta(angle - last_angle) * self.sensitivity),
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

fn wrap_angle_delta(delta: f32) -> f32 {
    let wrapped =
        (delta + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    wrapped / std::f32::consts::PI
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
                handedness: None,
                landmarks: None,
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

    #[test]
    fn two_pinched_hands_emit_zoom_and_pan_without_yaw() {
        let now = Instant::now();
        let mut controller = GestureController::new(Duration::from_millis(200), 2.0);
        let make = |left_x, right_x, y| HandPoseFrame {
            sequence: 1,
            captured_at: now,
            detect_ms: 3.0,
            hands: vec![
                TrackedHand {
                    id: 0,
                    x: left_x,
                    y,
                    pinch: 0.9,
                    confidence: 0.95,
                    handedness: None,
                    landmarks: None,
                },
                TrackedHand {
                    id: 1,
                    x: right_x,
                    y,
                    pinch: 0.9,
                    confidence: 0.95,
                    handedness: None,
                    landmarks: None,
                },
            ],
        };

        controller.observe(&make(0.35, 0.65, 0.5), now);
        let output = controller.observe(&make(0.30, 0.75, 0.55), now);
        assert!(output.engaged);
        assert_eq!(output.yaw_delta, 0.0);
        assert!(output.zoom_delta > 0.0);
        assert!(output.pan_x_delta > 0.0);
        assert!(output.pan_y_delta > 0.0);
    }
}
