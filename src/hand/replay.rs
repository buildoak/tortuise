use std::time::Instant;

use super::types::{HandPoseFrame, TrackedHand};

#[derive(Debug, Clone)]
pub struct ReplayHandSource {
    sequence: u64,
}

impl ReplayHandSource {
    pub fn new() -> Self {
        Self { sequence: 0 }
    }

    pub fn next_frame(&mut self, now: Instant) -> HandPoseFrame {
        self.sequence += 1;
        let t = self.sequence as f32 * 0.08;
        HandPoseFrame {
            sequence: self.sequence,
            captured_at: now,
            detect_ms: 2.5,
            hands: vec![TrackedHand {
                id: 0,
                x: 0.5 + 0.18 * t.sin(),
                y: 0.5 + 0.12 * (t * 0.7).cos(),
                pinch: 0.82,
                confidence: 0.95,
            }],
        }
    }
}

impl Default for ReplayHandSource {
    fn default() -> Self {
        Self::new()
    }
}
