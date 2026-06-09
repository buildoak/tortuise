use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use super::{
    bus::LatestBus,
    config::{HandBackend, HandConfig},
    replay::ReplayHandSource,
    types::{HandControlState, HandDrainStats, HandInputMessage},
};

#[derive(Debug)]
pub struct HandRuntime {
    config: HandConfig,
    bus: LatestBus<HandInputMessage>,
    shutdown: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    done_rx: Option<mpsc::Receiver<()>>,
}

impl HandRuntime {
    pub fn start(config: HandConfig) -> Self {
        let bus = LatestBus::new();
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut runtime = Self {
            config: config.clone(),
            bus: bus.clone(),
            shutdown: Arc::clone(&shutdown),
            handle: None,
            done_rx: None,
        };

        if !config.enabled || config.backend == HandBackend::Off {
            return runtime;
        }

        let (done_tx, done_rx) = mpsc::channel();
        runtime.done_rx = Some(done_rx);
        runtime.handle = Some(match config.backend {
            HandBackend::Replay => thread::spawn(move || {
                let mut source = ReplayHandSource::new();
                let frame_interval =
                    Duration::from_millis((1000 / config.target_fps.max(1) as u64).max(1));
                while !shutdown.load(Ordering::Relaxed) {
                    bus.publish(HandInputMessage::Sample(source.next_frame(Instant::now())));
                    thread::sleep(frame_interval);
                }
                let _ = done_tx.send(());
            }),
            HandBackend::Sidecar => thread::spawn(move || {
                bus.publish(HandInputMessage::Error("sidecar_unimplemented"));
                while !shutdown.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(10));
                }
                let _ = done_tx.send(());
            }),
            HandBackend::Off => unreachable!(),
        });

        runtime
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    pub fn join_with_deadline(&mut self, deadline: Duration) -> bool {
        self.request_shutdown();
        let done = self
            .done_rx
            .as_ref()
            .map(|rx| rx.recv_timeout(deadline).is_ok())
            .unwrap_or(true);
        if done {
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
        done
    }

    pub fn drain_into(&mut self, state: &mut HandControlState, now: Instant) -> HandDrainStats {
        let started = Instant::now();
        let drain = self.bus.take_latest();
        let mut stats = HandDrainStats {
            messages: drain.messages,
            dropped_or_superseded: drain.dropped_or_superseded,
            ..HandDrainStats::default()
        };

        match drain.value {
            Some(HandInputMessage::Sample(frame)) => {
                let age_ms = now
                    .saturating_duration_since(frame.captured_at)
                    .as_secs_f64()
                    * 1000.0;
                stats.samples = 1;
                stats.oldest_age_ms = age_ms;
                stats.newest_age_ms = age_ms;
                stats.drain_ms = started.elapsed().as_secs_f64() * 1000.0;
                state.observe(&frame, now, stats.clone());
                stats = state.last_drain.clone();
            }
            Some(HandInputMessage::Error(code)) => {
                stats.drain_ms = started.elapsed().as_secs_f64() * 1000.0;
                state.set_error(code, stats.clone());
            }
            None => {
                stats.drain_ms = started.elapsed().as_secs_f64() * 1000.0;
                state.last_drain = stats.clone();
                if self.config.enabled && state.status == super::types::HandStatus::Off {
                    state.status = super::types::HandStatus::Idle;
                }
            }
        }

        stats
    }

    #[cfg(test)]
    pub fn stalled_for_test() -> Self {
        let bus = LatestBus::new();
        let shutdown = Arc::new(AtomicBool::new(false));
        let (_done_tx, done_rx) = mpsc::channel();
        let handle = thread::spawn(|| loop {
            thread::sleep(Duration::from_millis(100));
        });
        Self {
            config: HandConfig::disabled(),
            bus,
            shutdown,
            handle: Some(handle),
            done_rx: Some(done_rx),
        }
    }
}

impl Drop for HandRuntime {
    fn drop(&mut self) {
        self.request_shutdown();
        let _ = self.join_with_deadline(Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::{HandBackend, HandConfig};

    #[test]
    fn replay_runtime_drains_latest_sample() {
        let config = HandConfig {
            enabled: true,
            backend: HandBackend::Replay,
            debug: true,
            target_fps: 60,
            timeout_ms: 200,
            sensitivity: 1.0,
        };
        let mut runtime = HandRuntime::start(config.clone());
        thread::sleep(Duration::from_millis(40));
        let mut state = HandControlState::new(config);
        let stats = runtime.drain_into(&mut state, Instant::now());
        runtime.request_shutdown();
        assert!(runtime.join_with_deadline(Duration::from_millis(100)));
        assert!(stats.messages <= 1);
        assert_eq!(stats.samples, 1);
        assert!(state.enabled);
        assert_eq!(state.backend, HandBackend::Replay);
    }

    #[test]
    fn stalled_runtime_join_respects_deadline() {
        let mut runtime = HandRuntime::stalled_for_test();
        let started = Instant::now();
        assert!(!runtime.join_with_deadline(Duration::from_millis(20)));
        assert!(started.elapsed() < Duration::from_millis(100));
    }
}
