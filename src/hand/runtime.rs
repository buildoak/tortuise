use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use super::{
    bus::LatestBus,
    config::{HandBackend, HandConfig},
    replay::ReplayHandSource,
    types::{HandControlState, HandDrainStats, HandInputMessage, HandPoseFrame, TrackedHand},
};

#[derive(Debug)]
pub struct HandRuntime {
    config: HandConfig,
    bus: LatestBus<HandInputMessage>,
    shutdown: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    done_rx: Option<mpsc::Receiver<()>>,
    child: Option<Arc<Mutex<Child>>>,
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
            child: None,
        };

        if !config.enabled || config.backend == HandBackend::Off {
            return runtime;
        }

        let (done_tx, done_rx) = mpsc::channel();
        runtime.done_rx = Some(done_rx);
        match config.backend {
            HandBackend::Replay => {
                runtime.handle = Some(thread::spawn(move || {
                    let mut source = ReplayHandSource::new();
                    let frame_interval =
                        Duration::from_millis((1000 / config.target_fps.max(1) as u64).max(1));
                    while !shutdown.load(Ordering::Relaxed) {
                        bus.publish(HandInputMessage::Sample(source.next_frame(Instant::now())));
                        thread::sleep(frame_interval);
                    }
                    let _ = done_tx.send(());
                }));
            }
            HandBackend::Sidecar => {
                runtime.handle = Some(thread::spawn(move || {
                    bus.publish(HandInputMessage::Error("sidecar_unimplemented"));
                    while !shutdown.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(10));
                    }
                    let _ = done_tx.send(());
                }));
            }
            HandBackend::AppleVision => {
                runtime.start_apple_vision_reader(done_tx, bus, shutdown, config.target_fps);
            }
            HandBackend::Off => unreachable!(),
        }

        runtime
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(child) = self.child.as_ref() {
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
            }
        }
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
            child: None,
        }
    }
}

impl HandRuntime {
    fn start_apple_vision_reader(
        &mut self,
        done_tx: mpsc::Sender<()>,
        bus: LatestBus<HandInputMessage>,
        shutdown: Arc<AtomicBool>,
        target_fps: u32,
    ) {
        let helper = match find_apple_vision_helper() {
            Some(path) => path,
            None => {
                bus.publish(HandInputMessage::Error("apple_vision_helper_missing"));
                return;
            }
        };

        let mut child = match Command::new(helper)
            .arg("--fps")
            .arg(target_fps.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => {
                bus.publish(HandInputMessage::Error("apple_vision_spawn_failed"));
                return;
            }
        };

        let Some(stdout) = child.stdout.take() else {
            bus.publish(HandInputMessage::Error("apple_vision_stdout_failed"));
            let _ = child.kill();
            return;
        };

        let child = Arc::new(Mutex::new(child));
        self.child = Some(Arc::clone(&child));
        self.handle = Some(thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(line) = line else {
                    bus.publish(HandInputMessage::Error("apple_vision_read_failed"));
                    break;
                };
                match parse_apple_vision_line(&line, Instant::now()) {
                    Some(message) => bus.publish(message),
                    None => {}
                }
            }
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
                let _ = child.wait();
            }
            let _ = done_tx.send(());
        }));
    }
}

fn find_apple_vision_helper() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("TORTUISE_APPLE_VISION_HELPER") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("tortuise-apple-vision-helper");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        let candidate = PathBuf::from(home)
            .join(".cargo")
            .join("bin")
            .join("tortuise-apple-vision-helper");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let candidate = PathBuf::from("target")
        .join("release")
        .join("tortuise-apple-vision-helper");
    if candidate.exists() {
        return Some(candidate);
    }

    None
}

fn parse_apple_vision_line(line: &str, captured_at: Instant) -> Option<HandInputMessage> {
    let line = line.trim();
    if line.is_empty() || line.starts_with("status ") {
        return None;
    }
    if let Some(code) = line.strip_prefix("error ") {
        return Some(HandInputMessage::Error(match code.trim() {
            "camera_denied" => "camera_denied",
            "camera_unavailable" => "camera_unavailable",
            "camera_input" => "camera_input",
            "camera_output" => "camera_output",
            "vision_perform" => "vision_perform",
            _ => "apple_vision_error",
        }));
    }

    let mut parts = line.split_whitespace();
    if parts.next()? != "sample" {
        return None;
    }
    let sequence = parts.next()?.parse::<u64>().ok()?;
    let detect_ms = parts.next()?.parse::<f32>().ok()?;
    let mut hands = Vec::new();
    for raw in parts {
        let fields = raw.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 5 {
            continue;
        }
        let id = fields[0].parse::<u8>().ok()?;
        let x = fields[1].parse::<f32>().ok()?;
        let y = fields[2].parse::<f32>().ok()?;
        let pinch = fields[3].parse::<f32>().ok()?;
        let confidence = fields[4].parse::<f32>().ok()?;
        hands.push(TrackedHand {
            id,
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            pinch: pinch.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        });
    }

    Some(HandInputMessage::Sample(HandPoseFrame {
        sequence,
        captured_at,
        detect_ms,
        hands,
    }))
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
        assert!(runtime.join_with_deadline(Duration::from_millis(500)));
        assert!(stats.messages <= 1);
        assert_eq!(stats.samples, 1);
        assert!(state.enabled);
        assert_eq!(state.backend, HandBackend::Replay);
    }

    #[test]
    fn apple_vision_line_parser_accepts_samples_and_errors() {
        let now = Instant::now();
        let message =
            parse_apple_vision_line("sample 42 7.5 0,0.25,0.75,0.9,0.8", now).expect("sample line");
        let HandInputMessage::Sample(frame) = message else {
            panic!("expected sample");
        };
        assert_eq!(frame.sequence, 42);
        assert_eq!(frame.hands.len(), 1);
        assert_eq!(frame.hands[0].id, 0);
        assert!((frame.hands[0].x - 0.25).abs() < f32::EPSILON);
        assert!((frame.hands[0].pinch - 0.9).abs() < f32::EPSILON);

        let message = parse_apple_vision_line("error camera_denied", now).expect("error line");
        assert!(matches!(message, HandInputMessage::Error("camera_denied")));
        assert!(parse_apple_vision_line("status apple_vision_ready", now).is_none());
    }

    #[test]
    fn stalled_runtime_join_respects_deadline() {
        let mut runtime = HandRuntime::stalled_for_test();
        let started = Instant::now();
        assert!(!runtime.join_with_deadline(Duration::from_millis(20)));
        assert!(started.elapsed() < Duration::from_millis(100));
    }
}
