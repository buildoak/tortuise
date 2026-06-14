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

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};

#[cfg(feature = "hands")]
use super::protocol::{parse_sidecar_line, SidecarProtocolMessage};
use super::{
    bus::LatestBus,
    config::{HandBackend, HandConfig},
    replay::ReplayHandSource,
    types::{
        CameraPreviewFrame, HandControlState, HandDrainStats, HandInputMessage, HandPoseFrame,
        TrackedHand,
    },
};

#[derive(Debug)]
pub struct HandRuntime {
    config: HandConfig,
    bus: LatestBus<HandInputMessage>,
    preview_bus: LatestBus<CameraPreviewFrame>,
    shutdown: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    done_rx: Option<mpsc::Receiver<()>>,
    child: Option<Arc<Mutex<Child>>>,
}

impl HandRuntime {
    pub fn start(config: HandConfig) -> Self {
        let bus = LatestBus::new();
        let preview_bus = LatestBus::new();
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut runtime = Self {
            config: config.clone(),
            bus: bus.clone(),
            preview_bus: preview_bus.clone(),
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
                        if config.camera_preview {
                            preview_bus.publish(synthetic_preview_frame(Instant::now()));
                        }
                        thread::sleep(frame_interval);
                    }
                    let _ = done_tx.send(());
                }));
            }
            HandBackend::Sidecar => {
                #[cfg(feature = "hands")]
                {
                    runtime.start_sidecar_reader(done_tx, bus, preview_bus, shutdown);
                }
                #[cfg(not(feature = "hands"))]
                {
                    runtime.handle = Some(thread::spawn(move || {
                        bus.publish(HandInputMessage::Error("sidecar_requires_hands"));
                        while !shutdown.load(Ordering::Relaxed) {
                            thread::sleep(Duration::from_millis(10));
                        }
                        let _ = done_tx.send(());
                    }));
                }
            }
            HandBackend::AppleVision => {
                runtime.start_apple_vision_reader(
                    done_tx,
                    bus,
                    preview_bus,
                    shutdown,
                    config.target_fps,
                    config.camera_preview,
                    config.camera_preview_fps,
                );
            }
            HandBackend::Off => unreachable!(),
        }

        runtime
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(child) = self.child.as_ref() {
            if let Ok(mut child) = child.lock() {
                terminate_child(&mut child, false);
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
        if !done {
            if let Some(child) = self.child.as_ref() {
                if let Ok(mut child) = child.lock() {
                    terminate_child(&mut child, true);
                }
            }
        }
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

        if self.config.camera_preview {
            if let Some(preview) = self.preview_bus.take_latest().value {
                state.observe_preview(preview, now);
            } else {
                state.update_preview_age(now);
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
            preview_bus: LatestBus::new(),
            shutdown,
            handle: Some(handle),
            done_rx: Some(done_rx),
            child: None,
        }
    }
}

fn terminate_child(child: &mut Child, force: bool) {
    if force {
        let _ = child.kill();
        return;
    }

    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(child.id().to_string())
            .status();
    }

    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
}

impl HandRuntime {
    #[cfg(feature = "hands")]
    fn start_sidecar_reader(
        &mut self,
        done_tx: mpsc::Sender<()>,
        bus: LatestBus<HandInputMessage>,
        preview_bus: LatestBus<CameraPreviewFrame>,
        shutdown: Arc<AtomicBool>,
    ) {
        let sidecar = match find_hand_sidecar_command(self.config.sidecar_command.as_deref()) {
            Some(command) => command,
            None => {
                bus.publish(HandInputMessage::Error("sidecar_missing"));
                let _ = done_tx.send(());
                return;
            }
        };

        let mut child = match Command::new("/bin/sh")
            .arg("-lc")
            .arg(sidecar)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => {
                bus.publish(HandInputMessage::Error("sidecar_spawn_failed"));
                let _ = done_tx.send(());
                return;
            }
        };

        let Some(stdout) = child.stdout.take() else {
            bus.publish(HandInputMessage::Error("sidecar_stdout_failed"));
            let _ = child.kill();
            let _ = done_tx.send(());
            return;
        };

        let child = Arc::new(Mutex::new(child));
        self.child = Some(Arc::clone(&child));
        self.handle = Some(thread::spawn(move || {
            let reader = BufReader::new(stdout);
            let mut saw_error = false;
            for line in reader.lines() {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(line) = line else {
                    bus.publish(HandInputMessage::Error("sidecar_read_failed"));
                    saw_error = true;
                    break;
                };
                match parse_sidecar_line(&line, Instant::now()) {
                    Ok(SidecarProtocolMessage::Ignored) => {}
                    Ok(SidecarProtocolMessage::Input(HandInputMessage::Error(code))) => {
                        saw_error = true;
                        bus.publish(HandInputMessage::Error(code));
                    }
                    Ok(SidecarProtocolMessage::Input(message)) => {
                        saw_error = false;
                        bus.publish(message);
                    }
                    Ok(SidecarProtocolMessage::Preview(frame)) => {
                        saw_error = false;
                        preview_bus.publish(frame);
                    }
                    Err(err) => {
                        saw_error = true;
                        bus.publish(HandInputMessage::Error(err.code()));
                    }
                }
            }
            if !shutdown.load(Ordering::Relaxed) && !saw_error {
                bus.publish(HandInputMessage::Error("sidecar_exit"));
            }
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
                let _ = child.wait();
            }
            let _ = done_tx.send(());
        }));
    }

    fn start_apple_vision_reader(
        &mut self,
        done_tx: mpsc::Sender<()>,
        bus: LatestBus<HandInputMessage>,
        preview_bus: LatestBus<CameraPreviewFrame>,
        shutdown: Arc<AtomicBool>,
        target_fps: u32,
        camera_preview: bool,
        camera_preview_fps: u32,
    ) {
        let helper = match find_apple_vision_helper() {
            Some(path) => path,
            None => {
                bus.publish(HandInputMessage::Error("apple_vision_helper_missing"));
                return;
            }
        };

        let mut command = Command::new(helper);
        command.arg("--fps").arg(target_fps.to_string());
        if camera_preview {
            command
                .arg("--preview")
                .arg("--preview-width")
                .arg("64")
                .arg("--preview-height")
                .arg("36")
                .arg("--preview-fps")
                .arg(camera_preview_fps.to_string());
        }

        let mut child = match command
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
                let captured_at = Instant::now();
                if let Some(preview) = parse_apple_vision_preview_line(&line, captured_at) {
                    preview_bus.publish(preview);
                    continue;
                }
                match parse_apple_vision_line(&line, captured_at) {
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

#[cfg(feature = "hands")]
fn find_hand_sidecar_command(configured_command: Option<&str>) -> Option<String> {
    if let Some(command) = configured_command {
        if !command.trim().is_empty() {
            return Some(command.to_string());
        }
    }

    for var in ["TORTUISE_HAND_SIDECAR", "TORTUISE_HANDS_SIDECAR"] {
        if let Ok(command) = std::env::var(var) {
            if !command.trim().is_empty() {
                return Some(command);
            }
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("tortuise-hand-sidecar");
            if candidate.exists() {
                return Some(shell_quote_path(&candidate));
            }
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        let candidate = PathBuf::from(home)
            .join(".cargo")
            .join("bin")
            .join("tortuise-hand-sidecar");
        if candidate.exists() {
            return Some(shell_quote_path(&candidate));
        }
    }

    for profile in ["release", "debug"] {
        let candidate = PathBuf::from("target")
            .join(profile)
            .join("tortuise-hand-sidecar");
        if candidate.exists() {
            return Some(shell_quote_path(&candidate));
        }
    }

    None
}

#[cfg(feature = "hands")]
fn shell_quote_path(path: &std::path::Path) -> String {
    let raw = path.to_string_lossy();
    format!("'{}'", raw.replace('\'', "'\\''"))
}

fn synthetic_preview_frame(captured_at: Instant) -> CameraPreviewFrame {
    let width = 64usize;
    let height = 36usize;
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = (x * 255 / width) as u8;
            rgb[idx + 1] = (y * 255 / height) as u8;
            rgb[idx + 2] = 72;
        }
    }
    CameraPreviewFrame {
        sequence: 0,
        captured_at,
        width,
        height,
        rgb,
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
            handedness: None,
            landmarks: None,
        });
    }

    Some(HandInputMessage::Sample(HandPoseFrame {
        sequence,
        captured_at,
        detect_ms,
        hands,
    }))
}

fn parse_apple_vision_preview_line(line: &str, captured_at: Instant) -> Option<CameraPreviewFrame> {
    let mut parts = line.trim().split_whitespace();
    if parts.next()? != "preview" {
        return None;
    }
    let sequence = parts.next()?.parse::<u64>().ok()?;
    let width = parts.next()?.parse::<usize>().ok()?;
    let height = parts.next()?.parse::<usize>().ok()?;
    if width == 0 || height == 0 || width > 512 || height > 512 {
        return None;
    }
    let payload = parts.next()?;
    let rgb = BASE64_STANDARD.decode(payload.as_bytes()).ok()?;
    if rgb.len() != width.saturating_mul(height).saturating_mul(3) {
        return None;
    }
    Some(CameraPreviewFrame {
        sequence,
        captured_at,
        width,
        height,
        rgb,
    })
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
            sidecar_command: None,
            camera_preview: false,
            camera_preview_scale: 0.15,
            camera_preview_fps: 8,
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
    fn apple_vision_preview_parser_accepts_rgb_payload() {
        let now = Instant::now();
        let payload = BASE64_STANDARD.encode([255u8, 0, 0, 0, 255, 0]);
        let frame = parse_apple_vision_preview_line(&format!("preview 7 2 1 {payload}"), now)
            .expect("preview line");
        assert_eq!(frame.sequence, 7);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 1);
        assert_eq!(frame.rgb, vec![255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn stalled_runtime_join_respects_deadline() {
        let mut runtime = HandRuntime::stalled_for_test();
        let started = Instant::now();
        assert!(!runtime.join_with_deadline(Duration::from_millis(20)));
        assert!(started.elapsed() < Duration::from_millis(100));
    }
}
