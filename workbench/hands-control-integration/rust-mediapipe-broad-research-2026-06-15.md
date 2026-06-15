# Rust x MediaPipe Broad Research

Date: 2026-06-15
Branch: `feat/hands-replay-core`

## Questions

1. Is there a plug-and-play Rust MediaPipe integration for native macOS/tortuise?
2. If not, is a standalone Rust x MediaPipe repo worth publishing?
3. What is the best creation path if we use Python/C++ MediaPipe as references and converge with agentic implementation loops?

## Executive Answer

No existing Rust option is genuinely plug-and-play for tortuise's needs:

- native macOS terminal app,
- live webcam,
- official MediaPipe Hand Landmarker/Tasks behavior,
- 21 landmarks,
- handedness/world landmarks,
- stable package/install story,
- usable through normal Rust APIs.

There are two realistic paths:

1. **Rust wrapper over official MediaPipe Tasks C API**
   - Most likely to become publishable and reliable quickly.
   - Uses official hand landmarker implementation.
   - Rust owns safe API, camera input, gesture layer, examples, doctor CLI, and tortuise adapter.
   - Hard part is building/packaging the native MediaPipe runtime.

2. **Full Rust reimplementation**
   - Most in the spirit of tortuise.
   - Uses Rust camera capture + ONNX/TFLite runtime + Rust implementation of MediaPipe graph glue.
   - Hard part is not inference; it is palm anchors/NMS, rotated ROI crop, landmark projection, and tracking loop.
   - Worth a bounded playground, not yet safe to assume as production path.

Recommendation:

Build a separate repo, tentatively `mediapipe-hand-rs`, with two engines behind one safe API:

- `Engine::MediaPipeC` first for correctness/reference.
- `Engine::RustOrt` as the pure-Rust playground.

Then tortuise can consume the same `HandFrame` / `GestureFrame` API regardless of engine.

## Plug-And-Play Rust Candidate Audit

| Rank | Candidate | Verdict | Why |
|---:|---|---|---|
| 1 | Official MediaPipe Tasks C API + custom Rust FFI | Best production base, not plug-and-play | Official hand landmarker exists as C API, but we must build/ship dylib and bindings. |
| 2 | `WasmEdge/mediapipe-rs` | Not suitable for tortuise native path | WasmEdge/WASI-NN oriented, not normal native macOS terminal dependency. |
| 3 | `ux-mediapipe` / old Rust MediaPipe graph wrappers | Not suitable | Stale, legacy graph API, docs build problems, custom Bazel dylib/OpenCV setup. |
| 4 | `handtrack-rs` | Not suitable | Bounding boxes only, no 21 landmarks / Tasks. |
| 5 | `rusted_pipe` | Not relevant | Generic graph/pipeline framework, no MediaPipe hand tracking. |

Conclusion: there is no `cargo add mediapipe-hands` answer today.

## Official MediaPipe Behavior We Must Match

The Hand Landmarker task supports:

- `IMAGE`: synchronous image calls, no tracking reuse.
- `VIDEO`: synchronous decoded video calls, monotonic `timestamp_ms`, tracking reuse.
- `LIVE_STREAM`: async callback mode, monotonic timestamps, may drop frames.

Outputs:

- handedness,
- 21 normalized image landmarks,
- 21 world landmarks,
- empty arrays when no hands are found.

The `.task` model bundle contains:

- `hand_detector.tflite`
- `hand_landmarks_detector.tflite`

The internal pipeline is:

```text
frame
  -> palm detector when needed
  -> SSD anchor decode + NMS
  -> rotated hand rect
  -> landmark model on cropped hand ROI
  -> landmark/presence/handedness/world outputs
  -> project landmarks back to original image
  -> compute next-frame rect
  -> skip palm detector while tracking stays confident
```

That detector-skipping tracking loop is why MediaPipe feels smooth.

## Full Rust Playground Architecture

Smallest viable Rust-native playground:

```text
camera frame
  -> nokhwa AVFoundation capture
  -> ort ONNX Runtime CPU
  -> palm_detection_mediapipe.onnx
  -> Rust anchor decode + NMS
  -> rotated ROI crop
  -> handpose_estimation_mediapipe.onnx
  -> Rust projection + tracking loop
  -> HandFrame JSON/debug overlay
```

Suggested crates:

- Camera: `nokhwa` first; `opencv` fallback.
- Inference: `ort` first.
- Later acceleration: ONNX Runtime CoreML EP or XNNPACK only after correctness.
- Possible alternates: `tract` / `tract-tflite`, `tflitec`, `tflite`, `candle-onnx`.

Why `ort` first:

- mature ONNX Runtime wrapper,
- dynamic/runtime setup patterns,
- execution providers exist,
- easier to load OpenCV/Hugging Face ONNX ports.

Biggest risks:

- palm detector postprocess,
- anchor constants,
- letterbox removal,
- rotated rect math,
- crop/rotate sampling,
- projecting landmark/world coordinates back,
- maintaining stable two-hand tracking.

## Publishable Repo Proposal

Repo: `mediapipe-hand-rs`

Workspace:

```text
mediapipe-hand-rs/
  crates/
    mediapipe-tasks-sys/     # unsafe FFI to official Tasks C API
    mediapipe-hand/          # safe Rust HandLandmarker API
    mediapipe-hand-gesture/  # pinch/open/fist/smoothing/control primitives
    mediapipe-hand-cli/      # doctor, fetch-model, webcam demo, image demo
    mediapipe-hand-rustort/  # optional pure Rust/ONNX playground engine
  examples/
    image.rs
    webcam.rs
    gesture_mouse.rs
    tortuise_camera.rs
  docs/
    install.md
    runtime-binaries.md
    coordinate-systems.md
    gestures.md
    tortuise.md
    licensing.md
```

Core API shape:

```rust
let mut tracker = HandLandmarker::builder()
    .model(ModelSource::cache_or_download(Model::HandLandmarkerFull)?)
    .running_mode(RunningMode::Video)
    .num_hands(2)
    .min_detection_confidence(0.5)
    .min_presence_confidence(0.5)
    .min_tracking_confidence(0.5)
    .build()?;

let frame = ImageFrame::srgb(width, height, &rgb_bytes)?;
let result = tracker.detect_video(frame, timestamp_ms)?;

for hand in result.hands() {
    let pinch = GestureAnalyzer::default().pinch(&hand);
}
```

Public types:

- `HandLandmarker`
- `RunningMode::{Image, Video, LiveStream}`
- `HandLandmarkerOptions`
- `HandResult`
- `Hand`
- `Handedness`
- `Landmark`
- `WorldLandmark`
- `HandLandmark` enum with all 21 joints
- `HAND_CONNECTIONS`
- `ImageFrame`
- `ImageFormat`
- `TimestampMs`
- `GestureAnalyzer`
- `GestureFrame`

## Packaging Law

Do not make users build all of MediaPipe from source during `cargo build`.

Preferred production packaging:

- small Rust crates on crates.io,
- `mediapipe-hand-cli doctor`,
- `mediapipe-hand-cli fetch-runtime`,
- runtime dylib/model cache with checksums,
- `MEDIAPIPE_LIBRARY_PATH` / `MEDIAPIPE_MODEL_PATH` overrides,
- GitHub Releases for macOS arm64 runtime bundles.

License:

- Rust repo: `MIT OR Apache-2.0` or Apache-2.0.
- MediaPipe/model: preserve Apache-2.0 notices.
- Do not vendor large model/runtime blobs into crates.

## Agentic Build Plan

### Phase 0 - Reference Fixtures

- Save frames from current Python sidecar.
- Save JSON hand results from Python MediaPipe.
- Save overlay images.
- These become golden fixtures for Rust/C parity.

### Phase 1 - C API Wrapper Spike

- Build/load official MediaPipe Tasks C API on macOS arm64.
- Wrap:
  - create,
  - detect image,
  - detect for video,
  - close result,
  - close landmarker.
- Feed one RGB frame.
- Compare JSON output to Python sidecar.

Gate:

- `cargo run --example image` produces matching landmarks on a saved fixture.

### Phase 2 - Webcam / Tortoise Adapter

- Rust camera capture via `nokhwa` or OpenCV fallback.
- Emit the exact same JSONL protocol our current Python helper emits.
- Plug into tortuise through existing sidecar backend.

Gate:

- 5-minute MacBook Neo run, no leaked process, no camera lock after quit.

### Phase 3 - Pure Rust ORT Playground

- Load ONNX palm detector.
- Implement anchors/NMS visually.
- Add hand landmark model on cropped ROI.
- Add projection back.
- Add tracker loop.

Gate:

- Compare against Python/MediaPipe fixtures.
- If too far after bounded work, keep it as research and ship C API wrapper.

### Phase 4 - Gesture Polish

- Move gestures into `mediapipe-hand-gesture`.
- Port all web gestures:
  - one-hand orbit,
  - two-hand zoom/pan,
  - dive mode,
  - reset/no-jump,
  - smoothing,
  - sensitivity profiles.

Gate:

- tortuise gestures match the web prototype behavior on bee and apartment splats.

## Evidence Links

- MediaPipe Hand Landmarker overview: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- MediaPipe Tasks platform docs: https://developers.google.com/edge/mediapipe/solutions/tasks
- MediaPipe Hand Landmarker C API tree: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/tasks/c/vision/hand_landmarker
- MediaPipe C++ getting started: https://developers.google.com/edge/mediapipe/framework/getting_started/cpp
- Hand Landmarker Python API: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker
- Hand Landmarker C++ task wrapper: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.cc
- Hand Landmarker graph: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_graph.cc
- Hand landmarks detector graph: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_detector_graph.cc
- Palm detector graph: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/hand_detector/hand_detector_graph.cc
- MediaPipe Hands paper: https://arxiv.org/abs/2006.10214
- WasmEdge MediaPipe Rust prior art: https://github.com/WasmEdge/mediapipe-rs
- WasmEdge WASI-NN MediaPipe docs: https://wasmedge.org/docs/develop/rust/wasinn/mediapipe/
- ONNX palm detector: https://huggingface.co/opencv/palm_detection_mediapipe
- ONNX handpose model: https://huggingface.co/opencv/handpose_estimation_mediapipe
- PINTO ONNX hand gesture reference: https://github.com/PINTO0309/hand-gesture-recognition-using-onnx
- `ort` docs: https://docs.rs/ort/latest/ort/
- ONNX Runtime CoreML EP: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- `nokhwa`: https://github.com/l1npengtul/nokhwa
- OpenCV Rust VideoCapture: https://docs.rs/opencv/latest/opencv/videoio/struct.VideoCapture.html

## Decision

Start a separate `mediapipe-hand-rs` playground/repo.

First milestone should not be pure Rust. It should be:

> Rust API + official MediaPipe C API runtime + fixture parity.

Second milestone:

> Rust ORT playground attempting the full Rust pipeline.

This gives us a publishable path either way:

- if C API wins: ship ergonomic Rust MediaPipe wrapper,
- if pure Rust catches up: ship a uniquely Rust-native hand tracker,
- if both work: expose both engines behind one API.

