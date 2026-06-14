# Native MediaPipe Port Plan

Date: 2026-06-14
Branch: `feat/hands-replay-core`

## Context

The Python MediaPipe sidecar proved the full signal chain:

camera -> MediaPipe Hand Landmarker -> 21 landmarks -> terminal preview skeleton -> pinch -> tortuise camera motion.

That makes MediaPipe the right recognition model family for tortuise. The next goal is to remove Python from the production path while keeping the current sidecar as a golden reference for behavior, calibration, and regression tests.

## Immediate Lifecycle Fix

The Python sidecar was not graceful enough before this note:

- Rust used `Child::kill()` on shutdown.
- Python only guaranteed `capture.release()` on normal loop exit / KeyboardInterrupt.
- A hard kill could skip cleanup and leave camera/MediaPipe resources to OS teardown.

Patch applied:

- Rust sends SIGTERM first for helper shutdown.
- Rust only force-kills after the join deadline.
- Python handles SIGTERM/SIGINT by setting a stop flag, exiting the capture loop, emitting a stopping status, and releasing the camera.

Verification:

- `python3 -m py_compile helpers/mediapipe_hands_sidecar.py`
- `cargo test --features "metal hands"` -> 100 passed.

## Research Findings

### MediaPipe Capability Surface

Official MediaPipe Hand Landmarker supports still images, decoded video, and live video, and returns handedness, 21 image-space landmarks, and world landmarks. This matches tortuise's needs exactly.

Sources:

- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- https://github.com/google-ai-edge/mediapipe
- https://github.com/google-ai-edge/mediapipe-samples
- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js
- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/ios

### C++ / Native Path

MediaPipe Framework is officially C++ capable, but the newer Tasks Hand Landmarker public docs emphasize Android, Python, Web, and iOS. Search results and GitHub issue history suggest C++ Hand Landmarker Tasks support has been less straightforward than Python/Web/iOS.

This means the C++ path is probably still best, but it should be approached as a native bridge spike, not assumed to be one clean Cargo dependency.

Likely shape:

- Build or vendor a small native C++ hand tracker binary/library.
- Feed frames from a macOS camera capture backend.
- Return the same JSON/protobuf-ish `HandFrame` shape tortuise already uses.
- Once stable, expose it to Rust via `cxx`, `bindgen`, or a minimal C ABI.

Why this is strongest:

- Preserves official MediaPipe behavior.
- Avoids reimplementing palm detection, ROI crop tracking, landmark postprocess, handedness, and temporal tracking.
- Lets us keep Python sidecar as a reference oracle during development.

Primary risk:

- Build complexity: Bazel/MediaPipe native deps, macOS packaging, model/resource loading, and binary size.

### Full Rust Path

There is no obvious mature, official, production-ready Rust equivalent of MediaPipe Tasks Hand Landmarker for native macOS.

Options found:

- `mediapipe-rs` exists, but it is WasmEdge-oriented and not clearly a direct native macOS terminal hand-tracking solution.
- TFLite Rust crates exist, including safe wrappers and XNNPACK-oriented paths.
- ONNX hand pipelines exist, but they are unofficial ports and would make us own more model/pipeline behavior.

Sources:

- https://github.com/WasmEdge/mediapipe-rs
- https://crates.io/crates/tflitec
- https://docs.rs/edgefirst-tflite
- https://github.com/PINTO0309/hand-gesture-recognition-using-onnx/blob/main/README_en.md
- https://blog.tensorflow.org/2020/07/accelerating-tensorflow-lite-xnnpack-integration.html

Full Rust can be done, but it is not just "load one model." MediaPipe Hands is a pipeline: palm detector, ROI transform, landmark model, hand tracking across frames, handedness, postprocess, and fallback detection when tracking is lost.

This is high-control, high-maintenance. It becomes attractive only if:

- C++ bridge is too painful to package,
- or we want a long-term custom Rust perception stack,
- or we can find an ONNX/TFLite pipeline with stable outputs and acceptable accuracy.

### Apple Vision Path

Apple Vision remains a useful fallback and benchmark, but not the preferred polished path. It is easier to integrate on macOS, but current prototype evidence and prior web prototype calibration show MediaPipe gives us the richer behavior we want.

## Recommendation

Use a three-lane strategy:

1. **Reference lane: keep Python sidecar**
   - Treat it as the correctness and gesture-calibration oracle.
   - Keep it opt-in for development.
   - Ensure graceful shutdown and resource telemetry stay clean.

2. **Production lane: C++ native MediaPipe bridge**
   - First as a separate native helper with the same JSONL protocol.
   - Then, if stable, convert to in-process Rust FFI.
   - Preserve the current Rust `HandInputMessage` protocol so gesture code does not care which tracker backend produced the landmarks.

3. **Exploration lane: full Rust TFLite/ONNX**
   - Run a bounded spike only after C++ feasibility is known.
   - Gate: can it match Python MediaPipe landmark quality and latency on MacBook Neo without making us own too much postprocess code?

## Proposed Execution Phases

### Phase A - Sidecar Hygiene And Gesture Parity

Goal: make current Python path reliable enough to be a reference.

Tasks:

- Add a `--hand-backend mediapipe-python` alias or preset command to remove long shell quoting.
- Add explicit camera device listing and stable camera selection.
- Add sidecar process telemetry: child pid, last sample time, shutdown status.
- Port exact web gesture mapping:
  - one-hand orbit direction fix,
  - two-hand zoom/pan,
  - roll decision,
  - dive mode,
  - hand reset/no-jump behavior.

Gate:

- Quit tortuise 20 times; no Python, camera, or helper processes survive.
- Pinch status and orbit direction match the web prototype on bee.

### Phase B - C++ Helper Spike

Goal: prove native MediaPipe outside Python.

Tasks:

- Build a minimal C++ executable:
  - open camera,
  - run hand landmarker,
  - emit same JSONL protocol as Python helper,
  - emit preview frame or omit preview at first.
- Use existing Rust sidecar backend unchanged.
- Compare against Python sidecar telemetry and visual overlay.

Gate:

- Runs on MacBook Neo from tortuise with no Python env.
- Tracks one/two hands for 5 minutes.
- No leaked process after quit.
- Latency equal or better than Python sidecar.

### Phase C - Rust FFI Integration

Goal: remove process boundary if the C++ helper is stable.

Tasks:

- Extract C++ tracker into a library with C ABI or `cxx`.
- Rust owns lifecycle explicitly: init, start, poll latest frame, stop, destroy.
- Keep gesture engine purely Rust.

Gate:

- No sidecar process.
- Rust shutdown releases camera every time.
- Same hand-frame outputs as helper.

### Phase D - Full Rust Spike

Goal: decide whether a pure Rust tracker is worth the cost.

Tasks:

- Try TFLite Rust with XNNPACK.
- Try ONNX Runtime Rust with available hand/palm models.
- Reproduce MediaPipe's palm/ROI/landmark/tracking loop on a narrow fixture set.

Gate:

- If not within striking distance of MediaPipe quality in 2-3 focused sessions, archive the Rust tracker path and stay C++/FFI.

## Current Decision

Do not jump directly to full Rust.

The pragmatic next step is:

1. Make Python sidecar fully clean and reference-grade.
2. Build C++ native helper with the same protocol.
3. Only then decide whether in-process FFI or full Rust is justified.

