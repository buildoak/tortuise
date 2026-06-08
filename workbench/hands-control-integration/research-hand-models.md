# Hand Model Runtime Research For tortuise Hand Control

Date: 2026-06-08

## Gate Decision

**Simplest MVP path:** Apple Vision `VNDetectHumanHandPoseRequest` through `objc2-vision`, with a small macOS-only camera adapter using either `ccap-rs` first or direct `objc2-av-foundation` if `ccap-rs` cannot expose the buffer shape needed without copies.

Why: tortuise is already a Rust terminal app with a macOS-heavy tested path. Vision needs no model download, is offline by default, ships with macOS, and Rust bindings already expose the hand-pose request, result observation, normalized hand joints, sequence request handler, and compute-device APIs. The MVP can map 2D hand landmarks to camera controls without solving model packaging or MediaPipe FFI.

**Robust longer-term path:** introduce a `HandTracker` trait with a native macOS Vision backend as the default and a MediaPipe backend behind an optional feature or sidecar. Keep the app control layer consuming one normalized internal `HandPoseFrame` regardless of runtime. Add MediaPipe only when cross-platform support, world landmarks, or MediaPipe gesture-recognizer compatibility becomes more valuable than keeping install/package friction low.

## Recommendation Matrix

| Option | Fit For tortuise | Output | Expected Latency | Hardware | Offline Packaging | Rust Integration | License / Legal | Main Risks |
|---|---|---|---|---|---|---|---|---|
| Apple Vision `VNDetectHumanHandPoseRequest` | **Best MVP** for macOS/Rust terminal app | 2D normalized hand joints with confidence; chirality available through `VNHumanHandPoseObservation`; no bundled world landmark model | No official per-frame macOS latency found in accessible docs; expected real-time on Apple Silicon, but must benchmark. Use separate thread and throttle to 15-30 Hz | Vision chooses implementation; `objc2-vision` exposes `usesCPUOnly` (deprecated), supported compute-stage devices, and `setComputeDevice` APIs. Do not assume ANE without measurement | No app model file; depends on macOS Vision framework | Good: `objc2-vision` has `VNDetectHumanHandPoseRequest`, `VNHumanHandPoseObservation`, `VNImageRequestHandler`, and `VNSequenceRequestHandler` bindings | Apple SDK/framework terms; Rust crate is `Zlib OR Apache-2.0 OR MIT` | macOS-only; Objective-C FFI sharp edges; camera permission/TCC behavior for terminal-launched CLI; no official model version pin |
| MediaPipe Tasks Hand Landmarker | **Best cross-platform/ML semantics**, not MVP | 21 landmarks in image coordinates, 21 world landmarks, handedness for multiple hands | Official benchmark for full pipeline on Pixel 6: CPU 17.12 ms, GPU 12.27 ms. Not a Mac benchmark | Python BaseOptions supports CPU/GPU, but Google docs say Python GPU support is limited to Ubuntu. Swift delegate has CPU/GPU and defaults to CPU. No ANE path | Bundled `.task` file; official hand landmarker task is 7,819,105 bytes by HEAD request to Google Storage | Weak for pure Rust: no official Rust Tasks API found. Choices are Python sidecar, C++/C ABI FFI, or unofficial crates | MediaPipe repo and samples Apache-2.0; model card says Apache License 2.0 | Packaging Python/OpenCV/mediapipe is heavy; C++/Bazel/FFI is high-friction; GPU path on macOS is unclear; sidecar IPC adds process management |
| OpenCV camera capture + external hand model | **Capture helper only**, not a hand-pose solution | Frames only unless paired with a model/runtime | Capture/conversion overhead only; model latency depends on external runtime | Uses platform video backends; no hand-model acceleration by itself | Requires OpenCV install or linking; heavier than native macOS capture | `opencv` crate exposes `VideoCapture`, but build/linking is historically heavier than pure Rust/native framework wrappers | Rust crate MIT; OpenCV library is separately licensed | Adds a large dependency for something AVFoundation can do; not useful without separate hand model |
| `ccap-rs` camera capture | **Best capture candidate for MVP if frame format works** | Camera frames, not landmarks | Should be low overhead; project claims hardware-accelerated pixel conversion via Apple Accelerate / NEON / AVX2 | Native macOS AVFoundation capture; conversion uses Apple Accelerate on macOS | Cargo crate can build source; no OpenCV install; no model | Good Rust wrapper; simple provider/open/grab API | MIT | Newer ecosystem than OpenCV; need verify camera permissions and pixel format bridge to Vision/CPU image buffers |
| `nokhwa` camera capture | Viable backup capture layer | Camera frames, not landmarks | Unknown; likely fine for 15-30 Hz hand-control input | AVFoundation feature on macOS; also V4L/MSMF for cross-platform capture | Cargo dependency, but feature/backends need testing on macOS | Simple camera abstraction, but docs.rs latest page only built Linux docs; macOS backend exists as optional feature | Apache-2.0 | Maintenance/docs uncertainty; may still have platform permission quirks |
| Direct `objc2-av-foundation` camera capture | Most native long-term macOS capture | CMSampleBuffer / CVPixelBuffer path can feed Vision with fewer conversions | Best potential latency/copy profile if implemented carefully | Native AVFoundation | No third-party camera runtime | Harder than `ccap-rs`; but same Objective-C family as Vision backend | Rust crate follows objc2 licensing | Higher first-implementation complexity; more unsafe FFI and delegate plumbing |

## Evidence Notes

### Apple Vision

- `objc2-vision` 0.3.2 provides bindings to Apple's Vision framework and includes `VNDetectHumanHandPoseRequest`, `VNHumanHandPoseObservation`, and the hand joint names. Source: [objc2-vision crate docs](https://docs.rs/objc2-vision/latest/objc2_vision/).
- `VNDetectHumanHandPoseRequest` detects landmark points on human hands and returns `VNHumanHandPoseObservation` results. The binding exposes `maximumHandCount`; docs state the default is 2 and revision 1 max is 6. Source: [VNDetectHumanHandPoseRequest docs](https://docs.rs/objc2-vision/latest/objc2_vision/struct.VNDetectHumanHandPoseRequest.html).
- `VNHumanHandPoseObservation` exposes `recognizedPointForJointName_error`, `recognizedPointsForJointsGroupName_error`, `chirality`, and an `MLMultiArray` form with `(x, y, confidence)` for recognized points. Source: [VNHumanHandPoseObservation docs](https://docs.rs/objc2-vision/latest/objc2_vision/struct.VNHumanHandPoseObservation.html).
- `VNSequenceRequestHandler` performs requests on image sequences and accepts `CVPixelBuffer` / `CMSampleBuffer`, which is the right shape for live camera frames. Source: [VNSequenceRequestHandler docs](https://docs.rs/objc2-vision/latest/objc2_vision/struct.VNSequenceRequestHandler.html).
- `VNImageRequestHandler` also accepts `CVPixelBuffer`, `CGImage`, and `CIImage`; useful for still-frame tests before live capture. Source: [VNImageRequestHandler docs](https://docs.rs/objc2-vision/latest/objc2_vision/struct.VNImageRequestHandler.html).
- Compute note: `objc2-vision` exposes deprecated `usesCPUOnly`, `supportedComputeStageDevicesAndReturnError`, `computeDeviceForComputeStage`, and `setComputeDevice_forComputeStage`. Treat hardware use as opaque until benchmarked. Source: [VNDetectHumanHandPoseRequest docs](https://docs.rs/objc2-vision/latest/objc2_vision/struct.VNDetectHumanHandPoseRequest.html).

### MediaPipe Tasks / Hand Landmarker

- MediaPipe Hand Landmarker accepts still images, decoded video frames, and live video feed. It outputs handedness, image-coordinate landmarks, and world-coordinate landmarks. Source: [Google Hand Landmarker overview](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- The task uses a bundle containing a palm detection model plus a hand landmarks model. In video/live modes it tracks from prior frames and only re-runs palm detection when presence/tracking fails, reducing latency. Source: [Google Hand Landmarker overview](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- Official model table lists `HandLandmarker (full)` with 192x192 and 224x224 float16 inputs. Official benchmark on Pixel 6: 17.12 ms CPU, 12.27 ms GPU. Source: [Google Hand Landmarker overview](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- Official Python guide uses `model_asset_path`, supports `IMAGE`, `VIDEO`, and `LIVE_STREAM`, and notes `detect_async` returns immediately while dropping frames when the landmarker is busy. Source: [Google Python Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).
- Python `BaseOptions` supports CPU/GPU delegates, but docs state Python GPU support is currently limited to Ubuntu platforms. Source: [Google BaseOptions docs](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions).
- Swift MediaPipe Tasks delegate defaults to CPU if unset and has CPU/GPU cases. Source: [MediaPipe Swift Delegate docs](https://developers.google.com/edge/api/mediapipe/swift/vision/Enums/Delegate).
- Model card says the hand tracking pipeline outputs 21 screen landmarks, 21 metric-scale world landmarks, and handedness; it is licensed under Apache License 2.0. Source: [MediaPipe hand tracking model card PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf).
- Model size checked by HEAD request on 2026-06-08: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task` has `Content-Length: 7819105` bytes, about 7.46 MiB.

### Camera Capture / Rust Bindings

- OpenCV Rust `VideoCapture` can open cameras with a device index and selected backend, and can read frames. Source: [opencv `VideoCapture` docs](https://docs.rs/opencv/latest/opencv/videoio/struct.VideoCapture.html).
- OpenCV macOS installation docs list Homebrew and pip options, which is useful but heavier than a native framework-only capture layer. Source: [OpenCV macOS install docs](https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html).
- `ccap-rs` is a Rust wrapper over ccap; ccap documents macOS AVFoundation support, no third-party dependencies, RGB/BGR/YUV conversion, and Apple Accelerate acceleration. Source: [ccap homepage](https://ccap.work/).
- `nokhwa` provides a simplified camera abstraction and optional macOS AVFoundation backend; `Camera::new`, `open_stream`, and frame methods are exposed. Source: [nokhwa Camera docs](https://docs.rs/nokhwa/latest/nokhwa/struct.Camera.html).

## MVP Architecture Sketch

Keep the first implementation macOS-only and reversible:

```text
camera thread
  -> capture RGB/CVPixelBuffer at 15-30 Hz
  -> Vision hand-pose request
  -> normalize into HandPoseFrame { timestamp, handedness, joints[21], confidence }
  -> gesture/control reducer
  -> existing tortuise camera commands

render thread
  -> remains CPU splat renderer
  -> consumes latest control intent, not raw vision frames
```

This avoids coupling terminal rendering FPS to camera inference FPS. The viewer can keep rendering even if hand tracking drops frames.

## Internal HandPoseFrame Contract

Use an internal format that can accept both Vision and MediaPipe later:

```rust
struct HandPoseFrame {
    timestamp_ms: u64,
    hands: Vec<TrackedHand>,
}

struct TrackedHand {
    handedness: Option<Handedness>,
    confidence: f32,
    joints_2d: [Option<Joint2D>; 21],
    joints_world: Option<[Joint3; 21]>,
}
```

For Vision MVP, fill `joints_2d` and leave `joints_world = None`. For MediaPipe later, fill both.

## Gesture Mapping For First Proof

Pick camera controls that are tolerant of jitter:

| Gesture Signal | Camera Control | Why |
|---|---|---|
| Open palm centroid moves left/right/up/down | Orbit yaw/elevation or free-look yaw/pitch | Direct, visible, easy to debug |
| Pinch distance thumb tip to index tip | Zoom or forward/back speed | Continuous scalar; common hand-control pattern |
| Fist/open palm confidence threshold | Enable/disable hand-control mode | Prevents accidental camera drift |
| Two hands distance | Optional zoom in later | Save for second pass |

Do not start with per-finger symbolic gestures. Start with continuous control plus dead zones and smoothing.

## Proof Plan

1. Build a standalone `vision_hand_probe` example before touching production control code.
2. Feed one saved frame through `VNImageRequestHandler` and print 21 joints.
3. Capture live frames at 640x480 or lower, run `VNSequenceRequestHandler`, and log per-frame latency, detection rate, and dropped frames.
4. Add a terminal debug overlay that prints: detected hand count, confidence, pinch scalar, palm centroid, inference ms.
5. Only then connect the reducer to tortuise camera commands.

## Caveats

- Apple docs are JS-rendered; accessible evidence above uses `objc2-vision` docs that mirror Apple API comments plus the Apple Developer URLs linked from those docs.
- Vision latency and actual compute device use on MacBook/Mac mini remain unverified. The first implementation gate should include a local benchmark on the actual machine.
- Terminal-launched camera permission should be tested early. If AVFoundation refuses access or prompts poorly, an app-bundled helper or a small Swift/ObjC capture bridge may be cleaner than fighting TCC in the core viewer.
- MediaPipe remains attractive for world landmarks and cross-platform parity, but it is not the smallest Mac/Rust MVP.
