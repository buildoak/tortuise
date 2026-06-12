#!/usr/bin/env python3
"""MediaPipe hand-tracking sidecar for tortuise.

The sidecar writes versioned JSONL events to stdout and human-readable logs to
stderr. It intentionally keeps OpenCV and MediaPipe imports lazy so --doctor can
explain missing dependencies without crashing.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = 1
SOURCE = "tortuise-mediapipe-hands-sidecar"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
PINKY_MCP = 17


def log(message: str) -> None:
    print(f"{SOURCE}: {message}", file=sys.stderr, flush=True)


def emit(event_type: str, **payload: Any) -> None:
    event = {
        "v": PROTOCOL_VERSION,
        "type": event_type,
        "source": SOURCE,
        **payload,
    }
    print(json.dumps(event, separators=(",", ":"), sort_keys=True), flush=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_size(raw: str) -> tuple[int, int]:
    if "x" not in raw.lower():
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    width_raw, height_raw = raw.lower().split("x", 1)
    try:
        width = int(width_raw)
        height = int(height_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer WIDTHxHEIGHT") from exc
    if width < 16 or height < 9 or width > 640 or height > 480:
        raise argparse.ArgumentTypeError("preview size must be within 16x9..640x480")
    return width, height


def module_smoke() -> dict[str, dict[str, Any]]:
    modules = {}
    for import_name, display_name in (
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
    ):
        try:
            module = __import__(import_name)
            modules[display_name] = {
                "ok": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:  # pragma: no cover - diagnostic path
            modules[display_name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return modules


def run_doctor(model: Path) -> int:
    modules = module_smoke()
    model_exists = model.is_file()
    ok = all(item["ok"] for item in modules.values()) and model_exists
    emit(
        "doctor",
        ok=ok,
        model={
            "path": str(model),
            "exists": model_exists,
            "bytes": model.stat().st_size if model_exists else 0,
        },
        modules=modules,
    )
    if not model_exists:
        log(f"model missing: {model}")
    for name, result in modules.items():
        if not result["ok"]:
            log(f"import failed for {name}: {result['error']}")
    return 0 if ok else 1


def import_runtime_modules() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as exc:
        emit("error", code="dependency_import_failed", message=f"{type(exc).__name__}: {exc}")
        raise
    return cv2, mp, np, mp_python, vision


def distance_2d(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def pinch_score(landmarks: list[Any]) -> float:
    if len(landmarks) <= max(THUMB_TIP, INDEX_TIP):
        return 0.0
    pinch_distance = distance_2d(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    if len(landmarks) > PINKY_MCP:
        # Match the browser prototype: thumb-tip/index-tip distance normalized
        # by palm width, then converted to a high-is-pinched score.
        palm_distance = max(0.04, distance_2d(landmarks[INDEX_MCP], landmarks[PINKY_MCP]))
        normalized = pinch_distance / palm_distance
        return clamp(1.0 - normalized / 0.55, 0.0, 1.0)
    return clamp(1.0 - pinch_distance / 0.18, 0.0, 1.0)


def handedness_payload(categories: Iterable[Any], index: int) -> tuple[str | None, float]:
    label = None
    score = 0.0
    best = None
    for category in categories:
        if best is None or float(category.score) > float(best.score):
            best = category
    if best is not None:
        label = str(best.category_name)
        score = float(best.score)
    return label or f"hand-{index}", score


def landmark_payload(landmarks: list[Any], mirror: bool) -> list[dict[str, float]]:
    out = []
    for idx, landmark in enumerate(landmarks):
        x = 1.0 - float(landmark.x) if mirror else float(landmark.x)
        out.append(
            {
                "i": idx,
                "x": clamp(x, 0.0, 1.0),
                "y": clamp(float(landmark.y), 0.0, 1.0),
                "z": float(landmark.z),
            }
        )
    return out


def world_landmark_payload(landmarks: list[Any]) -> list[dict[str, float]]:
    out = []
    for idx, landmark in enumerate(landmarks):
        out.append(
            {
                "i": idx,
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
            }
        )
    return out


def summarize_hand(
    index: int,
    landmarks: list[Any],
    world_landmarks: list[Any],
    handedness: Iterable[Any],
    mirror: bool,
) -> dict[str, Any]:
    label, score = handedness_payload(handedness, index)
    if landmarks:
        center_x = sum(float(point.x) for point in landmarks) / len(landmarks)
        center_y = sum(float(point.y) for point in landmarks) / len(landmarks)
    else:
        center_x = 0.0
        center_y = 0.0
    x = 1.0 - center_x if mirror else center_x
    confidence = score if score > 0.0 else 1.0
    return {
        "id": index,
        "label": label,
        "score": score,
        "x": clamp(x, 0.0, 1.0),
        "y": clamp(center_y, 0.0, 1.0),
        "pinch": pinch_score(landmarks),
        "confidence": confidence,
        "landmarks": landmark_payload(landmarks, mirror),
        "world_landmarks": world_landmark_payload(world_landmarks),
    }


def result_payload(result: Any, mirror: bool) -> list[dict[str, Any]]:
    hands = []
    landmarks_by_hand = result.hand_landmarks or []
    world_by_hand = result.hand_world_landmarks or []
    handedness_by_hand = result.handedness or []
    for idx, landmarks in enumerate(landmarks_by_hand):
        world_landmarks = world_by_hand[idx] if idx < len(world_by_hand) else []
        handedness = handedness_by_hand[idx] if idx < len(handedness_by_hand) else []
        hands.append(summarize_hand(idx, landmarks, world_landmarks, handedness, mirror))
    return hands


def preview_payload(cv2: Any, rgb_frame: Any, width: int, height: int) -> str:
    resized = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_AREA)
    return base64.b64encode(resized.tobytes()).decode("ascii")


def open_camera(cv2: Any, camera_index: int) -> Any:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        emit("error", code="camera_open_failed", camera_index=camera_index)
        raise RuntimeError(f"camera {camera_index} did not open")
    return capture


def run_capture(args: argparse.Namespace) -> int:
    model = Path(args.model)
    if not model.is_file():
        emit("error", code="model_missing", model=str(model))
        log(f"model missing: {model}")
        return 2

    try:
        cv2, mp, _np, mp_python, vision = import_runtime_modules()
        capture = open_camera(cv2, args.camera_index)
    except Exception as exc:
        log(f"startup failed: {type(exc).__name__}: {exc}")
        return 2

    target_interval = 1.0 / max(1, args.fps)
    next_sample_at = 0.0
    next_preview_at = 0.0
    preview_interval = 1.0 / max(1, args.preview_fps)
    sequence = 0
    emitted = 0
    read_failures = 0
    stream_started_at = time.monotonic()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    emit(
        "status",
        status="starting",
        mode="probe" if args.probe_camera else "live",
        model=str(model),
        fps=args.fps,
        preview=args.preview,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        mirror=args.mirror,
    )

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            emit("status", status="ready")
            while True:
                if args.probe_camera and emitted >= args.frames:
                    break

                ok, bgr_frame = capture.read()
                if not ok:
                    read_failures += 1
                    if read_failures >= args.max_camera_read_failures:
                        emit(
                            "error",
                            code="camera_read_failed",
                            sequence=sequence + 1,
                            failures=read_failures,
                        )
                        return 3
                    time.sleep(min(0.02 * read_failures, 0.25))
                    continue
                read_failures = 0

                now = time.monotonic()
                if now < next_sample_at:
                    sleep_for = min(next_sample_at - now, 0.01)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    continue

                sequence += 1
                emitted += 1
                next_sample_at = time.monotonic() + target_interval

                started = time.perf_counter()
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(time.time() * 1000)
                video_timestamp_ms = int((time.monotonic() - stream_started_at) * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, video_timestamp_ms)
                detect_ms = (time.perf_counter() - started) * 1000.0
                height, width = rgb_frame.shape[:2]

                emit(
                    "hand_sample",
                    sequence=sequence,
                    timestamp_ms=timestamp_ms,
                    video_timestamp_ms=video_timestamp_ms,
                    detect_ms=round(detect_ms, 3),
                    image={"width": width, "height": height, "mirrored": args.mirror},
                    hands=result_payload(result, args.mirror),
                )

                if args.preview and time.monotonic() >= next_preview_at:
                    next_preview_at = time.monotonic() + preview_interval
                    emit(
                        "preview",
                        sequence=sequence,
                        timestamp_ms=timestamp_ms,
                        width=args.preview_width,
                        height=args.preview_height,
                        format="rgb24",
                        encoding="base64",
                        data=preview_payload(cv2, rgb_frame, args.preview_width, args.preview_height),
                    )
    except KeyboardInterrupt:
        emit("status", status="interrupted")
        return 130
    finally:
        capture.release()

    emit("status", status="complete", frames=emitted)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor", action="store_true", help="check imports and model path")
    parser.add_argument("--probe-camera", action="store_true", help="run a finite camera probe")
    parser.add_argument("--frames", type=int, default=30, help="frames to emit for --probe-camera")
    parser.add_argument(
        "--model",
        default=os.environ.get("TORTUISE_MEDIAPIPE_MODEL", str(DEFAULT_MODEL)),
        help="path to hand_landmarker.task",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--fps", type=int, default=30, help="target hand sample FPS")
    parser.add_argument("--num-hands", type=int, default=2, help="maximum hands to track")
    parser.add_argument("--preview", action="store_true", help="emit preview RGB frames as base64")
    parser.add_argument("--preview-size", type=parse_size, help="preview size as WIDTHxHEIGHT")
    parser.add_argument("--preview-width", type=int, default=64, help="preview RGB width")
    parser.add_argument("--preview-height", type=int, default=36, help="preview RGB height")
    parser.add_argument("--preview-fps", type=int, default=8, help="target preview FPS")
    parser.add_argument(
        "--max-camera-read-failures",
        type=int,
        default=30,
        help="consecutive failed camera reads before emitting camera_read_failed",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="do not mirror normalized x coordinates",
    )
    parser.set_defaults(mirror=True)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be in 1..=60")
    if not 1 <= args.preview_fps <= 30:
        parser.error("--preview-fps must be in 1..=30")
    if not 1 <= args.num_hands <= 4:
        parser.error("--num-hands must be in 1..=4")
    if args.max_camera_read_failures < 1:
        parser.error("--max-camera-read-failures must be >= 1")
    if args.preview_size:
        args.preview_width, args.preview_height = args.preview_size
    if args.preview_width < 16 or args.preview_height < 9:
        parser.error("--preview-width/--preview-height are too small")
    if args.preview_width > 640 or args.preview_height > 480:
        parser.error("--preview-width/--preview-height are too large")
    for name in (
        "min_detection_confidence",
        "min_presence_confidence",
        "min_tracking_confidence",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in 0.0..=1.0")
    return args


def main() -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(), parser)
    if args.doctor:
        return run_doctor(Path(args.model))
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
