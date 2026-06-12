use std::time::Instant;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde_json::Value;

use super::types::{
    CameraPreviewFrame, HandInputMessage, HandLandmark, HandPoseFrame, Handedness, TrackedHand,
};

#[derive(Debug, Clone)]
pub enum SidecarProtocolMessage {
    Ignored,
    Input(HandInputMessage),
    Preview(CameraPreviewFrame),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarProtocolError {
    InvalidJson,
    InvalidSample,
    InvalidLandmarks,
    InvalidPreview,
}

impl SidecarProtocolError {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidJson => "sidecar_json_invalid",
            Self::InvalidSample => "sidecar_sample_invalid",
            Self::InvalidLandmarks => "sidecar_landmarks_invalid",
            Self::InvalidPreview => "sidecar_preview_invalid",
        }
    }
}

pub fn parse_sidecar_line(
    line: &str,
    captured_at: Instant,
) -> Result<SidecarProtocolMessage, SidecarProtocolError> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(SidecarProtocolMessage::Ignored);
    }

    let value: Value = serde_json::from_str(line).map_err(|_| SidecarProtocolError::InvalidJson)?;
    let kind = event_kind(&value);
    match kind.as_deref() {
        Some("hello") | Some("status") => Ok(SidecarProtocolMessage::Ignored),
        Some("error") => Ok(SidecarProtocolMessage::Input(HandInputMessage::Error(
            parse_error_code(&value),
        ))),
        Some("sample") | Some("hand_sample") | Some("gesture_sample") => {
            parse_sample(&value, captured_at).map(SidecarProtocolMessage::Input)
        }
        Some("preview") | Some("camera_preview") => {
            parse_preview(&value, captured_at).map(SidecarProtocolMessage::Preview)
        }
        None if value.get("hands").is_some() => {
            parse_sample(&value, captured_at).map(SidecarProtocolMessage::Input)
        }
        None if value.get("status").is_some() || value.get("hello").is_some() => {
            Ok(SidecarProtocolMessage::Ignored)
        }
        _ => Ok(SidecarProtocolMessage::Ignored),
    }
}

fn event_kind(value: &Value) -> Option<String> {
    value
        .get("type")
        .or_else(|| value.get("kind"))
        .or_else(|| value.get("event"))
        .and_then(Value::as_str)
        .map(|raw| raw.trim().to_ascii_lowercase())
}

fn parse_error_code(value: &Value) -> &'static str {
    let raw = value
        .get("code")
        .or_else(|| value.get("error"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    map_sidecar_error_code(raw)
}

fn map_sidecar_error_code(raw: &str) -> &'static str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "camera_denied" => "camera_denied",
        "camera_unavailable" => "camera_unavailable",
        "camera_input" => "camera_input",
        "camera_output" => "camera_output",
        "vision_perform" => "vision_perform",
        "model_missing" => "model_missing",
        "dependency_import_failed" => "dependency_import_failed",
        "camera_open_failed" => "camera_unavailable",
        "camera_read_failed" => "camera_input",
        "tracker_stalled" => "tracker_stalled",
        "sidecar_exit" => "sidecar_exit",
        "sidecar_read_failed" => "sidecar_read_failed",
        "sidecar_protocol" => "sidecar_protocol",
        _ => "sidecar_error",
    }
}

fn parse_sample(
    value: &Value,
    captured_at: Instant,
) -> Result<HandInputMessage, SidecarProtocolError> {
    let sequence = value
        .get("sequence")
        .or_else(|| value.get("seq"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let detect_ms =
        optional_f32(value.get("detect_ms").or_else(|| value.get("latency_ms"))).unwrap_or(0.0);
    let hands_value = value
        .get("hands")
        .and_then(Value::as_array)
        .ok_or(SidecarProtocolError::InvalidSample)?;
    let mut hands = Vec::with_capacity(hands_value.len());
    for (index, hand) in hands_value.iter().enumerate() {
        hands.push(parse_hand(index, hand)?);
    }

    Ok(HandInputMessage::Sample(HandPoseFrame {
        sequence,
        captured_at,
        detect_ms,
        hands,
    }))
}

fn parse_hand(index: usize, value: &Value) -> Result<TrackedHand, SidecarProtocolError> {
    let id = value
        .get("id")
        .and_then(Value::as_u64)
        .and_then(|id| u8::try_from(id).ok())
        .unwrap_or(index.min(u8::MAX as usize) as u8);
    let (x, y) = parse_hand_position(value)?;
    let pinch = optional_f32(value.get("pinch").or_else(|| value.get("pinch_score")))
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let confidence = optional_f32(
        value
            .get("confidence")
            .or_else(|| value.get("score"))
            .or_else(|| value.get("presence")),
    )
    .unwrap_or(1.0)
    .clamp(0.0, 1.0);
    let handedness = value
        .get("handedness")
        .or_else(|| value.get("chirality"))
        .or_else(|| value.get("label"))
        .and_then(Value::as_str)
        .and_then(parse_handedness);
    let landmarks = value
        .get("landmarks")
        .or_else(|| value.get("joints"))
        .or_else(|| value.get("joints_2d"))
        .map(parse_landmarks)
        .transpose()?;

    Ok(TrackedHand {
        id,
        x,
        y,
        pinch,
        confidence,
        handedness,
        landmarks,
    })
}

fn parse_hand_position(value: &Value) -> Result<(f32, f32), SidecarProtocolError> {
    if let Some(center) = value.get("center") {
        return parse_xy(center).ok_or(SidecarProtocolError::InvalidSample);
    }
    let x = optional_f32(value.get("x")).ok_or(SidecarProtocolError::InvalidSample)?;
    let y = optional_f32(value.get("y")).ok_or(SidecarProtocolError::InvalidSample)?;
    Ok((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)))
}

fn parse_handedness(raw: &str) -> Option<Handedness> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "left" | "l" => Some(Handedness::Left),
        "right" | "r" => Some(Handedness::Right),
        _ => None,
    }
}

fn parse_landmarks(value: &Value) -> Result<[HandLandmark; 21], SidecarProtocolError> {
    let landmarks = value
        .as_array()
        .ok_or(SidecarProtocolError::InvalidLandmarks)?;
    if landmarks.len() != 21 {
        return Err(SidecarProtocolError::InvalidLandmarks);
    }

    let mut parsed = [HandLandmark {
        x: 0.0,
        y: 0.0,
        z: None,
    }; 21];
    for (index, landmark) in landmarks.iter().enumerate() {
        parsed[index] = parse_landmark(landmark)?;
    }
    Ok(parsed)
}

fn parse_landmark(value: &Value) -> Result<HandLandmark, SidecarProtocolError> {
    if let Some(values) = value.as_array() {
        let x = optional_f32(values.first()).ok_or(SidecarProtocolError::InvalidLandmarks)?;
        let y = optional_f32(values.get(1)).ok_or(SidecarProtocolError::InvalidLandmarks)?;
        return Ok(HandLandmark {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            z: optional_f32(values.get(2)),
        });
    }

    let x = optional_f32(value.get("x")).ok_or(SidecarProtocolError::InvalidLandmarks)?;
    let y = optional_f32(value.get("y")).ok_or(SidecarProtocolError::InvalidLandmarks)?;
    Ok(HandLandmark {
        x: x.clamp(0.0, 1.0),
        y: y.clamp(0.0, 1.0),
        z: optional_f32(value.get("z")),
    })
}

fn parse_preview(
    value: &Value,
    captured_at: Instant,
) -> Result<CameraPreviewFrame, SidecarProtocolError> {
    let sequence = value
        .get("sequence")
        .or_else(|| value.get("seq"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let width = value
        .get("width")
        .and_then(Value::as_u64)
        .and_then(|width| usize::try_from(width).ok())
        .ok_or(SidecarProtocolError::InvalidPreview)?;
    let height = value
        .get("height")
        .and_then(Value::as_u64)
        .and_then(|height| usize::try_from(height).ok())
        .ok_or(SidecarProtocolError::InvalidPreview)?;
    if width == 0 || height == 0 || width > 512 || height > 512 {
        return Err(SidecarProtocolError::InvalidPreview);
    }
    let payload = value
        .get("rgb")
        .or_else(|| value.get("payload"))
        .or_else(|| value.get("data"))
        .or_else(|| value.get("rgb_base64"))
        .and_then(Value::as_str)
        .ok_or(SidecarProtocolError::InvalidPreview)?;
    let rgb = BASE64_STANDARD
        .decode(payload.as_bytes())
        .map_err(|_| SidecarProtocolError::InvalidPreview)?;
    if rgb.len() != width.saturating_mul(height).saturating_mul(3) {
        return Err(SidecarProtocolError::InvalidPreview);
    }

    Ok(CameraPreviewFrame {
        sequence,
        captured_at,
        width,
        height,
        rgb,
    })
}

fn parse_xy(value: &Value) -> Option<(f32, f32)> {
    if let Some(values) = value.as_array() {
        let x = optional_f32(values.first())?;
        let y = optional_f32(values.get(1))?;
        return Some((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)));
    }
    let x = optional_f32(value.get("x"))?;
    let y = optional_f32(value.get("y"))?;
    Some((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)))
}

fn optional_f32(value: Option<&Value>) -> Option<f32> {
    let value = value?;
    let number = value.as_f64()? as f32;
    number.is_finite().then_some(number)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn landmarks_json() -> String {
        let landmarks = (0..21)
            .map(|index| {
                format!(
                    "{{\"x\":{},\"y\":{},\"z\":{}}}",
                    index as f32 / 20.0,
                    1.0 - index as f32 / 20.0,
                    index as f32 * 0.01
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("[{landmarks}]")
    }

    #[test]
    fn sidecar_parser_accepts_valid_sample_with_landmarks() {
        let line = format!(
            "{{\"type\":\"sample\",\"sequence\":42,\"detect_ms\":6.5,\"hands\":[{{\"id\":1,\"x\":0.25,\"y\":0.75,\"pinch\":0.8,\"confidence\":0.9,\"handedness\":\"right\",\"landmarks\":{}}}]}}",
            landmarks_json()
        );
        let message = parse_sidecar_line(&line, Instant::now()).expect("sample parses");
        let SidecarProtocolMessage::Input(HandInputMessage::Sample(frame)) = message else {
            panic!("expected sample");
        };
        assert_eq!(frame.sequence, 42);
        assert!((frame.detect_ms - 6.5).abs() < f32::EPSILON);
        assert_eq!(frame.hands.len(), 1);
        assert_eq!(frame.hands[0].id, 1);
        assert_eq!(frame.hands[0].handedness, Some(Handedness::Right));
        let landmarks = frame.hands[0].landmarks.expect("landmarks");
        assert_eq!(landmarks.len(), 21);
        assert!((landmarks[20].x - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sidecar_parser_rejects_invalid_landmark_count() {
        let err = parse_sidecar_line(
            "{\"type\":\"sample\",\"hands\":[{\"x\":0.5,\"y\":0.5,\"landmarks\":[[0,0]]}]}",
            Instant::now(),
        )
        .unwrap_err();
        assert_eq!(err, SidecarProtocolError::InvalidLandmarks);
        assert_eq!(err.code(), "sidecar_landmarks_invalid");
    }

    #[test]
    fn sidecar_parser_rejects_preview_payload_mismatch() {
        let payload = BASE64_STANDARD.encode([255u8, 0, 0]);
        let err = parse_sidecar_line(
            &format!(
                "{{\"type\":\"preview\",\"sequence\":7,\"width\":2,\"height\":1,\"rgb\":\"{payload}\"}}"
            ),
            Instant::now(),
        )
        .unwrap_err();
        assert_eq!(err, SidecarProtocolError::InvalidPreview);
        assert_eq!(err.code(), "sidecar_preview_invalid");
    }

    #[test]
    fn sidecar_parser_maps_error_codes() {
        let message = parse_sidecar_line(
            "{\"type\":\"error\",\"code\":\"camera_denied\"}",
            Instant::now(),
        )
        .expect("error parses");
        assert!(matches!(
            message,
            SidecarProtocolMessage::Input(HandInputMessage::Error("camera_denied"))
        ));

        let message =
            parse_sidecar_line("{\"type\":\"error\",\"code\":\"unknown\"}", Instant::now())
                .expect("error parses");
        assert!(matches!(
            message,
            SidecarProtocolMessage::Input(HandInputMessage::Error("sidecar_error"))
        ));
    }

    #[test]
    fn sidecar_parser_maps_mediapipe_helper_errors() {
        let message = parse_sidecar_line(
            "{\"type\":\"error\",\"code\":\"camera_open_failed\"}",
            Instant::now(),
        )
        .expect("error parses");
        assert!(matches!(
            message,
            SidecarProtocolMessage::Input(HandInputMessage::Error("camera_unavailable"))
        ));

        let message = parse_sidecar_line(
            "{\"type\":\"error\",\"code\":\"model_missing\"}",
            Instant::now(),
        )
        .expect("error parses");
        assert!(matches!(
            message,
            SidecarProtocolMessage::Input(HandInputMessage::Error("model_missing"))
        ));
    }

    #[test]
    fn sidecar_parser_accepts_valid_preview_payload() {
        let payload = BASE64_STANDARD.encode([255u8, 0, 0, 0, 255, 0]);
        let message = parse_sidecar_line(
            &format!(
                "{{\"type\":\"preview\",\"sequence\":7,\"width\":2,\"height\":1,\"rgb\":\"{payload}\"}}"
            ),
            Instant::now(),
        )
        .expect("preview parses");
        let SidecarProtocolMessage::Preview(frame) = message else {
            panic!("expected preview");
        };
        assert_eq!(frame.sequence, 7);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 1);
        assert_eq!(frame.rgb, vec![255, 0, 0, 0, 255, 0]);
    }
}
