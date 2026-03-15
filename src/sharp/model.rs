use std::path::Path;

use ndarray::{Array1, Array4, Axis};
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;

use super::error::SharpError;

/// Maximum number of inference attempts before giving up.
///
/// ORT's CPU execution provider exhibits non-deterministic behavior with the
/// SHARP model. Some runs produce valid Gaussian positions while others yield
/// degenerate output (near-zero or astronomically large values). Retrying
/// with a fresh session reliably produces valid output within a few attempts.
const MAX_INFERENCE_ATTEMPTS: usize = 5;

/// Minimum acceptable position range (axis extent) for SHARP output.
///
/// Valid SHARP reconstructions produce positions spanning at least ~0.1 units
/// along each axis. Degenerate runs typically produce positions clustered
/// near zero (extent < 1e-3) or spread over 1e10+ units.
const MIN_POSITION_EXTENT: f32 = 0.05;

/// Maximum acceptable position magnitude.
///
/// Valid positions are typically in [-20, 20]. Anything beyond 1000 indicates
/// corrupted inference.
const MAX_POSITION_MAGNITUDE: f32 = 1000.0;

/// Raw SHARP model output tensors, converted to owned Rust vectors.
pub struct SharpOutputs {
    /// 3D Gaussian means from `mean_vectors_3d_positions`.
    pub positions: Vec<[f32; 3]>,
    /// Gaussian scales from `singular_values_scales`.
    pub scales: Vec<[f32; 3]>,
    /// Gaussian quaternion rotations from `quaternions_rotations`.
    pub rotations: Vec<[f32; 4]>,
    /// Linear RGB colors from `colors_rgb_linear`.
    pub colors: Vec<[f32; 3]>,
    /// Alpha values from `opacities_alpha_channel`.
    pub opacities: Vec<f32>,
}

/// Creates an ONNX Runtime session for the SHARP model.
///
/// Uses the CPU execution provider. CoreML is excluded because the FP16
/// weight Cast nodes in the model trigger type validation errors on CoreML.
pub fn create_session(model_path: &Path) -> Result<Session, SharpError> {
    let builder = Session::builder().map_err(to_model_error)?;
    builder
        .with_execution_providers([ort::ep::CPU::default().build()])
        .map_err(to_model_error)?
        .commit_from_file(model_path)
        .map_err(to_model_error)
}

/// Runs SHARP inference with automatic retry on degenerate output.
///
/// ORT's CPU EP can produce non-deterministic results with the SHARP model.
/// This function validates the output and retries with a fresh session if
/// the positions are degenerate (near-zero extent or astronomical values).
pub fn run_inference_with_retry(
    model_path: &Path,
    input_tensor: &Array4<f32>,
    disparity_factor: f32,
) -> Result<(Session, SharpOutputs), SharpError> {
    for attempt in 1..=MAX_INFERENCE_ATTEMPTS {
        let mut session = create_session(model_path)?;
        let outputs = run_inference(&mut session, input_tensor, disparity_factor)?;

        match validate_outputs(&outputs) {
            OutputQuality::Valid => {
                return Ok((session, outputs));
            }
            OutputQuality::Degenerate(reason) => {
                if attempt >= MAX_INFERENCE_ATTEMPTS {
                    return Err(SharpError::Model(format!(
                        "inference produced degenerate output after {} attempts: {}",
                        MAX_INFERENCE_ATTEMPTS, reason
                    )));
                }
            }
        }
    }

    unreachable!()
}

/// Runs SHARP inference and extracts the named output tensors.
fn run_inference(
    session: &mut Session,
    input_tensor: &Array4<f32>,
    disparity_factor: f32,
) -> Result<SharpOutputs, SharpError> {
    let disparity = Array1::<f32>::from_vec(vec![disparity_factor]);

    let image_value = TensorRef::from_array_view(input_tensor).map_err(to_model_error)?;
    let disparity_value = TensorRef::from_array_view(&disparity).map_err(to_model_error)?;

    let inputs = ort::inputs![
        "image" => image_value,
        "disparity_factor" => disparity_value,
    ];

    let outputs = session.run(inputs).map_err(to_model_error)?;

    let positions = extract_vec3_output(&outputs, "mean_vectors_3d_positions")?;
    let scales = extract_vec3_output(&outputs, "singular_values_scales")?;
    let rotations = extract_vec4_output(&outputs, "quaternions_rotations")?;
    let colors = extract_vec3_output(&outputs, "colors_rgb_linear")?;
    let opacities = extract_scalar_output(&outputs, "opacities_alpha_channel")?;

    Ok(SharpOutputs {
        positions,
        scales,
        rotations,
        colors,
        opacities,
    })
}

enum OutputQuality {
    Valid,
    Degenerate(String),
}

/// Validates SHARP output for signs of degenerate inference.
fn validate_outputs(outputs: &SharpOutputs) -> OutputQuality {
    let n = outputs.positions.len();
    if n == 0 {
        return OutputQuality::Degenerate("zero positions".to_string());
    }

    let mut pos_min = [f32::INFINITY; 3];
    let mut pos_max = [f32::NEG_INFINITY; 3];
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;

    for p in &outputs.positions {
        for j in 0..3 {
            if p[j].is_nan() {
                nan_count += 1;
            } else if p[j].is_infinite() {
                inf_count += 1;
            } else {
                if p[j] < pos_min[j] {
                    pos_min[j] = p[j];
                }
                if p[j] > pos_max[j] {
                    pos_max[j] = p[j];
                }
            }
        }
    }

    if nan_count > 0 || inf_count > 0 {
        return OutputQuality::Degenerate(format!(
            "{} NaN + {} Inf values in positions",
            nan_count, inf_count
        ));
    }

    let extent_x = pos_max[0] - pos_min[0];
    let extent_y = pos_max[1] - pos_min[1];
    let extent_z = pos_max[2] - pos_min[2];
    let max_extent = extent_x.max(extent_y).max(extent_z);

    if max_extent < MIN_POSITION_EXTENT {
        return OutputQuality::Degenerate(format!(
            "position extent too small ({:.2e})",
            max_extent
        ));
    }

    let max_abs = pos_max[0]
        .abs()
        .max(pos_max[1].abs())
        .max(pos_max[2].abs())
        .max(pos_min[0].abs())
        .max(pos_min[1].abs())
        .max(pos_min[2].abs());

    if max_abs > MAX_POSITION_MAGNITUDE {
        return OutputQuality::Degenerate(format!(
            "position magnitude too large ({:.2e})",
            max_abs
        ));
    }

    let zero_scale_count = outputs
        .scales
        .iter()
        .filter(|s| s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0)
        .count();
    let zero_scale_ratio = zero_scale_count as f32 / n as f32;
    if zero_scale_ratio > 0.95 {
        return OutputQuality::Degenerate(format!(
            "{:.0}% of scales are zero",
            zero_scale_ratio * 100.0
        ));
    }

    OutputQuality::Valid
}

fn extract_vec3_output(outputs: &SessionOutputs, name: &str) -> Result<Vec<[f32; 3]>, SharpError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| SharpError::Model(format!("missing output tensor '{name}'")))?;
    let view = value.try_extract_array::<f32>().map_err(to_model_error)?;
    let shape = view.shape();

    if shape.len() != 3 || shape[0] != 1 || shape[2] != 3 {
        return Err(SharpError::Model(format!(
            "unexpected shape for '{name}': expected (1, N, 3), got {shape:?}"
        )));
    }

    let rows = view.index_axis(Axis(0), 0);
    let mut values = Vec::with_capacity(rows.shape()[0]);
    for row in rows.outer_iter() {
        values.push([row[0], row[1], row[2]]);
    }
    Ok(values)
}

fn extract_vec4_output(outputs: &SessionOutputs, name: &str) -> Result<Vec<[f32; 4]>, SharpError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| SharpError::Model(format!("missing output tensor '{name}'")))?;
    let view = value.try_extract_array::<f32>().map_err(to_model_error)?;
    let shape = view.shape();

    if shape.len() != 3 || shape[0] != 1 || shape[2] != 4 {
        return Err(SharpError::Model(format!(
            "unexpected shape for '{name}': expected (1, N, 4), got {shape:?}"
        )));
    }

    let rows = view.index_axis(Axis(0), 0);
    let mut values = Vec::with_capacity(rows.shape()[0]);
    for row in rows.outer_iter() {
        values.push([row[0], row[1], row[2], row[3]]);
    }
    Ok(values)
}

fn extract_scalar_output(outputs: &SessionOutputs, name: &str) -> Result<Vec<f32>, SharpError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| SharpError::Model(format!("missing output tensor '{name}'")))?;
    let view = value.try_extract_array::<f32>().map_err(to_model_error)?;
    let shape = view.shape();

    if shape.len() != 2 || shape[0] != 1 {
        return Err(SharpError::Model(format!(
            "unexpected shape for '{name}': expected (1, N), got {shape:?}"
        )));
    }

    let values = view.index_axis(Axis(0), 0).iter().copied().collect();
    Ok(values)
}

fn to_model_error<E>(error: E) -> SharpError
where
    E: std::fmt::Display,
{
    SharpError::Model(error.to_string())
}
