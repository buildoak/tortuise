use std::path::Path;

use ndarray::{Array1, Array4, Axis};
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;

use super::error::SharpError;

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
pub fn create_session(model_path: &Path) -> Result<Session, SharpError> {
    let mut builder = Session::builder().map_err(to_model_error)?;

    #[cfg(target_os = "macos")]
    {
        builder = builder
            .with_execution_providers([ort::ep::CoreML::default().build()])
            .map_err(to_model_error)?;
    }

    builder.commit_from_file(model_path).map_err(to_model_error)
}

/// Runs SHARP inference and extracts the named output tensors.
pub fn run_inference(
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

fn extract_vec3_output(outputs: &SessionOutputs, name: &str) -> Result<Vec<[f32; 3]>, SharpError> {
    let view = outputs[name]
        .try_extract_array::<f32>()
        .map_err(to_model_error)?;
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
    let view = outputs[name]
        .try_extract_array::<f32>()
        .map_err(to_model_error)?;
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
    let view = outputs[name]
        .try_extract_array::<f32>()
        .map_err(to_model_error)?;
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
