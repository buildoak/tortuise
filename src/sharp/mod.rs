//! SHARP inference integration entrypoint.

use std::path::Path;

#[cfg(feature = "sharp")]
mod download;
#[cfg(feature = "sharp")]
mod error;
#[cfg(feature = "sharp")]
mod model;
#[cfg(feature = "sharp")]
mod postprocess;
#[cfg(feature = "sharp")]
mod preprocess;

/// Error type returned by SHARP reconstruction operations.
#[cfg(feature = "sharp")]
pub use error::SharpError;

/// Reconstructs Gaussian splats from a supported image using SHARP.
///
/// This orchestrates model availability, session setup, image preprocessing,
/// inference execution, and output postprocessing into `Splat` values.
#[cfg(feature = "sharp")]
pub fn reconstruct_from_image(image_path: &Path) -> Result<Vec<crate::splat::Splat>, SharpError> {
    let model_path = download::ensure_model_available()?;
    let mut session = model::create_session(&model_path)?;
    let (input_tensor, disparity) = preprocess::load_and_prepare(image_path)?;
    let outputs = model::run_inference(&mut session, &input_tensor, disparity)?;
    postprocess::extract_splats(&outputs)
}
