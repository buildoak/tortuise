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

/// Ensures the SHARP model is downloaded and cached locally.
///
/// If the model is missing, this will prompt the user for consent and show a
/// download progress bar. Call this BEFORE starting the Matrix rain spinner.
#[cfg(feature = "sharp")]
pub fn ensure_model_downloaded() -> Result<(), SharpError> {
    download::ensure_model_available()?;
    Ok(())
}

/// Reconstructs Gaussian splats from a supported image using SHARP.
///
/// Assumes `ensure_model_downloaded()` has already been called (or the model is
/// already cached). This is the compute-heavy part that should be wrapped in
/// the Matrix rain spinner.
///
/// Internally retries inference if ORT produces degenerate output (a known
/// non-determinism issue with the CPU execution provider).
#[cfg(feature = "sharp")]
pub fn reconstruct_from_image(image_path: &Path) -> Result<Vec<crate::splat::Splat>, SharpError> {
    let model_path = download::ensure_model_available()?;
    let (input_tensor, disparity, metadata) = preprocess::load_and_prepare(image_path)?;
    let (_session, outputs) =
        model::run_inference_with_retry(&model_path, &input_tensor, disparity)?;
    postprocess::extract_splats(&outputs, &metadata)
}
