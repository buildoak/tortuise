use std::path::Path;

use image::imageops::FilterType;
use ndarray::Array4;

use super::error::SharpError;

const SHARP_INPUT_SIZE: usize = 1536;
const SHARP_DISPARITY_FACTOR_DEFAULT: f32 = 1.0;

/// Loads an image and prepares SHARP input tensors.
///
/// The returned tensor uses NCHW layout with shape `(1, 3, 1536, 1536)` and
/// pixel values normalized to `[0.0, 1.0]`.
pub fn load_and_prepare(image_path: &Path) -> Result<(Array4<f32>, f32), SharpError> {
    let image = image::open(image_path).map_err(|err| {
        SharpError::Image(format!(
            "failed to open image '{}': {}",
            image_path.display(),
            err
        ))
    })?;

    let resized = image.resize_exact(
        SHARP_INPUT_SIZE as u32,
        SHARP_INPUT_SIZE as u32,
        FilterType::Lanczos3,
    );
    let rgb = resized.to_rgb8();

    let mut tensor = Array4::<f32>::zeros((1, 3, SHARP_INPUT_SIZE, SHARP_INPUT_SIZE));
    for (col, row, pixel) in rgb.enumerate_pixels() {
        let row = row as usize;
        let col = col as usize;
        let [r, g, b] = pixel.0;
        tensor[[0, 0, row, col]] = r as f32 / 255.0;
        tensor[[0, 1, row, col]] = g as f32 / 255.0;
        tensor[[0, 2, row, col]] = b as f32 / 255.0;
    }

    Ok((tensor, SHARP_DISPARITY_FACTOR_DEFAULT))
}
