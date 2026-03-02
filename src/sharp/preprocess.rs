use std::path::Path;

use image::imageops::FilterType;
use ndarray::Array4;

use super::error::SharpError;

pub(super) const SHARP_INPUT_SIZE: usize = 1536;

/// Metadata derived from the source image required for SHARP unprojection.
pub struct ImageMetadata {
    pub original_width: u32,
    pub original_height: u32,
    pub focal_length_px: f32,
}

/// Loads an image and prepares SHARP input tensors.
///
/// The returned tensor uses NCHW layout with shape `(1, 3, 1536, 1536)` and
/// pixel values normalized to `[0.0, 1.0]`.
pub fn load_and_prepare(image_path: &Path) -> Result<(Array4<f32>, f32, ImageMetadata), SharpError> {
    let image = image::open(image_path).map_err(|err| {
        SharpError::Image(format!(
            "failed to open image '{}': {}",
            image_path.display(),
            err
        ))
    })?;

    let original_width = image.width();
    let original_height = image.height();
    if original_width == 0 || original_height == 0 {
        return Err(SharpError::Image(format!(
            "image '{}' has invalid dimensions {}x{}",
            image_path.display(),
            original_width,
            original_height
        )));
    }

    let f_35mm = 30.0f32;
    let image_diag = ((original_width as f32).powi(2) + (original_height as f32).powi(2)).sqrt();
    let sensor_diag = (36.0f32.powi(2) + 24.0f32.powi(2)).sqrt();
    let focal_length_px = f_35mm * image_diag / sensor_diag;
    let disparity_factor = focal_length_px / original_width as f32;

    let metadata = ImageMetadata {
        original_width,
        original_height,
        focal_length_px,
    };

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

    Ok((tensor, disparity_factor, metadata))
}
