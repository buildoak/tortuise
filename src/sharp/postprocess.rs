use crate::math::{clamp_u8, quat_normalize, Vec3};
use crate::splat::Splat;

use super::error::SharpError;
use super::model::SharpOutputs;

/// Converts one SHARP linear-light RGB channel into display-ready sRGB 8-bit.
///
/// SHARP emits colors in linear RGB; tortuise stores colors as sRGB bytes.
/// We apply an approximate sRGB gamma curve (`x^(1/2.2)`) before quantization.
fn linear_to_srgb_u8(channel: f32) -> u8 {
    clamp_u8(channel.powf(1.0 / 2.2) * 255.0)
}

/// Converts SHARP inference outputs into tortuise splats.
pub fn extract_splats(outputs: &SharpOutputs) -> Result<Vec<Splat>, SharpError> {
    let n = outputs.positions.len();
    if outputs.scales.len() != n
        || outputs.rotations.len() != n
        || outputs.colors.len() != n
        || outputs.opacities.len() != n
    {
        return Err(SharpError::PostProcess(format!(
            "mismatched SHARP output lengths: positions={}, scales={}, rotations={}, colors={}, opacities={}",
            outputs.positions.len(),
            outputs.scales.len(),
            outputs.rotations.len(),
            outputs.colors.len(),
            outputs.opacities.len()
        )));
    }

    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        let position = Vec3 {
            x: outputs.positions[i][0],
            y: outputs.positions[i][1],
            z: outputs.positions[i][2],
        };
        let scale = Vec3 {
            x: outputs.scales[i][0],
            y: outputs.scales[i][1],
            z: outputs.scales[i][2],
        };
        let rotation = quat_normalize([
            outputs.rotations[i][0],
            outputs.rotations[i][1],
            outputs.rotations[i][2],
            outputs.rotations[i][3],
        ]);
        let color = [
            linear_to_srgb_u8(outputs.colors[i][0]),
            linear_to_srgb_u8(outputs.colors[i][1]),
            linear_to_srgb_u8(outputs.colors[i][2]),
        ];
        let opacity = outputs.opacities[i].clamp(0.0, 1.0);

        splats.push(Splat {
            position,
            color,
            opacity,
            scale,
            rotation,
        });
    }

    Ok(splats)
}
