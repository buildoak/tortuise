use crate::math::{
    clamp_u8, mat3_mul, mat3_transpose, mat4_inverse, mat4_mul, quat_normalize,
    quat_to_rotation_matrix, quaternion_from_rotation_matrix, symmetric_eigen3, Mat4,
    MAT4_IDENTITY, Vec3,
};
use crate::splat::Splat;

use super::error::SharpError;
use super::model::SharpOutputs;
use super::preprocess::{ImageMetadata, SHARP_INPUT_SIZE};

/// Converts one SHARP linear-light RGB channel into display-ready sRGB 8-bit.
///
/// SHARP emits colors in linear RGB; tortuise stores colors as sRGB bytes.
/// We apply an approximate sRGB gamma curve (`x^(1/2.2)`) before quantization.
fn linear_to_srgb_u8(channel: f32) -> u8 {
    clamp_u8(channel.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0)
}

/// Converts SHARP inference outputs into tortuise splats.
pub fn extract_splats(outputs: &SharpOutputs, metadata: &ImageMetadata) -> Result<Vec<Splat>, SharpError> {
    let n = outputs.positions.len();
    if n == 0 {
        return Err(SharpError::PostProcess(
            "SHARP produced zero Gaussians — check the input image".to_string(),
        ));
    }
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

    let unprojection = compute_unprojection_matrix(metadata);
    let t_linear = [
        [unprojection[0][0], unprojection[0][1], unprojection[0][2]],
        [unprojection[1][0], unprojection[1][1], unprojection[1][2]],
        [unprojection[2][0], unprojection[2][1], unprojection[2][2]],
    ];
    let t_offset = [unprojection[0][3], unprojection[1][3], unprojection[2][3]];

    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        let position = Vec3 {
            x: outputs.positions[i][0],
            y: outputs.positions[i][1],
            z: outputs.positions[i][2],
        };

        let rotation = quat_normalize([
            outputs.rotations[i][0],
            outputs.rotations[i][1],
            outputs.rotations[i][2],
            outputs.rotations[i][3],
        ]);

        let scale = Vec3 {
            x: outputs.scales[i][0],
            y: outputs.scales[i][1],
            z: outputs.scales[i][2],
        };
        // Apple: new_pos = pos @ T_linear^T + offset
        // (pos @ T_linear^T)[j] = sum_i pos[i] * T_linear[j][i]
        // SHARP uses OpenCV conventions (y-down, z-forward).
        // tortuise's renderer uses y-up, z-backward. Flip y and z after unprojection.
        let new_position = Vec3 {
            x: position.x * t_linear[0][0]
                + position.y * t_linear[0][1]
                + position.z * t_linear[0][2]
                + t_offset[0],
            y: -(position.x * t_linear[1][0]
                + position.y * t_linear[1][1]
                + position.z * t_linear[1][2]
                + t_offset[1]),
            z: -(position.x * t_linear[2][0]
                + position.y * t_linear[2][1]
                + position.z * t_linear[2][2]
                + t_offset[2]),
        };

        let rotation_matrix = quat_to_rotation_matrix(rotation);
        let scale_diag = [
            [scale.x * scale.x, 0.0, 0.0],
            [0.0, scale.y * scale.y, 0.0],
            [0.0, 0.0, scale.z * scale.z],
        ];
        let covariance = mat3_mul(
            mat3_mul(rotation_matrix, scale_diag),
            mat3_transpose(rotation_matrix),
        );
        // Apply the same y/z flip to the covariance transform.
        // flip = diag(1, -1, -1) converts OpenCV → tortuise coordinate system.
        let flip_t = [
            [t_linear[0][0], t_linear[0][1], t_linear[0][2]],
            [-t_linear[1][0], -t_linear[1][1], -t_linear[1][2]],
            [-t_linear[2][0], -t_linear[2][1], -t_linear[2][2]],
        ];
        let covariance_new = mat3_mul(mat3_mul(flip_t, covariance), mat3_transpose(flip_t));
        let (eigvecs, eigvals) = symmetric_eigen3(covariance_new);

        let new_rotation = quat_normalize(quaternion_from_rotation_matrix(eigvecs));
        let new_scale = Vec3::new(
            eigvals[0].max(0.0).sqrt().max(1e-7),
            eigvals[1].max(0.0).sqrt().max(1e-7),
            eigvals[2].max(0.0).sqrt().max(1e-7),
        );

        let color = [
            linear_to_srgb_u8(outputs.colors[i][0]),
            linear_to_srgb_u8(outputs.colors[i][1]),
            linear_to_srgb_u8(outputs.colors[i][2]),
        ];
        let opacity = outputs.opacities[i].clamp(0.0, 1.0);

        splats.push(Splat {
            position: new_position,
            color,
            opacity,
            scale: new_scale,
            rotation: new_rotation,
        });
    }

    Ok(splats)
}

fn compute_unprojection_matrix(metadata: &ImageMetadata) -> Mat4 {
    let sharp_size = SHARP_INPUT_SIZE as f32;
    let fx = metadata.focal_length_px * sharp_size / metadata.original_width as f32;
    let fy = metadata.focal_length_px * sharp_size / metadata.original_height as f32;
    let cx = sharp_size * 0.5;
    let cy = sharp_size * 0.5;

    let mut intrinsics = MAT4_IDENTITY;
    intrinsics[0][0] = fx;
    intrinsics[1][1] = fy;
    intrinsics[0][2] = cx;
    intrinsics[1][2] = cy;

    let mut ndc = MAT4_IDENTITY;
    ndc[0][0] = 2.0 / sharp_size;
    ndc[1][1] = 2.0 / sharp_size;
    ndc[0][2] = -1.0;
    ndc[1][2] = -1.0;

    let projection = mat4_mul(ndc, intrinsics);
    mat4_inverse(projection).expect("projection matrix must be invertible")
}
