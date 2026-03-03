use crate::camera::Camera;
use crate::math::{mat3_mul, mat3_transpose, quat_to_rotation_matrix, Vec3};

pub const GAUSSIAN_SIGMA_CUTOFF: f32 = 4.0;
pub const MIN_SPLAT_RADIUS: f32 = 0.3;
pub const MIN_GAUSSIAN_CONTRIBUTION: f32 = 0.001;
pub const SATURATION_EPSILON: f32 = 0.999;

#[derive(Debug, Clone, Copy)]
pub struct Splat {
    pub position: Vec3,
    pub color: [u8; 3],
    pub opacity: f32,
    pub scale: Vec3,
    pub rotation: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectedSplat {
    pub screen_x: f32,
    pub screen_y: f32,
    pub depth: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub color: [u8; 3],
    pub opacity: f32,
    pub inv_cov_a: f32,
    pub inv_cov_b: f32,
    pub inv_cov_c: f32,
    pub original_index: usize,
}

pub fn compute_3d_covariance(scale: Vec3, rotation: [f32; 4]) -> [[f32; 3]; 3] {
    let r = quat_to_rotation_matrix(rotation);
    let s = [
        scale.x.max(1e-4) * scale.x.max(1e-4),
        scale.y.max(1e-4) * scale.y.max(1e-4),
        scale.z.max(1e-4) * scale.z.max(1e-4),
    ];

    // Cov = R * diag(s^2) * R^T
    let mut d = [[0.0; 3]; 3];
    d[0][0] = s[0];
    d[1][1] = s[1];
    d[2][2] = s[2];

    mat3_mul(mat3_mul(r, d), mat3_transpose(r))
}

pub fn project_covariance_to_2d(
    cov_3d: [[f32; 3]; 3],
    camera: &Camera,
    point_view: Vec3,
    fx: f32,
    fy: f32,
) -> (f32, f32, f32) {
    let view_rot = camera.view_rotation();
    let cov_view = mat3_mul(mat3_mul(view_rot, cov_3d), mat3_transpose(view_rot));

    let z = point_view.z.max(1e-4);
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;

    let jac = [
        [fx * inv_z, 0.0, -fx * point_view.x * inv_z2],
        [0.0, fy * inv_z, -fy * point_view.y * inv_z2],
    ];

    let mut j_cov = [[0.0; 3]; 2];
    for row in 0..2 {
        for col in 0..3 {
            j_cov[row][col] = jac[row][0] * cov_view[0][col]
                + jac[row][1] * cov_view[1][col]
                + jac[row][2] * cov_view[2][col];
        }
    }

    let cov_a = j_cov[0][0] * jac[0][0] + j_cov[0][1] * jac[0][1] + j_cov[0][2] * jac[0][2];
    let cov_b = j_cov[0][0] * jac[1][0] + j_cov[0][1] * jac[1][1] + j_cov[0][2] * jac[1][2];
    let cov_c = j_cov[1][0] * jac[1][0] + j_cov[1][1] * jac[1][1] + j_cov[1][2] * jac[1][2];

    // Small diagonal stabilization in pixel units.
    (cov_a + 1e-3, cov_b, cov_c + 1e-3)
}

pub fn compute_2d_gaussian_extent(cov_a: f32, cov_b: f32, cov_c: f32) -> (f32, f32) {
    let trace = cov_a + cov_c;
    let det = cov_a * cov_c - cov_b * cov_b;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();

    let lambda1 = 0.5 * (trace + disc);
    let _lambda2 = 0.5 * (trace - disc);

    // Use 4-sigma cutoff for softer, wider falloff.
    let extent = GAUSSIAN_SIGMA_CUTOFF * lambda1.max(0.0).sqrt();
    (extent, extent)
}

pub fn invert_2x2_covariance(cov_a: f32, cov_b: f32, cov_c: f32) -> Option<(f32, f32, f32)> {
    let det = cov_a * cov_c - cov_b * cov_b;
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some((cov_c * inv_det, -cov_b * inv_det, cov_a * inv_det))
}

pub fn evaluate_2d_gaussian(
    dx: f32,
    dy: f32,
    inv_cov_a: f32,
    inv_cov_b: f32,
    inv_cov_c: f32,
) -> f32 {
    let q = dx * dx * inv_cov_a + 2.0 * dx * dy * inv_cov_b + dy * dy * inv_cov_c;
    // 4-sigma: cutoff at q = 2*(4^2) = 32
    if q > 32.0 {
        return 0.0;
    }
    (-0.5 * q).exp()
}

/// Bounding information for a set of splats.
pub struct SceneBounds {
    pub centroid: Vec3,
    /// Maximum distance from the centroid to any splat.
    pub extent: f32,
}

/// Compute the centroid and extent of a splat cloud.
///
/// Returns `None` for an empty slice.
pub fn compute_scene_bounds(splats: &[Splat]) -> Option<SceneBounds> {
    if splats.is_empty() {
        return None;
    }

    let n = splats.len() as f32;
    let mut sum = Vec3::ZERO;
    for s in splats {
        sum += s.position;
    }
    let centroid = sum * (1.0 / n);

    let mut max_dist_sq: f32 = 0.0;
    for s in splats {
        let d = s.position - centroid;
        max_dist_sq = max_dist_sq.max(d.length_squared());
    }

    Some(SceneBounds {
        centroid,
        extent: max_dist_sq.sqrt(),
    })
}
