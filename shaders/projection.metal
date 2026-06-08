#include <metal_stdlib>
using namespace metal;

struct SplatData {
    float pos_x, pos_y, pos_z;
    float scale_x, scale_y, scale_z;
    float rot_w, rot_x, rot_y, rot_z;
    float opacity;
    uint packed_color; // RGBA packed as 0xAABBGGRR
};

struct CameraData {
    float pos_x, pos_y, pos_z;
    float right_x, right_y, right_z;
    float up_x, up_y, up_z;
    float forward_x, forward_y, forward_z;
    float fx, fy;
    float half_w, half_h;
    float near_plane, far_plane;
};

struct TileConfig {
    uint tile_count_x;
    uint tile_count_y;
    uint screen_width;
    uint screen_height;
};

struct ProjectedSplat {
    float screen_x, screen_y, depth;
    float radius_x, radius_y;
    float cov_a, cov_b, cov_c;
    float opacity;
    uint packed_color;
    uint original_index;
    uint tile_min; // packed: (tile_min_y << 16) | tile_min_x
    uint tile_max; // packed: (tile_max_y << 16) | tile_max_x
};

// All matrices stored as row-major float arrays: m[row][col]
// This matches the CPU Rust code exactly and avoids MSL matrix convention confusion.

struct Mat3 {
    float m[3][3]; // m[row][col]
};

// Quaternion to rotation matrix (row-major, matches CPU)
Mat3 quat_to_rotation_matrix(float4 q) {
    float norm = length(q);
    if (norm <= 1e-8) {
        q = float4(1.0, 0.0, 0.0, 0.0);
    } else {
        q /= norm;
    }

    float w = q.x, x = q.y, y = q.z, z = q.w;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    Mat3 r;
    r.m[0][0] = 1.0 - 2.0 * (yy + zz);
    r.m[0][1] = 2.0 * (xy - wz);
    r.m[0][2] = 2.0 * (xz + wy);
    r.m[1][0] = 2.0 * (xy + wz);
    r.m[1][1] = 1.0 - 2.0 * (xx + zz);
    r.m[1][2] = 2.0 * (yz - wx);
    r.m[2][0] = 2.0 * (xz - wy);
    r.m[2][1] = 2.0 * (yz + wx);
    r.m[2][2] = 1.0 - 2.0 * (xx + yy);
    return r;
}

// 3x3 matrix multiply: out = a * b (row-major)
Mat3 mat3_mul(Mat3 a, Mat3 b) {
    Mat3 out;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            out.m[r][c] = a.m[r][0] * b.m[0][c]
                        + a.m[r][1] * b.m[1][c]
                        + a.m[r][2] * b.m[2][c];
        }
    }
    return out;
}

// 3x3 matrix transpose
Mat3 mat3_transpose(Mat3 a) {
    Mat3 out;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            out.m[r][c] = a.m[c][r];
        }
    }
    return out;
}

// Compute 3D covariance: R * diag(s^2) * R^T
Mat3 compute_3d_covariance(float3 scale, float4 rotation) {
    Mat3 r = quat_to_rotation_matrix(rotation);

    float3 s = max(scale, 1e-4f);
    float3 s2 = s * s;

    Mat3 d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            d.m[i][j] = 0.0;
    d.m[0][0] = s2.x;
    d.m[1][1] = s2.y;
    d.m[2][2] = s2.z;

    return mat3_mul(mat3_mul(r, d), mat3_transpose(r));
}

// Project 3D covariance to 2D screen space
// Returns (cov_a, cov_b, cov_c) representing the 2x2 symmetric covariance matrix
float3 project_covariance_to_2d(Mat3 cov_3d, constant CameraData& camera, float3 point_view) {
    // View rotation matrix (row-major), matching CPU camera.view_rotation()
    Mat3 view_rot;
    view_rot.m[0][0] = camera.right_x;
    view_rot.m[0][1] = camera.right_y;
    view_rot.m[0][2] = camera.right_z;
    view_rot.m[1][0] = camera.up_x;
    view_rot.m[1][1] = camera.up_y;
    view_rot.m[1][2] = camera.up_z;
    view_rot.m[2][0] = camera.forward_x;
    view_rot.m[2][1] = camera.forward_y;
    view_rot.m[2][2] = camera.forward_z;

    Mat3 cov_view = mat3_mul(mat3_mul(view_rot, cov_3d), mat3_transpose(view_rot));

    float z = max(point_view.z, 1e-4f);
    float inv_z = 1.0 / z;
    float inv_z2 = inv_z * inv_z;

    // Jacobian J is 2x3 (2 rows, 3 columns), stored as jac[row][col]
    float jac[2][3];
    jac[0][0] = camera.fx * inv_z;
    jac[0][1] = 0.0;
    jac[0][2] = -camera.fx * point_view.x * inv_z2;
    jac[1][0] = 0.0;
    jac[1][1] = camera.fy * inv_z;
    jac[1][2] = -camera.fy * point_view.y * inv_z2;

    // j_cov = J * cov_view (2x3 * 3x3 = 2x3)
    float j_cov[2][3];
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
            j_cov[row][col] = jac[row][0] * cov_view.m[0][col]
                            + jac[row][1] * cov_view.m[1][col]
                            + jac[row][2] * cov_view.m[2][col];
        }
    }

    // 2D covariance = j_cov * J^T (2x3 * 3x2 = 2x2)
    // Only need upper triangle of the symmetric result: cov_a, cov_b, cov_c
    float cov_a = j_cov[0][0] * jac[0][0] + j_cov[0][1] * jac[0][1] + j_cov[0][2] * jac[0][2];
    float cov_b = j_cov[0][0] * jac[1][0] + j_cov[0][1] * jac[1][1] + j_cov[0][2] * jac[1][2];
    float cov_c = j_cov[1][0] * jac[1][0] + j_cov[1][1] * jac[1][1] + j_cov[1][2] * jac[1][2];

    return float3(cov_a + 1e-3, cov_b, cov_c + 1e-3);
}

float2 compute_2d_gaussian_extent(float3 cov) {
    float trace = cov.x + cov.z;
    float det = cov.x * cov.z - cov.y * cov.y;
    float disc = sqrt(max(trace * trace - 4.0 * det, 0.0));

    float lambda1 = 0.5 * (trace + disc);
    float extent = 4.0 * sqrt(max(lambda1, 0.0)); // 4-sigma cutoff
    return float2(extent, extent);
}

float2 compute_snug_tile_extent(float3 cov) {
    // Conservative axis-aligned bbox for the q <= 16 ellipse. Rasterization
    // later drops Gaussian contributions below 0.001, which happens before
    // q = 16, so this still covers all pixels that can contribute.
    const float sigma_cutoff = 4.0f;
    return float2(
        sigma_cutoff * sqrt(max(cov.x, 0.0f)),
        sigma_cutoff * sqrt(max(cov.z, 0.0f))
    );
}

float2 compute_fast_opacity_tile_extent(float3 cov, float opacity, float alpha_cutoff) {
    // Fast-preview mode is allowed to drop contributions whose peak alpha is
    // below the configured cutoff.  For the remaining splats, solve
    // opacity * exp(-0.5q) >= alpha_cutoff for q and convert the axis-aligned
    // covariance diagonal to a tighter pixel extent.
    const float safe_opacity = max(opacity, 1e-6f);
    const float safe_cutoff = clamp(alpha_cutoff, 1e-6f, 0.05f);
    if (safe_opacity <= safe_cutoff) {
        return float2(0.0f, 0.0f);
    }
    const float q_cutoff = max(-2.0f * log(safe_cutoff / safe_opacity), 0.25f);
    const float sigma_cutoff = min(sqrt(q_cutoff), 4.0f);
    return float2(
        sigma_cutoff * sqrt(max(cov.x, 0.0f)),
        sigma_cutoff * sqrt(max(cov.z, 0.0f))
    );
}

kernel void project_splats(
    constant SplatData* splats [[buffer(0)]],
    device ProjectedSplat* projected_splats [[buffer(1)]],
    device atomic_uint* valid_count [[buffer(2)]],
    constant CameraData& camera [[buffer(3)]],
    constant uint& active_splat_count [[buffer(4)]],
    constant TileConfig& tile_config [[buffer(5)]],
    constant uint& use_snug_tile_bounds [[buffer(6)]],
    constant uint& fast_quality [[buffer(7)]],
    constant float& fast_alpha_cutoff [[buffer(8)]],
    constant uint& source_splat_count [[buffer(9)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= active_splat_count || source_splat_count == 0u || active_splat_count == 0u) {
        return;
    }

    const ulong mapped = (ulong(index) * ulong(source_splat_count)) / ulong(active_splat_count);
    const uint source_index = uint(min(mapped, ulong(source_splat_count - 1u)));
    SplatData splat = splats[source_index];

    // World to view transformation
    float3 rel = float3(splat.pos_x, splat.pos_y, splat.pos_z) -
                 float3(camera.pos_x, camera.pos_y, camera.pos_z);
    float3 right = float3(camera.right_x, camera.right_y, camera.right_z);
    float3 up = float3(camera.up_x, camera.up_y, camera.up_z);
    float3 forward = float3(camera.forward_x, camera.forward_y, camera.forward_z);

    float3 view_pos = float3(dot(rel, right), dot(rel, up), dot(rel, forward));

    // Near/far culling
    if (view_pos.z < camera.near_plane || view_pos.z > camera.far_plane) {
        return;
    }

    // Screen space projection
    float inv_z = 1.0 / max(view_pos.z, 1e-5);
    float screen_x = camera.half_w + view_pos.x * camera.fx * inv_z;
    float screen_y = camera.half_h - view_pos.y * camera.fy * inv_z;

    if (!isfinite(screen_x) || !isfinite(screen_y)) {
        return;
    }

    // Broad frustum culling
    const float BROAD_MARGIN = 120.0;
    if (screen_x < -BROAD_MARGIN || screen_x > (camera.half_w * 2.0) + BROAD_MARGIN ||
        screen_y < -BROAD_MARGIN || screen_y > (camera.half_h * 2.0) + BROAD_MARGIN) {
        return;
    }

    // Compute 3D covariance
    float3 scale = float3(splat.scale_x, splat.scale_y, splat.scale_z);
    float4 rotation = float4(splat.rot_w, splat.rot_x, splat.rot_y, splat.rot_z);
    Mat3 cov_3d = compute_3d_covariance(scale, rotation);

    // Project to 2D
    float3 cov_2d = project_covariance_to_2d(cov_3d, camera, view_pos);

    if (cov_2d.x <= 0.0 || cov_2d.z <= 0.0) {
        return;
    }

    // Compute extent
    float2 extent = compute_2d_gaussian_extent(cov_2d);
    if (extent.x < 0.3 || extent.y < 0.3) {
        return;
    }
    float2 tile_extent = use_snug_tile_bounds != 0u
        ? compute_snug_tile_extent(cov_2d)
        : extent;
    float2 render_extent = extent;
    if (fast_quality != 0u) {
        tile_extent = compute_fast_opacity_tile_extent(cov_2d, splat.opacity, fast_alpha_cutoff);
        if (tile_extent.x < 0.3f || tile_extent.y < 0.3f) {
            return;
        }
        render_extent = tile_extent;
    }

    // Compute tile bounds from the same inclusive integer bbox used by the CPU
    // rasterizer/probe telemetry: floor(min edge), ceil(max edge), then clamp to screen.
    float max_pixel_x = float(tile_config.screen_width - 1);
    float max_pixel_y = float(tile_config.screen_height - 1);
    float splat_min_x = min(max(floor(screen_x - tile_extent.x), 0.0f), max_pixel_x);
    float splat_min_y = min(max(floor(screen_y - tile_extent.y), 0.0f), max_pixel_y);
    float splat_max_x = min(max(ceil(screen_x + tile_extent.x), 0.0f), max_pixel_x);
    float splat_max_y = min(max(ceil(screen_y + tile_extent.y), 0.0f), max_pixel_y);

    uint tile_min_x = uint(splat_min_x) / 16;  // TILE_SIZE = 16
    uint tile_min_y = uint(splat_min_y) / 16;
    uint tile_max_x = uint(splat_max_x) / 16;
    uint tile_max_y = uint(splat_max_y) / 16;

    uint packed_tile_min = (tile_min_y << 16) | tile_min_x;
    uint packed_tile_max = (tile_max_y << 16) | tile_max_x;

    // Final screen bounds check
    if (screen_x + extent.x < 0.0 || screen_x - extent.x > (camera.half_w * 2.0) ||
        screen_y + extent.y < 0.0 || screen_y - extent.y > (camera.half_h * 2.0)) {
        return;
    }

    // Write projected splat
    uint output_index = atomic_fetch_add_explicit(valid_count, 1, memory_order_relaxed);

    projected_splats[output_index] = {
        screen_x, screen_y, view_pos.z,
        render_extent.x, render_extent.y,
        cov_2d.x, cov_2d.y, cov_2d.z,
        splat.opacity,
        splat.packed_color,
        source_index,
        packed_tile_min,
        packed_tile_max
    };
}
