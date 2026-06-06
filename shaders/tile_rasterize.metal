#include <metal_stdlib>
using namespace metal;

struct ProjectedSplat {
    float screen_x, screen_y, depth;
    float radius_x, radius_y;
    float cov_a, cov_b, cov_c;
    float opacity;
    uint packed_color;
    uint original_index;
    uint tile_min;
    uint tile_max;
};

struct TileConfig {
    uint tile_count_x;
    uint tile_count_y;
    uint screen_width;
    uint screen_height;
};

#define TILE_SIZE 16
#define BATCH_SIZE 32
#define SATURATION_EPSILON 0.999f
#define MIN_GAUSSIAN_CONTRIB 0.001f

kernel void rasterize_tiles(
    constant ProjectedSplat* projected [[buffer(0)]],
    constant uint* sort_keys [[buffer(1)]],
    constant uint* sort_values [[buffer(2)]],
    constant uint* tile_ranges [[buffer(3)]],
    device uint* framebuffer [[buffer(4)]],
    constant TileConfig& tile_config [[buffer(5)]],
    constant uint& sort_capacity [[buffer(6)]],
    uint2 threadgroup_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]],
    uint linear_tid [[thread_index_in_threadgroup]])
{
    (void)sort_keys; // Kept for parity with pipeline bindings; sort_values drives splat lookup.

    const uint tile_x = threadgroup_pos.x;
    const uint tile_y = threadgroup_pos.y;

    // Dispatch is expected to match tile grid exactly, but guard anyway.
    if (tile_x >= tile_config.tile_count_x || tile_y >= tile_config.tile_count_y) {
        return;
    }

    const uint tile_id = tile_y * tile_config.tile_count_x + tile_x;
    const uint unclamped_start = tile_ranges[tile_id];
    const uint unclamped_end = tile_ranges[tile_id + 1];
    const uint range_start = min(unclamped_start, sort_capacity);
    const uint range_end = min(unclamped_end, sort_capacity);
    if (range_end < range_start) {
        return;
    }

    const uint pixel_x = tile_x * TILE_SIZE + local_pos.x;
    const uint pixel_y = tile_y * TILE_SIZE + local_pos.y;
    const bool pixel_in_bounds = (pixel_x < tile_config.screen_width) &&
                                 (pixel_y < tile_config.screen_height);

    // Front-to-back compositing state:
    // color accumulates premultiplied contributions and transmittance tracks remaining light.
    float color_r = 0.0f;
    float color_g = 0.0f;
    float color_b = 0.0f;
    float transmittance = 1.0f;
    bool done = false;

    // Batch splat fields into threadgroup memory so 256 pixels in the tile reuse the same loads.
    threadgroup float shared_screen_x[BATCH_SIZE];
    threadgroup float shared_screen_y[BATCH_SIZE];
    threadgroup float shared_radius_x[BATCH_SIZE];
    threadgroup float shared_radius_y[BATCH_SIZE];
    threadgroup float shared_cov_a[BATCH_SIZE];
    threadgroup float shared_cov_b[BATCH_SIZE];
    threadgroup float shared_cov_c[BATCH_SIZE];
    threadgroup float shared_opacity[BATCH_SIZE];
    threadgroup uint shared_packed_color[BATCH_SIZE];
    threadgroup uint shared_bbox_min_x[BATCH_SIZE];
    threadgroup uint shared_bbox_min_y[BATCH_SIZE];
    threadgroup uint shared_bbox_max_x[BATCH_SIZE];
    threadgroup uint shared_bbox_max_y[BATCH_SIZE];

    const float pixel_center_x = float(pixel_x) + 0.5f;
    const float pixel_center_y = float(pixel_y) + 0.5f;
    const float saturation_transmittance_threshold = 1.0f - SATURATION_EPSILON;
    const float max_pixel_x = float(tile_config.screen_width - 1);
    const float max_pixel_y = float(tile_config.screen_height - 1);

    const uint total_splats = range_end - range_start;
    for (uint batch_offset = 0; batch_offset < total_splats; batch_offset += BATCH_SIZE) {
        const uint batch_count = min(uint(BATCH_SIZE), total_splats - batch_offset);

        // Cooperative load: first BATCH_SIZE threads pull one splat each into shared memory.
        if (linear_tid < batch_count) {
            const uint sorted_index = range_start + batch_offset + linear_tid;
            const uint splat_index = sort_values[sorted_index];
            const ProjectedSplat splat = projected[splat_index];

            shared_screen_x[linear_tid] = splat.screen_x;
            shared_screen_y[linear_tid] = splat.screen_y;
            shared_radius_x[linear_tid] = splat.radius_x;
            shared_radius_y[linear_tid] = splat.radius_y;
            shared_cov_a[linear_tid] = splat.cov_a;
            shared_cov_b[linear_tid] = splat.cov_b;
            shared_cov_c[linear_tid] = splat.cov_c;
            shared_opacity[linear_tid] = splat.opacity;
            shared_packed_color[linear_tid] = splat.packed_color;
            shared_bbox_min_x[linear_tid] =
                uint(min(max(floor(splat.screen_x - splat.radius_x), 0.0f), max_pixel_x));
            shared_bbox_min_y[linear_tid] =
                uint(min(max(floor(splat.screen_y - splat.radius_y), 0.0f), max_pixel_y));
            shared_bbox_max_x[linear_tid] =
                uint(min(max(ceil(splat.screen_x + splat.radius_x), 0.0f), max_pixel_x));
            shared_bbox_max_y[linear_tid] =
                uint(min(max(ceil(splat.screen_y + splat.radius_y), 0.0f), max_pixel_y));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process the loaded batch for this pixel, preserving sorted front-to-back order.
        if (!done && pixel_in_bounds) {
            for (uint i = 0; i < batch_count; ++i) {
                // Match the CPU rasterizer's inclusive integer bbox. This admits edge pixels
                // selected by ceil(max) while still excluding pixels the CPU never visits.
                if (pixel_x < shared_bbox_min_x[i] || pixel_x > shared_bbox_max_x[i] ||
                    pixel_y < shared_bbox_min_y[i] || pixel_y > shared_bbox_max_y[i]) {
                    continue;
                }

                const float dx = pixel_center_x - shared_screen_x[i];
                const float dy = pixel_center_y - shared_screen_y[i];

                // Invert the 2x2 covariance matrix:
                // [a b; b c]^-1 = (1/det) * [ c -b; -b a ], det = a*c - b*b
                const float cov_a = shared_cov_a[i];
                const float cov_b = shared_cov_b[i];
                const float cov_c = shared_cov_c[i];
                const float det = cov_a * cov_c - cov_b * cov_b;
                if (fabs(det) < 1e-8f) {
                    continue;
                }

                const float inv_det = 1.0f / det;
                const float inv_a = cov_c * inv_det;
                const float inv_b = -cov_b * inv_det;
                const float inv_c = cov_a * inv_det;

                // Quadratic form for Gaussian exponent: q = [dx dy] * inv_cov * [dx dy]^T.
                const float q = dx * dx * inv_a + 2.0f * dx * dy * inv_b + dy * dy * inv_c;
                if (q > 32.0f) {
                    continue;
                }

                const float g = exp(-0.5f * q);
                if (g < MIN_GAUSSIAN_CONTRIB) {
                    continue;
                }

                const float alpha = shared_opacity[i] * g;
                const float weight = alpha * transmittance;
                if (weight < 1e-4f) {
                    continue;
                }

                const uint packed = shared_packed_color[i];
                const float splat_r = float(packed & 0xFFu);
                const float splat_g = float((packed >> 8) & 0xFFu);
                const float splat_b = float((packed >> 16) & 0xFFu);

                // CPU-equivalent front-to-back accumulation. The CPU framebuffer stores
                // u8 channels and truncates after each splat contribution, not just at
                // final framebuffer writeback.
                color_r = floor(clamp(color_r + splat_r * weight, 0.0f, 255.0f));
                color_g = floor(clamp(color_g + splat_g * weight, 0.0f, 255.0f));
                color_b = floor(clamp(color_b + splat_b * weight, 0.0f, 255.0f));

                transmittance *= (1.0f - alpha);
                transmittance = max(transmittance, 0.0f);

                // Pixel-level early-out once effectively saturated.
                if (transmittance < saturation_transmittance_threshold) {
                    done = true;
                    break;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // SIMD-level early-out signal for this pixel lane; keep looping for barriers,
        // but skip remaining per-pixel work once the SIMDgroup is fully saturated.
        // Call simd_all() unconditionally so every lane in the SIMDgroup participates.
        const bool simd_group_done = simd_all(done || !pixel_in_bounds || (transmittance < 0.001f));
        if (simd_group_done) {
            done = true;
        }
    }

    if (!pixel_in_bounds) {
        return;
    }

    const uint pixel_r = uint(clamp(color_r, 0.0f, 255.0f));
    const uint pixel_g = uint(clamp(color_g, 0.0f, 255.0f));
    const uint pixel_b = uint(clamp(color_b, 0.0f, 255.0f));

    const float accumulated_alpha = clamp(1.0f - transmittance, 0.0f, 1.0f);
    const uint alpha_u8 = uint(accumulated_alpha * 255.0f);

    const uint packed_out =
        pixel_r |
        (pixel_g << 8) |
        (pixel_b << 16) |
        (alpha_u8 << 24);

    framebuffer[pixel_y * tile_config.screen_width + pixel_x] = packed_out;
}
