#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16
#define MAX_LOCAL_TILE_SORT 2048u

struct ProjectedSplat {
    float screen_x, screen_y, depth;
    float radius_x, radius_y;
    float cov_a, cov_b, cov_c;
    float opacity;
    uint packed_color;
    uint original_index;
    uint tile_min;  // packed: (tile_min_y << 16) | tile_min_x
    uint tile_max;  // packed: (tile_max_y << 16) | tile_max_x
};

struct TileConfig {
    uint tile_count_x;
    uint tile_count_y;
    uint screen_width;
    uint screen_height;
};

kernel void prepare_dispatch_1d_indirect_args(
    device uint* dispatch_args [[buffer(0)]],
    constant uint& item_count [[buffer(1)]],
    constant uint& threads_per_group [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0u) {
        return;
    }

    const uint groups = max((item_count + threads_per_group - 1u) / threads_per_group, 1u);
    dispatch_args[0] = groups;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

inline uint2 unpack_tile(uint packed) {
    return uint2(packed & 0xFFFFu, packed >> 16);
}

inline uint float_to_sortable_uint(float f) {
    uint v = as_type<uint>(f);
    return (v & 0x80000000u) ? ~v : (v | 0x80000000u);
}

kernel void count_tile_overlaps(
    constant ProjectedSplat* projected [[buffer(0)]],
    device atomic_uint* tile_counts [[buffer(1)]],
    device atomic_uint* total_overlaps [[buffer(2)]],
    constant uint& valid_count [[buffer(3)]],
    constant TileConfig& tile_config [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= valid_count) {
        return;
    }

    const ProjectedSplat splat = projected[index];
    const uint2 tile_min = unpack_tile(splat.tile_min);
    const uint2 tile_max = unpack_tile(splat.tile_max);

    // Guard against unsigned underflow when tile_min > tile_max (splat
    // bounding box collapsed to zero after clamping).
    if (tile_min.x > tile_max.x || tile_min.y > tile_max.y) {
        return;
    }

    // Number of tiles covered by this splat's AABB in tile space.
    const uint overlap_count = (tile_max.x - tile_min.x + 1u) * (tile_max.y - tile_min.y + 1u);
    atomic_fetch_add_explicit(total_overlaps, overlap_count, memory_order_relaxed);

    // Increment per-tile overlap counts for prefix scan / allocation.
    for (uint ty = tile_min.y; ty <= tile_max.y; ++ty) {
        const uint row_offset = ty * tile_config.tile_count_x;
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx) {
            const uint tile_id = row_offset + tx;
            atomic_fetch_add_explicit(&tile_counts[tile_id], 1u, memory_order_relaxed);
        }
    }
}

/// Clamp the total_overlaps counter to sort_capacity so that the radix sort
/// never addresses more elements than were actually emitted into the sort
/// buffers.  Dispatched as a single thread between emit_tile_keys and the
/// radix sort passes.
kernel void clamp_total_overlaps(
    device uint* total_overlaps [[buffer(0)]],
    constant uint& sort_capacity [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) return;
    if (total_overlaps[0] > sort_capacity) {
        total_overlaps[0] = sort_capacity;
    }
}

kernel void emit_tile_keys(
    constant ProjectedSplat* projected [[buffer(0)]],
    constant uint* tile_offsets [[buffer(1)]],
    device atomic_uint* tile_counters [[buffer(2)]],
    device uint64_t* sort_keys [[buffer(3)]],
    device uint* sort_values [[buffer(4)]],
    constant uint& valid_count [[buffer(5)]],
    constant TileConfig& tile_config [[buffer(6)]],
    device atomic_uint* overflow_flag [[buffer(7)]],
    constant uint& sort_capacity [[buffer(8)]],
    constant uint& key_mode [[buffer(9)]],
    constant uint& approx_depth_bits [[buffer(10)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= valid_count) {
        return;
    }

    const ProjectedSplat splat = projected[index];
    const uint2 tile_min = unpack_tile(splat.tile_min);
    const uint2 tile_max = unpack_tile(splat.tile_max);

    if (tile_min.x > tile_max.x || tile_min.y > tile_max.y) {
        return;
    }

    // Sort key layout: 10-bit tile_id | 32-bit sortable depth | 22-bit original_index
    //
    // 10 bits supports up to 1023 tiles (a 500x160 terminal with 16x16 tiles
    // = ~320 tiles, plenty of headroom).  The Rust render path rejects larger
    // tile grids before this kernel is dispatched.  Full sortable f32 depth
    // bits match CPU total_cmp ordering for finite visible depths, and 22
    // original_index bits give a deterministic tiebreaker for loaded scenes
    // below 4,194,304 source splats.
    //
    // The atomic_fetch_add for slot assignment in emit_tile_keys remains
    // non-deterministic, but this only affects the input order fed to the
    // radix sort.  Since the sort is keyed on (tile, depth, original_index),
    // the final sorted order is determined by the key, not the slot.
    const uint depth_key = float_to_sortable_uint(splat.depth);
    const uint index_key = splat.original_index & 0x3FFFFFu;

    // Emit one key/value pair for each tile this splat overlaps.
    for (uint ty = tile_min.y; ty <= tile_max.y; ++ty) {
        const uint row_offset = ty * tile_config.tile_count_x;
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx) {
            const uint tile_id = row_offset + tx;
            const uint local_offset =
                atomic_fetch_add_explicit(&tile_counters[tile_id], 1u, memory_order_relaxed);
            const uint slot = tile_offsets[tile_id] + local_offset;
            if (slot >= sort_capacity) {
                atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
                return;
            }

            if (key_mode == 1u) {
                const uint safe_depth_bits = clamp(approx_depth_bits, 1u, 22u);
                const uint depth_bin = depth_key >> (32u - safe_depth_bits);
                sort_keys[slot] =
                    (uint64_t(tile_id) << safe_depth_bits) |
                    uint64_t(depth_bin);
            } else {
                sort_keys[slot] =
                    (uint64_t(tile_id) << 54) |
                    (uint64_t(depth_key) << 22) |
                    uint64_t(index_key);
            }
            sort_values[slot] = index;
        }
    }
}

kernel void local_tile_sort(
    constant uint* tile_offsets [[buffer(0)]],
    constant uint64_t* sort_keys_in [[buffer(1)]],
    constant uint* sort_values_in [[buffer(2)]],
    device uint64_t* sort_keys_out [[buffer(3)]],
    device uint* sort_values_out [[buffer(4)]],
    device atomic_uint* overflow_flag [[buffer(5)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    const uint start = tile_offsets[tile_id];
    const uint end = tile_offsets[tile_id + 1u];
    const uint count = end - start;
    if (count > MAX_LOCAL_TILE_SORT) {
        if (ltid == 0u) {
            atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
        }
        return;
    }
    if (count == 0u) {
        return;
    }

    uint sort_len = 1u;
    while (sort_len < count) {
        sort_len <<= 1u;
    }

    threadgroup uint64_t keys[MAX_LOCAL_TILE_SORT];
    threadgroup uint values[MAX_LOCAL_TILE_SORT];

    for (uint i = ltid; i < sort_len; i += threads_per_group) {
        if (i < count) {
            keys[i] = sort_keys_in[start + i];
            values[i] = sort_values_in[start + i];
        } else {
            keys[i] = ~uint64_t(0);
            values[i] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= sort_len; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint i = ltid; i < sort_len; i += threads_per_group) {
                const uint ixj = i ^ j;
                if (ixj > i) {
                    const bool ascending = (i & k) == 0u;
                    const uint64_t key_i = keys[i];
                    const uint64_t key_j = keys[ixj];
                    const bool swap_pair =
                        (ascending && key_i > key_j) ||
                        (!ascending && key_i < key_j);

                    if (swap_pair) {
                        const uint value_i = values[i];
                        keys[i] = key_j;
                        values[i] = values[ixj];
                        keys[ixj] = key_i;
                        values[ixj] = value_i;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = ltid; i < count; i += threads_per_group) {
        sort_keys_out[start + i] = keys[i];
        sort_values_out[start + i] = values[i];
    }
}
