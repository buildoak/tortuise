#include <metal_stdlib>
using namespace metal;

// 8-bit radix over uint64_t keys => 256 buckets, 8 LSD passes.
static constant uint kRadixBuckets = 256u;
static constant uint kBlockSize = 256u;
static constant uint kRadixMask = 0xFFu;

kernel void radix_sort_histogram(
    constant uint64_t* keys [[buffer(0)]],
    device uint* histograms [[buffer(1)]],
    device const uint& num_elements [[buffer(2)]],
    constant uint& bit_offset [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint block_id [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]]
) {
    uint num_blocks = (num_elements + kBlockSize - 1u) / kBlockSize;

    // Guard: dispatch may have more threadgroups than actual num_blocks
    // (when estimated overlaps > actual overlaps). Extra blocks must not
    // touch the histogram to avoid clobbering column-major data.
    if (block_id >= num_blocks) {
        return;
    }

    // Per-block histogram in threadgroup memory (256 buckets).
    threadgroup atomic_uint local_histogram[kRadixBuckets];

    // One thread clears one bucket.
    atomic_store_explicit(&local_histogram[ltid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One thread processes one key, contributes to its radix bucket.
    if (gid < num_elements) {
        uint64_t key = keys[gid];
        uint digit = uint((key >> bit_offset) & uint64_t(kRadixMask));
        atomic_fetch_add_explicit(&local_histogram[digit], 1u, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Column-major layout:
    // histograms[bucket * num_blocks + block_id] = count
    uint out_index = ltid * num_blocks + block_id;
    uint count = atomic_load_explicit(&local_histogram[ltid], memory_order_relaxed);
    histograms[out_index] = count;
}

kernel void radix_sort_scatter(
    constant uint64_t* keys_in [[buffer(0)]],
    constant uint* values_in [[buffer(1)]],
    device uint64_t* keys_out [[buffer(2)]],
    device uint* values_out [[buffer(3)]],
    constant uint* histograms [[buffer(4)]],
    device const uint& num_elements [[buffer(5)]],
    constant uint& bit_offset [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint block_id [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]]
) {
    uint num_blocks = (num_elements + kBlockSize - 1u) / kBlockSize;

    if (block_id >= num_blocks) {
        return;
    }

    if (gid >= num_elements) {
        return;
    }

    uint64_t key = keys_in[gid];
    uint value = values_in[gid];
    uint digit = uint((key >> bit_offset) & uint64_t(kRadixMask));

    // LSD radix sort must be stable on every pass.  A per-bucket atomic rank
    // is not stable because atomic arrival order is not guaranteed to match
    // input order, which can scramble tile ranges after later passes.  The
    // bounded 256-element block lets us compute the deterministic in-block rank
    // directly from prior input positions.
    const uint block_start = block_id * kBlockSize;
    uint local_rank = 0u;
    for (uint i = block_start; i < gid; ++i) {
        uint prior_digit = uint((keys_in[i] >> bit_offset) & uint64_t(kRadixMask));
        if (prior_digit == digit) {
            local_rank += 1u;
        }
    }

    uint base_offset = histograms[digit * num_blocks + block_id];
    uint out_index = base_offset + local_rank;

    keys_out[out_index] = key;
    values_out[out_index] = value;
}
