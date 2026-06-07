#include <metal_stdlib>
using namespace metal;

struct HalfblockConfig {
    uint framebuffer_width;
    uint framebuffer_height;
    uint term_cols;
    uint term_rows;
    uint supersample;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct HalfblockCell {
    uint top_rgb;
    uint bottom_rgb;
};

inline uint pack_rgb(uint r, uint g, uint b) {
    return (r & 0xFFu) | ((g & 0xFFu) << 8) | ((b & 0xFFu) << 16);
}

inline uint average_region(
    constant uint* framebuffer,
    constant HalfblockConfig& cfg,
    uint x0,
    uint x1,
    uint y0,
    uint y1
) {
    uint sum_r = 0u;
    uint sum_g = 0u;
    uint sum_b = 0u;
    uint count = 0u;

    for (uint y = y0; y < y1; ++y) {
        const uint row_offset = y * cfg.framebuffer_width;
        for (uint x = x0; x < x1; ++x) {
            const uint pixel = framebuffer[row_offset + x];
            sum_r += pixel & 0xFFu;
            sum_g += (pixel >> 8) & 0xFFu;
            sum_b += (pixel >> 16) & 0xFFu;
            count += 1u;
        }
    }

    if (count == 0u) {
        return 0u;
    }

    return pack_rgb(sum_r / count, sum_g / count, sum_b / count);
}

kernel void downsample_halfblock_cells(
    constant uint* framebuffer [[buffer(0)]],
    device HalfblockCell* halfblock_cells [[buffer(1)]],
    constant HalfblockConfig& cfg [[buffer(2)]],
    uint2 cell [[thread_position_in_grid]]
) {
    if (cell.x >= cfg.term_cols || cell.y >= cfg.term_rows || cfg.supersample == 0u) {
        return;
    }

    const uint ss = cfg.supersample;
    const uint x0 = cell.x * ss;
    const uint x1 = min((cell.x + 1u) * ss, cfg.framebuffer_width);
    const uint top_y0 = cell.y * 2u * ss;
    const uint top_y1 = min(cell.y * 2u * ss + ss, cfg.framebuffer_height);
    const uint bot_y0 = min(cell.y * 2u * ss + ss, cfg.framebuffer_height);
    const uint bot_y1 = min((cell.y + 1u) * 2u * ss, cfg.framebuffer_height);

    const uint top_rgb = average_region(framebuffer, cfg, x0, x1, top_y0, top_y1);
    const uint bottom_rgb = average_region(framebuffer, cfg, x0, x1, bot_y0, bot_y1);
    halfblock_cells[cell.y * cfg.term_cols + cell.x] = { top_rgb, bottom_rgb };
}
