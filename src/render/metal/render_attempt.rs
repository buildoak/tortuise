use std::{
    ffi::c_void,
    mem,
    time::{Duration, Instant},
};

use metal::{MTLSize, NSRange};

use crate::camera::Camera;

use super::error::MetalRenderError;
use super::pipeline::{read_shared_u32, set_bytes_u32, write_shared_struct};
use super::sort::{dispatch_1d, div_ceil_u32, MAX_LOCAL_TILE_SORT, RADIX_SORT_BIT_OFFSETS};
use super::sync::commit_and_wait_or_disable_gpu;
use super::types::{
    GpuCameraData, TileConfig, RADIX_BUCKETS, SHADER_TILE_SIZE, THREADS_PER_GROUP_1D, TILE_SIZE,
};
use super::MetalBackend;

const GPU_WAIT_TIMEOUT: Duration = Duration::from_millis(500);
const METAL_STAGE_TIMING_ENV: &str = "TORTUISE_METAL_STAGE_TIMING";
const METAL_LOCAL_TILE_SORT_ENV: &str = "TORTUISE_METAL_LOCAL_TILE_SORT";
const METAL_FAST_UNSORTED_ENV: &str = "TORTUISE_METAL_FAST_UNSORTED";

#[derive(Debug)]
pub(super) struct RenderAttemptResult {
    pub overflow_flag: u32,
    pub total_overlaps: u32,
    pub valid_count: u32,
}

pub(super) fn run_single_render_attempt(
    backend: &mut MetalBackend,
    camera: &Camera,
    screen_width: usize,
    screen_height: usize,
    splat_count: usize,
) -> Result<RenderAttemptResult, MetalRenderError> {
    let stage_timing_enabled = metal_stage_timing_enabled();
    let local_tile_sort_requested = metal_local_tile_sort_enabled();
    let fast_unsorted_enabled = metal_fast_unsorted_enabled();
    let screen_width_u32 = u32::try_from(screen_width)?;
    let screen_height_u32 = u32::try_from(screen_height)?;
    let tile_count_x = div_ceil_u32(screen_width_u32, TILE_SIZE).max(1);
    let tile_count_y = div_ceil_u32(screen_height_u32, TILE_SIZE).max(1);
    let num_tiles_u64 = u64::from(tile_count_x) * u64::from(tile_count_y);
    let num_tiles = usize::try_from(num_tiles_u64)?;

    let sort_capacity_u32 = u32::try_from(backend.sort_capacity)?;
    backend.ensure_block_sums_capacity_for_count(num_tiles as u32)?;

    let (fx, fy) = camera.focal_lengths(screen_width, screen_height);
    let gpu_camera = GpuCameraData {
        pos_x: camera.position.x,
        pos_y: camera.position.y,
        pos_z: camera.position.z,
        right_x: camera.right.x,
        right_y: camera.right.y,
        right_z: camera.right.z,
        up_x: camera.up.x,
        up_y: camera.up.y,
        up_z: camera.up.z,
        forward_x: camera.forward.x,
        forward_y: camera.forward.y,
        forward_z: camera.forward.z,
        fx,
        fy,
        half_w: screen_width as f32 * 0.5,
        half_h: screen_height as f32 * 0.5,
        near_plane: camera.near,
        far_plane: camera.far,
    };

    let tile_config = TileConfig {
        tile_count_x,
        tile_count_y,
        screen_width: screen_width_u32,
        screen_height: screen_height_u32,
    };

    write_shared_struct(&backend.camera_buffer, &gpu_camera);
    write_shared_struct(&backend.tile_config_buffer, &tile_config);

    let tile_bytes = super::buffers::bytes_for_u32_elems(num_tiles)? as u64;
    let tile_offsets_bytes = super::buffers::bytes_for_u32_elems(num_tiles + 1)? as u64;
    let splat_count_u32 = u32::try_from(splat_count)?;
    let framebuffer_pixels = screen_width
        .checked_mul(screen_height)
        .ok_or_else(|| MetalRenderError::Other("framebuffer pixel count overflow".to_string()))?;
    let framebuffer_clear_bytes = framebuffer_pixels
        .checked_mul(mem::size_of::<u32>())
        .ok_or_else(|| MetalRenderError::Other("framebuffer clear size overflow".to_string()))?
        as u64;

    let stage_a_encode_started = stage_timing_enabled.then(Instant::now);
    let stage_a = backend.command_queue.new_command_buffer();

    let blit = stage_a.new_blit_command_encoder();
    blit.fill_buffer(
        &backend.framebuffer,
        NSRange::new(0, framebuffer_clear_bytes),
        0,
    );
    blit.fill_buffer(&backend.tile_counts, NSRange::new(0, tile_bytes), 0);
    blit.fill_buffer(
        &backend.valid_count_buffer,
        NSRange::new(0, mem::size_of::<u32>() as u64),
        0,
    );
    blit.fill_buffer(
        &backend.total_overlaps_buffer,
        NSRange::new(0, mem::size_of::<u32>() as u64),
        0,
    );
    blit.fill_buffer(
        &backend.overflow_flag_buffer,
        NSRange::new(0, mem::size_of::<u32>() as u64),
        0,
    );
    blit.end_encoding();

    let encoder = stage_a.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&backend.project_splats_pipeline);
    encoder.set_buffer(0, Some(&backend.splat_buffer), 0);
    encoder.set_buffer(1, Some(&backend.projected_buffer), 0);
    encoder.set_buffer(2, Some(&backend.valid_count_buffer), 0);
    encoder.set_buffer(3, Some(&backend.camera_buffer), 0);
    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &splat_count_u32 as *const _ as *const c_void,
    );
    encoder.set_buffer(5, Some(&backend.tile_config_buffer), 0);
    dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
    encoder.end_encoding();

    let encoder = stage_a.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&backend.count_tile_overlaps_pipeline);
    encoder.set_buffer(0, Some(&backend.projected_buffer), 0);
    encoder.set_buffer(1, Some(&backend.tile_counts), 0);
    encoder.set_buffer(2, Some(&backend.total_overlaps_buffer), 0);
    encoder.set_buffer(3, Some(&backend.valid_count_buffer), 0);
    encoder.set_buffer(4, Some(&backend.tile_config_buffer), 0);
    dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
    encoder.end_encoding();

    let blit = stage_a.new_blit_command_encoder();
    blit.copy_from_buffer(
        &backend.tile_counts,
        0,
        &backend.tile_offsets,
        0,
        tile_bytes,
    );
    blit.copy_from_buffer(
        &backend.total_overlaps_buffer,
        0,
        &backend.tile_offsets,
        tile_bytes,
        mem::size_of::<u32>() as u64,
    );
    blit.fill_buffer(&backend.tile_counters, NSRange::new(0, tile_bytes), 0);
    blit.end_encoding();

    backend.encode_prefix_scan_in_place(stage_a, &backend.tile_offsets, 0, num_tiles as u32)?;
    let should_read_tile_ranges = stage_timing_enabled || local_tile_sort_requested;
    if should_read_tile_ranges {
        let blit = stage_a.new_blit_command_encoder();
        blit.copy_from_buffer(
            &backend.tile_offsets,
            0,
            &backend.tile_offsets_readback,
            0,
            tile_offsets_bytes,
        );
        blit.end_encoding();
    }
    let stage_a_encode_ms = stage_a_encode_started.map(|started| duration_ms(started.elapsed()));
    let stage_a_wait_started = stage_timing_enabled.then(Instant::now);
    let stage_a_result = commit_and_wait_or_disable_gpu(
        stage_a,
        "project_count_scan",
        GPU_WAIT_TIMEOUT,
        &mut backend.gpu_disabled,
    );
    if stage_timing_enabled {
        let stage_a_wait_ms =
            stage_a_wait_started.map_or(0.0, |started| duration_ms(started.elapsed()));
        eprintln!(
            concat!(
                "tortuise_metal_stage_timing ",
                "{{",
                "\"stage\":\"project_count_scan\",",
                "\"ok\":{},",
                "\"encode_ms\":{:.6},",
                "\"wait_ms\":{:.6},",
                "\"splat_count\":{},",
                "\"num_tiles\":{},",
                "\"tile_count_x\":{},",
                "\"tile_count_y\":{}",
                "}}"
            ),
            stage_a_result.is_ok(),
            stage_a_encode_ms.unwrap_or(0.0),
            stage_a_wait_ms,
            splat_count,
            num_tiles,
            tile_count_x,
            tile_count_y
        );
    }
    stage_a_result?;

    let total_overlaps = read_shared_u32(&backend.total_overlaps_buffer);
    let valid_count = read_shared_u32(&backend.valid_count_buffer);
    if total_overlaps > sort_capacity_u32 {
        return Ok(RenderAttemptResult {
            overflow_flag: 1,
            total_overlaps,
            valid_count,
        });
    }

    let max_tile_range = if should_read_tile_ranges {
        read_max_tile_range(&backend.tile_offsets_readback, num_tiles)?
    } else {
        0
    };
    let use_local_tile_sort = local_tile_sort_requested
        && !fast_unsorted_enabled
        && max_tile_range <= MAX_LOCAL_TILE_SORT;

    let dispatch_overlaps = total_overlaps;
    let mut sort_num_blocks = 0u32;
    let mut histogram_count = 0u32;

    if dispatch_overlaps > 0 && !fast_unsorted_enabled && !use_local_tile_sort {
        sort_num_blocks = div_ceil_u32(dispatch_overlaps, THREADS_PER_GROUP_1D);
        histogram_count = sort_num_blocks
            .checked_mul(RADIX_BUCKETS)
            .ok_or_else(|| MetalRenderError::Other("histogram count overflow".to_string()))?;
        backend.ensure_histogram_capacity(histogram_count as usize)?;
        backend.ensure_block_sums_capacity_for_count(histogram_count)?;
    }

    let stage_b_encode_started = stage_timing_enabled.then(Instant::now);
    let stage_b = backend.command_queue.new_command_buffer();
    let mut keys_in_a = true;
    let mut radix_passes = 0usize;
    let mut local_tile_sort_used = false;

    let encoder = stage_b.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&backend.emit_tile_keys_pipeline);
    encoder.set_buffer(0, Some(&backend.projected_buffer), 0);
    encoder.set_buffer(1, Some(&backend.tile_offsets), 0);
    encoder.set_buffer(2, Some(&backend.tile_counters), 0);
    encoder.set_buffer(3, Some(&backend.sort_keys_a), 0);
    encoder.set_buffer(4, Some(&backend.sort_values_a), 0);
    encoder.set_buffer(5, Some(&backend.valid_count_buffer), 0);
    encoder.set_buffer(6, Some(&backend.tile_config_buffer), 0);
    encoder.set_buffer(7, Some(&backend.overflow_flag_buffer), 0);
    set_bytes_u32(encoder, 8, sort_capacity_u32);
    dispatch_1d(encoder, splat_count_u32, THREADS_PER_GROUP_1D);
    encoder.end_encoding();

    if dispatch_overlaps > 0 {
        let (sorted_keys, sorted_values) = if fast_unsorted_enabled {
            (&backend.sort_keys_a, &backend.sort_values_a)
        } else if use_local_tile_sort {
            backend.encode_local_tile_sort(stage_b, num_tiles as u32);
            local_tile_sort_used = true;
            (&backend.sort_keys_b, &backend.sort_values_b)
        } else {
            backend.run_radix_sort_passes(stage_b, dispatch_overlaps, &mut keys_in_a)?;
            radix_passes = RADIX_SORT_BIT_OFFSETS.len();
            let (keys, values) = if keys_in_a {
                (&backend.sort_keys_a, &backend.sort_values_a)
            } else {
                (&backend.sort_keys_b, &backend.sort_values_b)
            };
            (keys, values)
        };

        let encoder = stage_b.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&backend.rasterize_tiles_pipeline);
        encoder.set_buffer(0, Some(&backend.projected_buffer), 0);
        encoder.set_buffer(1, Some(sorted_keys), 0);
        encoder.set_buffer(2, Some(sorted_values), 0);
        encoder.set_buffer(3, Some(&backend.tile_offsets), 0);
        encoder.set_buffer(4, Some(&backend.framebuffer), 0);
        encoder.set_buffer(5, Some(&backend.tile_config_buffer), 0);
        set_bytes_u32(encoder, 6, dispatch_overlaps);
        debug_assert_eq!(TILE_SIZE, SHADER_TILE_SIZE);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(tile_count_x), u64::from(tile_count_y), 1),
            MTLSize::new(u64::from(TILE_SIZE), u64::from(TILE_SIZE), 1),
        );
        encoder.end_encoding();
    }

    let stage_b_encode_ms = stage_b_encode_started.map(|started| duration_ms(started.elapsed()));
    let stage_b_wait_started = stage_timing_enabled.then(Instant::now);
    let stage_b_result = commit_and_wait_or_disable_gpu(
        stage_b,
        "sort_rasterize",
        GPU_WAIT_TIMEOUT,
        &mut backend.gpu_disabled,
    );
    if stage_timing_enabled {
        let stage_b_wait_ms =
            stage_b_wait_started.map_or(0.0, |started| duration_ms(started.elapsed()));
        eprintln!(
            concat!(
                "tortuise_metal_stage_timing ",
                "{{",
                "\"stage\":\"sort_rasterize\",",
                "\"ok\":{},",
                "\"encode_ms\":{:.6},",
                "\"wait_ms\":{:.6},",
                "\"splat_count\":{},",
                "\"dispatch_overlaps\":{},",
                "\"sort_capacity\":{},",
                "\"sort_num_blocks\":{},",
                "\"radix_histogram_count\":{},",
                "\"radix_passes\":{},",
                "\"local_tile_sort\":{},",
                "\"fast_unsorted\":{},",
                "\"max_tile_range\":{},",
                "\"max_local_tile_sort\":{}",
                "}}"
            ),
            stage_b_result.is_ok(),
            stage_b_encode_ms.unwrap_or(0.0),
            stage_b_wait_ms,
            splat_count,
            dispatch_overlaps,
            sort_capacity_u32,
            sort_num_blocks,
            histogram_count,
            radix_passes,
            local_tile_sort_used,
            fast_unsorted_enabled,
            max_tile_range,
            MAX_LOCAL_TILE_SORT
        );
    }
    stage_b_result?;

    let overflow_flag = read_shared_u32(&backend.overflow_flag_buffer);

    Ok(RenderAttemptResult {
        overflow_flag,
        total_overlaps,
        valid_count,
    })
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn metal_stage_timing_enabled() -> bool {
    std::env::var_os(METAL_STAGE_TIMING_ENV).is_some()
}

fn metal_local_tile_sort_enabled() -> bool {
    env_flag_enabled(METAL_LOCAL_TILE_SORT_ENV)
}

fn metal_fast_unsorted_enabled() -> bool {
    env_flag_enabled(METAL_FAST_UNSORTED_ENV)
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn read_max_tile_range(
    tile_offsets_readback: &metal::Buffer,
    num_tiles: usize,
) -> Result<u32, MetalRenderError> {
    if num_tiles == 0 {
        return Ok(0);
    }

    let offsets = unsafe {
        std::slice::from_raw_parts(
            tile_offsets_readback.contents() as *const u32,
            num_tiles + 1,
        )
    };
    let mut max_range = 0u32;
    for window in offsets.windows(2) {
        let start = window[0];
        let end = window[1];
        let count = end.checked_sub(start).ok_or_else(|| {
            MetalRenderError::Other("Metal tile offsets were not monotonic after scan".to_string())
        })?;
        max_range = max_range.max(count);
    }
    Ok(max_range)
}
