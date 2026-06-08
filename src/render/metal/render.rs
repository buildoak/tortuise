use crate::camera::Camera;
use objc::rc::autoreleasepool;

use super::error::MetalRenderError;
use super::render_attempt::run_single_render_attempt;
use super::sort::div_ceil_u32;
use super::types::TILE_SIZE;
use super::MetalBackend;
use crate::render::MetalLodMode;

const MAX_SORT_KEY_ORIGINAL_INDEX: usize = (1 << 22) - 1;
const MAX_SORT_KEY_TILE_COUNT: u64 = (1 << 16) - 1;
const WARMED_OVERLAP_HEADROOM_NUM: usize = 5;
const WARMED_OVERLAP_HEADROOM_DEN: usize = 4;
const COLD_START_OVERLAP_FACTOR: usize = 8;
const SMALL_SCENE_SPLAT_THRESHOLD: usize = 1024;
const SMALL_SCENE_OVERLAP_FLOOR_FACTOR: usize = 16;
const MAX_OVERFLOW_RETRIES: u32 = 2;

pub(super) fn estimate_overlaps_for_attempt(
    splat_count: usize,
    previous_total_overlaps: u32,
    previous_num_tiles: usize,
    num_tiles: usize,
) -> usize {
    if previous_total_overlaps == 0 {
        let cold_start_factor = if splat_count < SMALL_SCENE_SPLAT_THRESHOLD {
            SMALL_SCENE_OVERLAP_FLOOR_FACTOR
        } else {
            COLD_START_OVERLAP_FACTOR
        };
        return splat_count.saturating_mul(cold_start_factor).max(1);
    }

    let previous_overlaps = previous_total_overlaps as usize;
    let tile_scaled_previous = if previous_num_tiles > 0 {
        previous_overlaps
            .saturating_mul(num_tiles)
            .div_ceil(previous_num_tiles)
    } else {
        previous_overlaps
    };

    let warmed_estimate = tile_scaled_previous
        .saturating_mul(WARMED_OVERLAP_HEADROOM_NUM)
        .div_ceil(WARMED_OVERLAP_HEADROOM_DEN);

    if splat_count < SMALL_SCENE_SPLAT_THRESHOLD {
        warmed_estimate.max(splat_count.saturating_mul(SMALL_SCENE_OVERLAP_FLOOR_FACTOR))
    } else {
        warmed_estimate
    }
    .max(1)
}

fn retry_overlap_target(total_overlaps: u32, attempt_sort_count: usize) -> usize {
    let observed_with_headroom = (total_overlaps as usize)
        .saturating_mul(WARMED_OVERLAP_HEADROOM_NUM)
        .div_ceil(WARMED_OVERLAP_HEADROOM_DEN);
    observed_with_headroom
        .max(attempt_sort_count.saturating_mul(2))
        .max(1)
}

impl MetalBackend {
    pub fn render(
        &mut self,
        camera: &Camera,
        screen_width: usize,
        screen_height: usize,
        active_splat_count: usize,
        source_splat_count: usize,
        lod_mode: MetalLodMode,
        lod_requested_splat_count: Option<usize>,
    ) -> Result<(), MetalRenderError> {
        autoreleasepool(|| {
            if self.gpu_disabled {
                return Err(MetalRenderError::GpuDisabled);
            }

            if !self.splats_uploaded {
                return Err("No splats uploaded to Metal backend".into());
            }

            if screen_width == 0 || screen_height == 0 {
                self.last_render_width = screen_width;
                self.last_render_height = screen_height;
                return Ok(());
            }

            if source_splat_count > self.max_splats {
                return Err("Too many splats for GPU buffers".into());
            }
            if active_splat_count > source_splat_count {
                return Err("Metal active splat count exceeds source splat count".into());
            }
            if source_splat_count > MAX_SORT_KEY_ORIGINAL_INDEX + 1 {
                return Err(
                    "Splat count exceeds 22-bit Metal sort key original_index encoding".into(),
                );
            }

            let screen_width_u32 = u32::try_from(screen_width)?;
            let screen_height_u32 = u32::try_from(screen_height)?;

            let tile_count_x = div_ceil_u32(screen_width_u32, TILE_SIZE).max(1);
            let tile_count_y = div_ceil_u32(screen_height_u32, TILE_SIZE).max(1);
            let num_tiles_u64 = u64::from(tile_count_x) * u64::from(tile_count_y);
            if num_tiles_u64 > MAX_SORT_KEY_TILE_COUNT {
                return Err("Tile count exceeds 16-bit Metal sort key tile_id encoding".into());
            }
            let num_tiles = usize::try_from(num_tiles_u64)?;
            let previous_num_tiles = self.last_num_tiles;
            let overlap_history_reset = self.previous_source_splat_count != source_splat_count
                || self.previous_active_splat_count != active_splat_count;
            if overlap_history_reset {
                self.previous_total_overlaps = 0;
                self.frames_below_threshold = 0;
            }
            self.previous_source_splat_count = source_splat_count;
            self.previous_active_splat_count = active_splat_count;

            self.last_tile_count_x = tile_count_x;
            self.last_tile_count_y = tile_count_y;
            self.last_num_tiles = num_tiles;
            self.last_sort_capacity_before = self.sort_capacity;
            self.last_sort_capacity_after = self.sort_capacity;
            self.last_estimated_overlaps = 0;
            self.last_attempt_sort_count = 0;
            self.last_previous_total_overlaps = self.previous_total_overlaps;
            self.last_actual_total_overlaps = 0;
            self.last_valid_count = 0;
            self.last_retry_count = 0;
            self.last_overflow_flag = 0;
            self.last_tile_density = Default::default();
            self.last_lod_mode = lod_mode.name();
            self.last_lod_mapping = lod_mode.mapping_name();
            self.last_lod_requested_splat_count = lod_requested_splat_count;
            self.last_source_splat_count = source_splat_count;
            self.last_active_splat_count = active_splat_count;
            self.last_overlap_history_reset = overlap_history_reset;
            self.clear_stage_timings();

            self.ensure_framebuffer_capacity(screen_width, screen_height)?;
            if active_splat_count == 0 {
                self.clear_framebuffer(screen_width, screen_height);
                self.last_render_width = screen_width;
                self.last_render_height = screen_height;
                self.last_sort_capacity_after = self.sort_capacity;
                return Ok(());
            }
            self.ensure_tile_capacity(num_tiles)?;

            let estimate_previous_total_overlaps =
                if self.previous_total_overlaps == 0 && lod_mode == MetalLodMode::Fixed {
                    active_splat_count.min(u32::MAX as usize) as u32
                } else {
                    self.previous_total_overlaps
                };
            let estimate_previous_num_tiles =
                if self.previous_total_overlaps == 0 && lod_mode == MetalLodMode::Fixed {
                    num_tiles
                } else {
                    previous_num_tiles
                };
            let mut estimated_overlaps = estimate_overlaps_for_attempt(
                active_splat_count,
                estimate_previous_total_overlaps,
                estimate_previous_num_tiles,
                num_tiles,
            );

            let mut retry_count = 0;
            loop {
                self.ensure_sort_capacity_with_headroom(estimated_overlaps, 2, 1)?;
                let attempt_sort_count = estimated_overlaps.min(self.sort_capacity).max(1);
                self.last_estimated_overlaps = estimated_overlaps;
                self.last_attempt_sort_count = attempt_sort_count;
                self.clear_stage_timings();

                let result = run_single_render_attempt(
                    self,
                    camera,
                    screen_width,
                    screen_height,
                    active_splat_count,
                    source_splat_count,
                    attempt_sort_count,
                )?;

                self.last_actual_total_overlaps = result.total_overlaps;
                self.last_valid_count = result.valid_count;
                self.last_retry_count = retry_count;
                self.last_overflow_flag = result.overflow_flag;
                self.last_tile_density = result.tile_density;
                if result.overflow_flag == 0 {
                    self.previous_total_overlaps = result.total_overlaps;
                    self.maybe_shrink_sort_capacity(result.total_overlaps as usize)?;
                    self.last_sort_capacity_after = self.sort_capacity;
                    break;
                }

                let growth_target = retry_overlap_target(result.total_overlaps, attempt_sort_count);
                self.ensure_sort_capacity(growth_target)?;
                self.last_sort_capacity_after = self.sort_capacity;
                if retry_count >= MAX_OVERFLOW_RETRIES {
                    return Err(MetalRenderError::OverflowDeferred {
                        requested_capacity: growth_target,
                        overlaps: result.total_overlaps,
                    });
                }
                retry_count += 1;
                estimated_overlaps = growth_target;
            }

            self.last_render_width = screen_width;
            self.last_render_height = screen_height;
            Ok(())
        })
    }
}
