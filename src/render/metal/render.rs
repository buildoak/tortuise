use crate::camera::Camera;
use objc::rc::autoreleasepool;

use super::error::MetalRenderError;
use super::render_attempt::run_single_render_attempt;
use super::sort::div_ceil_u32;
use super::types::TILE_SIZE;
use super::MetalBackend;

const MAX_SORT_KEY_ORIGINAL_INDEX: usize = (1 << 22) - 1;

impl MetalBackend {
    pub fn render(
        &mut self,
        camera: &Camera,
        screen_width: usize,
        screen_height: usize,
        splat_count: usize,
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

            if splat_count > self.max_splats {
                return Err("Too many splats for GPU buffers".into());
            }
            if splat_count > MAX_SORT_KEY_ORIGINAL_INDEX + 1 {
                return Err(
                    "Splat count exceeds 22-bit Metal sort key original_index encoding".into(),
                );
            }

            let screen_width_u32 = u32::try_from(screen_width)?;
            let screen_height_u32 = u32::try_from(screen_height)?;

            let tile_count_x = div_ceil_u32(screen_width_u32, TILE_SIZE).max(1);
            let tile_count_y = div_ceil_u32(screen_height_u32, TILE_SIZE).max(1);
            let num_tiles_u64 = u64::from(tile_count_x) * u64::from(tile_count_y);
            if num_tiles_u64 > 1023 {
                return Err("Tile count exceeds 10-bit tile_id encoding (max 1023 tiles)".into());
            }
            let num_tiles = usize::try_from(num_tiles_u64)?;
            let previous_num_tiles = self.last_num_tiles;

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

            self.ensure_framebuffer_capacity(screen_width, screen_height)?;
            if splat_count == 0 {
                self.clear_framebuffer(screen_width, screen_height);
                self.last_render_width = screen_width;
                self.last_render_height = screen_height;
                self.last_sort_capacity_after = self.sort_capacity;
                return Ok(());
            }
            self.ensure_tile_capacity(num_tiles)?;

            let estimated_overlaps = if self.previous_total_overlaps > 0 {
                let previous_overlaps = self.previous_total_overlaps as usize;
                let scene_floor_factor = if splat_count < 1024 { 16 } else { 1 };
                let tile_scaled_previous = if previous_num_tiles > 0 {
                    previous_overlaps
                        .saturating_mul(num_tiles)
                        .div_ceil(previous_num_tiles)
                } else {
                    previous_overlaps
                };
                tile_scaled_previous
                    .saturating_mul(5)
                    .div_ceil(4)
                    .max(splat_count.saturating_mul(scene_floor_factor))
            } else {
                let cold_start_factor = if splat_count < 1024 { 16 } else { 8 };
                splat_count.saturating_mul(cold_start_factor)
            }
            .max(1);

            self.ensure_sort_capacity_with_headroom(estimated_overlaps, 2, 1)?;
            let attempt_sort_count = estimated_overlaps.min(self.sort_capacity).max(1);
            self.last_estimated_overlaps = estimated_overlaps;
            self.last_attempt_sort_count = attempt_sort_count;
            let result = run_single_render_attempt(
                self,
                camera,
                screen_width,
                screen_height,
                splat_count,
                attempt_sort_count,
            )?;

            self.previous_total_overlaps = result.total_overlaps;
            self.last_actual_total_overlaps = result.total_overlaps;
            self.last_valid_count = result.valid_count;
            self.last_retry_count = 0;
            self.last_overflow_flag = result.overflow_flag;
            self.last_tile_density = result.tile_density;
            if result.overflow_flag == 0 {
                self.maybe_shrink_sort_capacity(result.total_overlaps as usize)?;
                self.last_sort_capacity_after = self.sort_capacity;
            } else {
                let growth_target = (result.total_overlaps as usize)
                    .max(attempt_sort_count.saturating_mul(2))
                    .max(1);
                self.ensure_sort_capacity(growth_target)?;
                self.last_sort_capacity_after = self.sort_capacity;
                return Err(MetalRenderError::OverflowDeferred {
                    requested_capacity: growth_target,
                    overlaps: result.total_overlaps,
                });
            }

            self.last_render_width = screen_width;
            self.last_render_height = screen_height;
            Ok(())
        })
    }
}
