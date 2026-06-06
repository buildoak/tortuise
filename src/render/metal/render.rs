use crate::camera::Camera;
use objc::rc::autoreleasepool;

use super::error::MetalRenderError;
use super::render_attempt::run_single_render_attempt;
use super::sort::div_ceil_u32;
use super::types::TILE_SIZE;
use super::MetalBackend;

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

            let screen_width_u32 = u32::try_from(screen_width)?;
            let screen_height_u32 = u32::try_from(screen_height)?;

            let tile_count_x = div_ceil_u32(screen_width_u32, TILE_SIZE).max(1);
            let tile_count_y = div_ceil_u32(screen_height_u32, TILE_SIZE).max(1);
            let num_tiles_u64 = u64::from(tile_count_x) * u64::from(tile_count_y);
            if num_tiles_u64 > 1023 {
                return Err("Tile count exceeds 10-bit tile_id encoding (max 1023 tiles)".into());
            }
            let num_tiles = usize::try_from(num_tiles_u64)?;

            self.last_tile_count_x = tile_count_x;
            self.last_tile_count_y = tile_count_y;
            self.last_num_tiles = num_tiles;
            self.last_sort_capacity_before = self.sort_capacity;
            self.last_sort_capacity_after = self.sort_capacity;
            self.last_previous_total_overlaps = self.previous_total_overlaps;
            self.last_actual_total_overlaps = 0;
            self.last_valid_count = 0;
            self.last_retry_count = 0;
            self.last_overflow_flag = 0;

            self.ensure_framebuffer_capacity(screen_width, screen_height)?;
            if splat_count == 0 {
                self.clear_framebuffer(screen_width, screen_height);
                self.last_render_width = screen_width;
                self.last_render_height = screen_height;
                self.last_sort_capacity_after = self.sort_capacity;
                return Ok(());
            }
            self.ensure_tile_capacity(num_tiles)?;

            let mut attempt = 0u32;
            loop {
                let estimated_overlaps = if self.previous_total_overlaps > 0 {
                    (self.previous_total_overlaps as usize)
                        .saturating_mul(2)
                        .max(splat_count.saturating_mul(4))
                } else {
                    splat_count.saturating_mul(8)
                }
                .max(1);

                self.ensure_sort_capacity_with_headroom(estimated_overlaps, 2, 1)?;
                let result = run_single_render_attempt(
                    self,
                    camera,
                    screen_width,
                    screen_height,
                    splat_count,
                )?;

                self.previous_total_overlaps = result.total_overlaps;
                self.last_actual_total_overlaps = result.total_overlaps;
                self.last_valid_count = result.valid_count;
                self.last_retry_count = attempt;
                self.last_overflow_flag = result.overflow_flag;
                if result.overflow_flag == 0 {
                    self.maybe_shrink_sort_capacity(result.total_overlaps as usize)?;
                    self.last_sort_capacity_after = self.sort_capacity;
                    break;
                }

                if attempt >= 1 {
                    let growth_target = (result.total_overlaps as usize).saturating_mul(2).max(1);
                    self.ensure_sort_capacity(growth_target)?;
                    self.last_retry_count = attempt + 1;
                    self.last_sort_capacity_after = self.sort_capacity;
                    return Err(MetalRenderError::OverflowDeferred {
                        requested_capacity: growth_target,
                        overlaps: result.total_overlaps,
                    });
                }

                let retry_target = (result.total_overlaps as usize).saturating_mul(2).max(1);
                self.ensure_sort_capacity(retry_target)?;
                attempt += 1;
            }

            self.last_render_width = screen_width;
            self.last_render_height = screen_height;
            Ok(())
        })
    }
}
