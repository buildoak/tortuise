use std::{mem, ptr};

use super::error::MetalRenderError;
use super::pipeline::{new_private_buffer, new_shared_buffer};
use super::types::THREADS_PER_GROUP_1D;
use super::MetalBackend;

impl MetalBackend {
    pub(super) fn ensure_framebuffer_capacity(
        &mut self,
        screen_width: usize,
        screen_height: usize,
    ) -> Result<(), MetalRenderError> {
        let pixels = screen_width.checked_mul(screen_height).ok_or_else(|| {
            MetalRenderError::Other("framebuffer pixel count overflow".to_string())
        })?;

        if pixels > self.framebuffer_capacity_pixels {
            self.framebuffer = new_shared_buffer(
                &self.device,
                pixels.checked_mul(mem::size_of::<u32>()).ok_or_else(|| {
                    MetalRenderError::Other("framebuffer size overflow".to_string())
                })?,
            );
            self.framebuffer_capacity_pixels = pixels;
        }

        Ok(())
    }

    pub(super) fn ensure_tile_capacity(
        &mut self,
        num_tiles: usize,
    ) -> Result<(), MetalRenderError> {
        if num_tiles <= self.tile_capacity {
            return Ok(());
        }

        self.tile_counts = new_private_buffer(&self.device, bytes_for_u32_elems(num_tiles)?);
        self.tile_offsets = new_private_buffer(&self.device, bytes_for_u32_elems(num_tiles + 1)?);
        self.tile_offsets_readback =
            new_shared_buffer(&self.device, bytes_for_u32_elems(num_tiles + 1)?);
        self.tile_counters = new_private_buffer(&self.device, bytes_for_u32_elems(num_tiles)?);
        self.tile_capacity = num_tiles;
        Ok(())
    }

    pub(super) fn ensure_sort_capacity(&mut self, overlaps: usize) -> Result<(), MetalRenderError> {
        if overlaps <= self.sort_capacity {
            return Ok(());
        }
        let new_capacity = overlaps.next_power_of_two();
        self.reallocate_sort_buffers(new_capacity)
    }

    pub(super) fn reallocate_sort_buffers(
        &mut self,
        new_capacity: usize,
    ) -> Result<(), MetalRenderError> {
        let key_bytes = bytes_for_u64_elems(new_capacity)?;
        let value_bytes = bytes_for_u32_elems(new_capacity)?;
        self.sort_keys_a = new_private_buffer(&self.device, key_bytes);
        self.sort_keys_b = new_private_buffer(&self.device, key_bytes);
        self.sort_values_a = new_private_buffer(&self.device, value_bytes);
        self.sort_values_b = new_private_buffer(&self.device, value_bytes);
        self.sort_capacity = new_capacity;
        Ok(())
    }

    pub fn ensure_sort_capacity_with_headroom(
        &mut self,
        required_overlaps: usize,
        headroom_factor_num: usize,
        headroom_factor_den: usize,
    ) -> Result<(), MetalRenderError> {
        let headroom = required_overlaps
            .checked_mul(headroom_factor_num)
            .ok_or_else(|| MetalRenderError::Other("sort headroom overflow".to_string()))?
            .div_ceil(headroom_factor_den.max(1));

        self.ensure_sort_capacity(headroom.max(1))
    }

    pub fn maybe_shrink_sort_capacity(
        &mut self,
        actual_overlaps: usize,
    ) -> Result<(), MetalRenderError> {
        if self.sort_capacity <= 1 {
            self.frames_below_threshold = 0;
            return Ok(());
        }

        let shrink_threshold = self.sort_capacity / 2;
        if actual_overlaps < shrink_threshold {
            self.frames_below_threshold = self.frames_below_threshold.saturating_add(1);
            if self.frames_below_threshold >= 60 {
                let new_capacity = (self.sort_capacity / 2).max(1);
                if actual_overlaps <= new_capacity {
                    self.reallocate_sort_buffers(new_capacity)?;
                }
                self.frames_below_threshold = 0;
            }
        } else {
            self.frames_below_threshold = 0;
        }

        Ok(())
    }

    pub(super) fn ensure_histogram_capacity(
        &mut self,
        histogram_count: usize,
    ) -> Result<(), MetalRenderError> {
        if histogram_count <= self.histogram_capacity {
            return Ok(());
        }

        self.radix_histograms =
            new_private_buffer(&self.device, bytes_for_u32_elems(histogram_count)?);
        self.histogram_capacity = histogram_count;
        Ok(())
    }

    pub(super) fn ensure_block_sums_capacity_for_count(
        &mut self,
        count: u32,
    ) -> Result<(), MetalRenderError> {
        let required = required_block_sum_elements(count as usize);
        if required <= self.block_sums_capacity {
            return Ok(());
        }

        self.block_sums = new_private_buffer(&self.device, bytes_for_u32_elems(required)?);
        self.block_sums_capacity = required;
        Ok(())
    }

    pub(super) fn clear_framebuffer(&mut self, screen_width: usize, screen_height: usize) {
        if screen_width == 0 || screen_height == 0 {
            return;
        }

        let pixel_count = screen_width.saturating_mul(screen_height);
        let byte_count = pixel_count.saturating_mul(mem::size_of::<u32>());
        unsafe {
            ptr::write_bytes(self.framebuffer.contents() as *mut u8, 0, byte_count);
        }
    }

    pub fn framebuffer_slice(&self) -> &[u32] {
        let pixel_count = self
            .last_render_width
            .saturating_mul(self.last_render_height);
        if pixel_count == 0 {
            return &[];
        }

        let src = self.framebuffer.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(src, pixel_count) }
    }
}

pub(super) fn bytes_for_u32_elems(count: usize) -> Result<usize, MetalRenderError> {
    count
        .checked_mul(mem::size_of::<u32>())
        .ok_or_else(|| MetalRenderError::Other("buffer size overflow".to_string()))
}

pub(super) fn bytes_for_u64_elems(count: usize) -> Result<usize, MetalRenderError> {
    count
        .checked_mul(mem::size_of::<u64>())
        .ok_or_else(|| MetalRenderError::Other("buffer size overflow".to_string()))
}

pub(super) fn required_block_sum_elements(count: usize) -> usize {
    if count == 0 {
        return 1;
    }

    let mut total = 0usize;
    let mut blocks = count.div_ceil(THREADS_PER_GROUP_1D as usize);
    loop {
        total = total.saturating_add(blocks);
        if blocks <= 1 {
            break;
        }
        blocks = blocks.div_ceil(THREADS_PER_GROUP_1D as usize);
    }
    total.max(1)
}
