use std::{ffi::c_void, mem};

use metal::{Buffer, MTLSize, NSRange};

use super::error::MetalRenderError;
use super::types::{RADIX_BUCKETS, THREADS_PER_GROUP_1D};
use super::MetalBackend;

pub(super) const RADIX_SORT_BIT_OFFSETS: [u32; 8] = [0u32, 8, 16, 24, 32, 40, 48, 56];

pub fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

pub fn dispatch_1d(encoder: &metal::ComputeCommandEncoderRef, count: u32, threads_per_group: u32) {
    if count == 0 {
        return;
    }

    let groups = u64::from(div_ceil_u32(count, threads_per_group));
    encoder.dispatch_thread_groups(
        MTLSize::new(groups, 1, 1),
        MTLSize::new(u64::from(threads_per_group), 1, 1),
    );
}

impl MetalBackend {
    pub(super) fn encode_prefix_scan_in_place(
        &self,
        command_buffer: &metal::CommandBufferRef,
        data_buffer: &Buffer,
        data_offset_bytes: u64,
        count: u32,
    ) -> Result<(), MetalRenderError> {
        if count == 0 {
            return Ok(());
        }

        self.encode_prefix_scan_recursive(command_buffer, data_buffer, data_offset_bytes, count, 0)
    }

    fn encode_prefix_scan_recursive(
        &self,
        command_buffer: &metal::CommandBufferRef,
        data_buffer: &Buffer,
        data_offset_bytes: u64,
        count: u32,
        scratch_offset_elems: u64,
    ) -> Result<(), MetalRenderError> {
        if count == 0 {
            return Ok(());
        }

        let num_blocks = div_ceil_u32(count, THREADS_PER_GROUP_1D);
        if num_blocks == 0 {
            return Ok(());
        }

        let block_sums_offset_bytes = scratch_offset_elems
            .checked_mul(mem::size_of::<u32>() as u64)
            .ok_or_else(|| MetalRenderError::Other("block sums offset overflow".to_string()))?;

        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.prefix_scan_blocks_pipeline);
            encoder.set_buffer(0, Some(data_buffer), data_offset_bytes);
            encoder.set_buffer(1, Some(&self.block_sums), block_sums_offset_bytes);
            encoder.set_bytes(
                2,
                mem::size_of::<u32>() as u64,
                &count as *const _ as *const c_void,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(u64::from(num_blocks), 1, 1),
                MTLSize::new(u64::from(THREADS_PER_GROUP_1D), 1, 1),
            );
            encoder.end_encoding();
        }

        if num_blocks > 1 {
            let next_scratch_offset = scratch_offset_elems
                .checked_add(u64::from(num_blocks))
                .ok_or_else(|| {
                    MetalRenderError::Other("block sums recursion overflow".to_string())
                })?;

            self.encode_prefix_scan_recursive(
                command_buffer,
                &self.block_sums,
                block_sums_offset_bytes,
                num_blocks,
                next_scratch_offset,
            )?;

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.prefix_scan_add_offsets_pipeline);
            encoder.set_buffer(0, Some(data_buffer), data_offset_bytes);
            encoder.set_buffer(1, Some(&self.block_sums), block_sums_offset_bytes);
            encoder.set_bytes(
                2,
                mem::size_of::<u32>() as u64,
                &count as *const _ as *const c_void,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(u64::from(num_blocks), 1, 1),
                MTLSize::new(u64::from(THREADS_PER_GROUP_1D), 1, 1),
            );
            encoder.end_encoding();
        }

        Ok(())
    }

    pub(super) fn run_radix_sort_passes(
        &self,
        command_buffer: &metal::CommandBufferRef,
        dispatch_overlaps: u32,
        keys_in_a: &mut bool,
    ) -> Result<(), MetalRenderError> {
        let num_blocks = div_ceil_u32(dispatch_overlaps.max(1), THREADS_PER_GROUP_1D).max(1);
        let histogram_count = num_blocks
            .checked_mul(RADIX_BUCKETS)
            .ok_or_else(|| MetalRenderError::Other("histogram count overflow".to_string()))?;
        let histogram_bytes = histogram_count as u64 * mem::size_of::<u32>() as u64;

        for bit_offset in RADIX_SORT_BIT_OFFSETS {
            let blit = command_buffer.new_blit_command_encoder();
            blit.fill_buffer(&self.radix_histograms, NSRange::new(0, histogram_bytes), 0);
            blit.end_encoding();

            let (keys_in, values_in, keys_out, values_out) = if *keys_in_a {
                (
                    &self.sort_keys_a,
                    &self.sort_values_a,
                    &self.sort_keys_b,
                    &self.sort_values_b,
                )
            } else {
                (
                    &self.sort_keys_b,
                    &self.sort_values_b,
                    &self.sort_keys_a,
                    &self.sort_values_a,
                )
            };

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.radix_sort_histogram_pipeline);
            encoder.set_buffer(0, Some(keys_in), 0);
            encoder.set_buffer(1, Some(&self.radix_histograms), 0);
            encoder.set_buffer(2, Some(&self.total_overlaps_buffer), 0);
            encoder.set_bytes(
                3,
                mem::size_of::<u32>() as u64,
                &bit_offset as *const _ as *const c_void,
            );
            dispatch_1d(encoder, dispatch_overlaps, THREADS_PER_GROUP_1D);
            encoder.end_encoding();

            self.encode_prefix_scan_in_place(
                command_buffer,
                &self.radix_histograms,
                0,
                histogram_count,
            )?;

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.radix_sort_scatter_pipeline);
            encoder.set_buffer(0, Some(keys_in), 0);
            encoder.set_buffer(1, Some(values_in), 0);
            encoder.set_buffer(2, Some(keys_out), 0);
            encoder.set_buffer(3, Some(values_out), 0);
            encoder.set_buffer(4, Some(&self.radix_histograms), 0);
            encoder.set_buffer(5, Some(&self.total_overlaps_buffer), 0);
            encoder.set_bytes(
                6,
                mem::size_of::<u32>() as u64,
                &bit_offset as *const _ as *const c_void,
            );
            dispatch_1d(encoder, dispatch_overlaps, THREADS_PER_GROUP_1D);
            encoder.end_encoding();

            *keys_in_a = !*keys_in_a;
        }

        Ok(())
    }
}
