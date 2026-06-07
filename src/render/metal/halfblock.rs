use std::time::Duration;

use metal::MTLSize;

use super::error::MetalRenderError;
use super::pipeline::write_shared_struct;
use super::sync::commit_and_wait_or_disable_gpu;
use super::types::GpuHalfblockConfig;
use super::MetalBackend;

const GPU_HALFBLOCK_WAIT_TIMEOUT: Duration = Duration::from_millis(500);
const HALFBLOCK_THREADS_X: u64 = 16;
const HALFBLOCK_THREADS_Y: u64 = 16;

impl MetalBackend {
    pub fn downsample_halfblock_cells(
        &mut self,
        framebuffer_width: usize,
        framebuffer_height: usize,
        term_cols: usize,
        term_rows: usize,
        supersample: usize,
    ) -> Result<(), MetalRenderError> {
        if self.gpu_disabled {
            return Err(MetalRenderError::GpuDisabled);
        }
        if term_cols == 0 || term_rows == 0 {
            self.last_halfblock_cols = term_cols;
            self.last_halfblock_rows = term_rows;
            return Ok(());
        }

        let framebuffer_width = u32::try_from(framebuffer_width)?;
        let framebuffer_height = u32::try_from(framebuffer_height)?;
        let term_cols_u32 = u32::try_from(term_cols)?;
        let term_rows_u32 = u32::try_from(term_rows)?;
        let supersample_u32 = u32::try_from(supersample)?;
        if supersample_u32 == 0 {
            return Err(
                "supersample factor must be non-zero for Metal halfblock downsample".into(),
            );
        }

        self.ensure_halfblock_capacity(term_cols, term_rows)?;
        write_shared_struct(
            &self.halfblock_config_buffer,
            &GpuHalfblockConfig {
                framebuffer_width,
                framebuffer_height,
                term_cols: term_cols_u32,
                term_rows: term_rows_u32,
                supersample: supersample_u32,
                _reserved0: 0,
                _reserved1: 0,
                _reserved2: 0,
            },
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.downsample_halfblock_pipeline);
        encoder.set_buffer(0, Some(&self.framebuffer), 0);
        encoder.set_buffer(1, Some(&self.halfblock_cells), 0);
        encoder.set_buffer(2, Some(&self.halfblock_config_buffer), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(term_cols_u32).div_ceil(HALFBLOCK_THREADS_X),
                u64::from(term_rows_u32).div_ceil(HALFBLOCK_THREADS_Y),
                1,
            ),
            MTLSize::new(HALFBLOCK_THREADS_X, HALFBLOCK_THREADS_Y, 1),
        );
        encoder.end_encoding();

        commit_and_wait_or_disable_gpu(
            command_buffer,
            "halfblock_downsample",
            GPU_HALFBLOCK_WAIT_TIMEOUT,
            &mut self.gpu_disabled,
        )?;

        self.last_halfblock_cols = term_cols;
        self.last_halfblock_rows = term_rows;
        Ok(())
    }
}
