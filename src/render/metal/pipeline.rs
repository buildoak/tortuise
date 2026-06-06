use std::{ffi::c_void, mem};

use metal::{Buffer, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions};

use crate::splat::Splat;

use super::error::MetalRenderError;
use super::types::{
    GpuCameraData, GpuProjectedSplat, GpuSplatData, TileConfig, SHADER_TILE_SIZE, TILE_SIZE,
};
use super::MetalBackend;

impl MetalBackend {
    pub fn new(max_splats: usize) -> Result<Self, MetalRenderError> {
        // Keep Rust and shader tile constants in lockstep: the tile overlap logic
        // and rasterization threadgroup dimensions must both use the same TILE_SIZE.
        assert_eq!(
            TILE_SIZE, SHADER_TILE_SIZE,
            "TILE_SIZE mismatch: update Rust constants and shaders/tile_ops.metal + shaders/tile_rasterize.metal together"
        );

        let device = Device::system_default()
            .ok_or_else(|| MetalRenderError::Other("No Metal device found".to_string()))?;
        let command_queue = device.new_command_queue();

        let projection_library =
            compile_library(&device, include_str!("../../../shaders/projection.metal"))?;
        let prefix_scan_library =
            compile_library(&device, include_str!("../../../shaders/prefix_scan.metal"))?;
        let radix_sort_library =
            compile_library(&device, include_str!("../../../shaders/radix_sort.metal"))?;
        let tile_ops_library =
            compile_library(&device, include_str!("../../../shaders/tile_ops.metal"))?;
        let tile_rasterize_library = compile_library(
            &device,
            include_str!("../../../shaders/tile_rasterize.metal"),
        )?;

        let project_splats_pipeline =
            create_pipeline(&device, &projection_library, "project_splats")?;
        let prefix_scan_blocks_pipeline =
            create_pipeline(&device, &prefix_scan_library, "prefix_scan_blocks")?;
        let prefix_scan_add_offsets_pipeline =
            create_pipeline(&device, &prefix_scan_library, "prefix_scan_add_offsets")?;
        let radix_sort_histogram_pipeline =
            create_pipeline(&device, &radix_sort_library, "radix_sort_histogram")?;
        let radix_sort_scatter_pipeline =
            create_pipeline(&device, &radix_sort_library, "radix_sort_scatter")?;
        let count_tile_overlaps_pipeline =
            create_pipeline(&device, &tile_ops_library, "count_tile_overlaps")?;
        let emit_tile_keys_pipeline =
            create_pipeline(&device, &tile_ops_library, "emit_tile_keys")?;
        let rasterize_tiles_pipeline =
            create_pipeline(&device, &tile_rasterize_library, "rasterize_tiles")?;

        let splat_buffer = new_shared_buffer(
            &device,
            max_splats
                .checked_mul(mem::size_of::<GpuSplatData>())
                .ok_or_else(|| MetalRenderError::Other("splat buffer size overflow".to_string()))?,
        );
        let projected_buffer = new_private_buffer(
            &device,
            max_splats
                .checked_mul(mem::size_of::<GpuProjectedSplat>())
                .ok_or_else(|| {
                    MetalRenderError::Other("projected buffer size overflow".to_string())
                })?,
        );

        let camera_buffer = new_shared_buffer(&device, mem::size_of::<GpuCameraData>());
        let valid_count_buffer = new_shared_buffer(&device, mem::size_of::<u32>());
        let total_overlaps_buffer = new_shared_buffer(&device, mem::size_of::<u32>());
        let overflow_flag_buffer = new_shared_buffer(&device, mem::size_of::<u32>());
        let tile_config_buffer = new_shared_buffer(&device, mem::size_of::<TileConfig>());
        let framebuffer = new_shared_buffer(&device, mem::size_of::<u32>());
        let tile_counts = new_private_buffer(&device, mem::size_of::<u32>());
        let tile_offsets = new_private_buffer(&device, mem::size_of::<u32>() * 2);
        let tile_counters = new_private_buffer(&device, mem::size_of::<u32>());
        let sort_keys_a = new_private_buffer(&device, mem::size_of::<u32>());
        let sort_keys_b = new_private_buffer(&device, mem::size_of::<u32>());
        let sort_values_a = new_private_buffer(&device, mem::size_of::<u32>());
        let sort_values_b = new_private_buffer(&device, mem::size_of::<u32>());
        let radix_histograms = new_private_buffer(&device, mem::size_of::<u32>());
        let block_sums = new_private_buffer(&device, mem::size_of::<u32>());

        Ok(Self {
            device,
            command_queue,
            project_splats_pipeline,
            prefix_scan_blocks_pipeline,
            prefix_scan_add_offsets_pipeline,
            radix_sort_histogram_pipeline,
            radix_sort_scatter_pipeline,
            count_tile_overlaps_pipeline,
            emit_tile_keys_pipeline,
            rasterize_tiles_pipeline,
            splat_buffer,
            camera_buffer,
            valid_count_buffer,
            total_overlaps_buffer,
            tile_config_buffer,
            framebuffer,
            projected_buffer,
            tile_counts,
            tile_offsets,
            tile_counters,
            sort_keys_a,
            sort_keys_b,
            sort_values_a,
            sort_values_b,
            radix_histograms,
            block_sums,
            max_splats,
            tile_capacity: 1,
            sort_capacity: 1,
            histogram_capacity: 1,
            block_sums_capacity: 1,
            framebuffer_capacity_pixels: 1,
            splats_uploaded: false,
            previous_total_overlaps: 0,
            overflow_flag_buffer,
            last_render_width: 0,
            last_render_height: 0,
            frames_below_threshold: 0,
            gpu_disabled: false,
            last_tile_count_x: 0,
            last_tile_count_y: 0,
            last_num_tiles: 0,
            last_sort_capacity_before: 1,
            last_sort_capacity_after: 1,
            last_previous_total_overlaps: 0,
            last_actual_total_overlaps: 0,
            last_valid_count: 0,
            last_retry_count: 0,
            last_overflow_flag: 0,
        })
    }

    pub fn upload_splats(&mut self, splats: &[Splat]) -> Result<(), MetalRenderError> {
        if splats.len() > self.max_splats {
            return Err("Too many splats for GPU buffer".into());
        }

        let contents = self.splat_buffer.contents() as *mut GpuSplatData;
        for (i, splat) in splats.iter().enumerate() {
            let packed_color = (splat.color[0] as u32)
                | ((splat.color[1] as u32) << 8)
                | ((splat.color[2] as u32) << 16)
                | (255u32 << 24);

            let gpu = GpuSplatData {
                pos_x: splat.position.x,
                pos_y: splat.position.y,
                pos_z: splat.position.z,
                scale_x: splat.scale.x,
                scale_y: splat.scale.y,
                scale_z: splat.scale.z,
                rot_w: splat.rotation[0],
                rot_x: splat.rotation[1],
                rot_y: splat.rotation[2],
                rot_z: splat.rotation[3],
                opacity: splat.opacity,
                packed_color,
            };

            unsafe {
                *contents.add(i) = gpu;
            }
        }

        self.splats_uploaded = true;
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.splats_uploaded
    }
}

pub(super) fn compile_library(device: &Device, source: &str) -> Result<Library, MetalRenderError> {
    device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|e| MetalRenderError::Other(e.to_string()))
}

pub(super) fn create_pipeline(
    device: &Device,
    library: &Library,
    function_name: &str,
) -> Result<ComputePipelineState, MetalRenderError> {
    let function = library
        .get_function(function_name, None)
        .map_err(|e| MetalRenderError::Other(e.to_string()))?;

    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalRenderError::Other(e.to_string()))
}

pub(super) fn new_shared_buffer(device: &Device, size_bytes: usize) -> Buffer {
    device.new_buffer(
        size_bytes.max(mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub(super) fn new_private_buffer(device: &Device, size_bytes: usize) -> Buffer {
    device.new_buffer(
        size_bytes.max(mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModePrivate,
    )
}

pub(super) fn write_shared_struct<T: Copy>(buffer: &Buffer, value: &T) {
    unsafe {
        *(buffer.contents() as *mut T) = *value;
    }
}

pub(super) fn read_shared_u32(buffer: &Buffer) -> u32 {
    unsafe { *(buffer.contents() as *const u32) }
}

pub(super) fn set_bytes_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        mem::size_of::<u32>() as u64,
        &value as *const _ as *const c_void,
    );
}
