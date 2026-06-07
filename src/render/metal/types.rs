pub(super) const TILE_SIZE: u32 = 16;
pub(super) const SHADER_TILE_SIZE: u32 = 16;
pub(super) const THREADS_PER_GROUP_1D: u32 = 256;
pub(super) const RADIX_BUCKETS: u32 = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuSplatData {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    pub rot_w: f32,
    pub rot_x: f32,
    pub rot_y: f32,
    pub rot_z: f32,
    pub opacity: f32,
    pub packed_color: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuCameraData {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub right_x: f32,
    pub right_y: f32,
    pub right_z: f32,
    pub up_x: f32,
    pub up_y: f32,
    pub up_z: f32,
    pub forward_x: f32,
    pub forward_y: f32,
    pub forward_z: f32,
    pub fx: f32,
    pub fy: f32,
    pub half_w: f32,
    pub half_h: f32,
    pub near_plane: f32,
    pub far_plane: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuProjectedSplat {
    pub screen_x: f32,
    pub screen_y: f32,
    pub depth: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub cov_a: f32,
    pub cov_b: f32,
    pub cov_c: f32,
    pub opacity: f32,
    pub packed_color: u32,
    pub original_index: u32,
    pub tile_min: u32,
    pub tile_max: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    pub screen_width: u32,
    pub screen_height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuHalfblockConfig {
    pub framebuffer_width: u32,
    pub framebuffer_height: u32,
    pub term_cols: u32,
    pub term_rows: u32,
    pub supersample: u32,
    pub _reserved0: u32,
    pub _reserved1: u32,
    pub _reserved2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuHalfblockCell {
    pub top_rgb: u32,
    pub bottom_rgb: u32,
}

const _: [(); 48] = [(); std::mem::size_of::<GpuSplatData>()];
const _: [(); 72] = [(); std::mem::size_of::<GpuCameraData>()];
const _: [(); 52] = [(); std::mem::size_of::<GpuProjectedSplat>()];
const _: [(); 16] = [(); std::mem::size_of::<TileConfig>()];
const _: [(); 32] = [(); std::mem::size_of::<GpuHalfblockConfig>()];
const _: [(); 8] = [(); std::mem::size_of::<GpuHalfblockCell>()];
