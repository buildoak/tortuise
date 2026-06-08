pub mod frame;
mod frame_halfblock;
pub mod frame_kitty;
pub mod hud;
pub mod live_telemetry;
#[cfg(feature = "metal")]
pub mod lod;
#[cfg(feature = "metal")]
pub mod metal;
pub mod modes;
pub mod pipeline;
pub mod probe;
pub mod rasterizer;

use std::time::Instant;

use crate::camera::Camera;
use crate::math::Vec3;
use crate::splat::{ProjectedSplat, Splat};
use crossterm::style::Color;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraMode {
    Free,
    Orbit,
}

impl CameraMode {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Free => "FREE",
            Self::Orbit => "ORBIT",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudMode {
    Minimal,
    Debug,
    Hidden,
}

impl HudMode {
    pub fn next(self) -> Self {
        match self {
            Self::Minimal => Self::Debug,
            Self::Debug => Self::Hidden,
            Self::Hidden => Self::Minimal,
        }
    }

    pub fn is_visible(self) -> bool {
        !matches!(self, Self::Hidden)
    }
}

/// Weighted perceptual distance squared (green 2x, red 1.5x, blue 1x sensitivity).
fn perceptual_dist_sq(r1: u8, g1: u8, b1: u8, r2: u8, g2: u8, b2: u8) -> u32 {
    let dr = r1 as i32 - r2 as i32;
    let dg = g1 as i32 - g2 as i32;
    let db = b1 as i32 - b2 as i32;
    (2 * dr * dr + 4 * dg * dg + db * db) as u32
}

pub fn rgb_to_ansi256(r: u8, g: u8, b: u8) -> u8 {
    // --- Candidate 1: best match from 6x6x6 color cube (indices 16-231) ---
    // Round each channel independently, then search ±1 neighbors for perceptual best.
    let ri0 = ((r as f32 / 255.0 * 5.0) + 0.5) as i8;
    let gi0 = ((g as f32 / 255.0 * 5.0) + 0.5) as i8;
    let bi0 = ((b as f32 / 255.0 * 5.0) + 0.5) as i8;

    let mut best_cube_idx: u8 = 16 + 36 * ri0 as u8 + 6 * gi0 as u8 + bi0 as u8;
    let mut best_cube_dist =
        perceptual_dist_sq(r, g, b, ri0 as u8 * 51, gi0 as u8 * 51, bi0 as u8 * 51);

    for dr in -1i8..=1 {
        for dg in -1i8..=1 {
            for db in -1i8..=1 {
                let ri = ri0 + dr;
                let gi = gi0 + dg;
                let bi = bi0 + db;
                if !(0..=5).contains(&ri) || !(0..=5).contains(&gi) || !(0..=5).contains(&bi) {
                    continue;
                }
                let cr = ri as u8 * 51;
                let cg = gi as u8 * 51;
                let cb = bi as u8 * 51;
                let d = perceptual_dist_sq(r, g, b, cr, cg, cb);
                if d < best_cube_dist {
                    best_cube_dist = d;
                    best_cube_idx = 16 + 36 * ri as u8 + 6 * gi as u8 + bi as u8;
                }
            }
        }
    }

    // --- Candidate 2: grayscale ramp (indices 232-255, levels 8,18,...,238) ---
    // Use it when channels are close (near-gray) OR exactly equal.
    let max_ch = r.max(g).max(b);
    let min_ch = r.min(g).min(b);
    if max_ch - min_ch < 12 {
        let avg = (r as u16 + g as u16 + b as u16) / 3;
        if avg < 4 {
            return 16; // near-black: use cube black
        }
        if avg > 246 {
            return 231; // near-white: use cube white
        }
        // Nearest grayscale ramp entry: levels are 8 + 10*i for i in 0..24
        let gi = (((avg as f32 - 8.0) / 10.0).round() as u8).min(23);
        let gray_level = (8 + 10 * gi as u16) as u8;
        let gray_dist = perceptual_dist_sq(r, g, b, gray_level, gray_level, gray_level);
        if gray_dist <= best_cube_dist {
            return 232 + gi;
        }
    }

    best_cube_idx
}

pub fn make_color(r: u8, g: u8, b: u8, use_truecolor: bool) -> Color {
    if use_truecolor {
        Color::Rgb { r, g, b }
    } else {
        Color::AnsiValue(rgb_to_ansi256(r, g, b))
    }
}

pub use crate::AppResult;
pub type HalfblockCell = ([u8; 3], [u8; 3]);

pub const HALF_BLOCK: char = '\u{2584}';
pub const FRAME_TARGET: std::time::Duration = std::time::Duration::from_millis(8);

#[derive(Debug)]
pub struct RenderState {
    pub framebuffer: Vec<[u8; 3]>,
    pub alpha_buffer: Vec<f32>,
    pub depth_buffer: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderMode {
    Halfblock,
    #[cfg(feature = "metal")]
    Kitty,
    PointCloud,
    Matrix,
    BlockDensity,
    Braille,
    AsciiClassic,
}

impl RenderMode {
    pub fn next(self) -> Self {
        match self {
            Self::Halfblock => {
                #[cfg(feature = "metal")]
                {
                    Self::Kitty
                }
                #[cfg(not(feature = "metal"))]
                {
                    Self::PointCloud
                }
            }
            #[cfg(feature = "metal")]
            Self::Kitty => Self::PointCloud,
            Self::PointCloud => Self::Matrix,
            Self::Matrix => Self::BlockDensity,
            Self::BlockDensity => Self::Braille,
            Self::Braille => Self::AsciiClassic,
            Self::AsciiClassic => Self::Halfblock,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Halfblock => "Halfblock",
            #[cfg(feature = "metal")]
            Self::Kitty => "Kitty",
            Self::PointCloud => "PointCloud",
            Self::Matrix => "Matrix",
            Self::BlockDensity => "BlockDensity",
            Self::Braille => "Braille",
            Self::AsciiClassic => "AsciiClassic",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RenderMode;

    #[test]
    fn render_mode_cycle_returns_to_halfblock() {
        let mut mode = RenderMode::Halfblock;
        let mut names = Vec::new();
        loop {
            names.push(mode.name());
            mode = mode.next();
            if mode == RenderMode::Halfblock {
                break;
            }
        }

        #[cfg(feature = "metal")]
        assert_eq!(
            names,
            vec![
                "Halfblock",
                "Kitty",
                "PointCloud",
                "Matrix",
                "BlockDensity",
                "Braille",
                "AsciiClassic"
            ]
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            names,
            vec![
                "Halfblock",
                "PointCloud",
                "Matrix",
                "BlockDensity",
                "Braille",
                "AsciiClassic"
            ]
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "metal")]
    Metal,
}

impl Backend {
    #[allow(dead_code)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(feature = "metal")]
            Self::Metal => "Metal",
        }
    }
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalLodMode {
    Off,
    Fixed,
}

#[cfg(feature = "metal")]
impl MetalLodMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fixed => "fixed",
        }
    }
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalLodOrder {
    FloorEven,
    Voxel,
}

#[cfg(feature = "metal")]
impl MetalLodOrder {
    pub fn name(self) -> &'static str {
        match self {
            Self::FloorEven => "floor_even",
            Self::Voxel => "voxel",
        }
    }
}

#[derive(Debug)]
pub struct AppState {
    pub camera: Camera,
    pub splats: Vec<Splat>,
    pub projected_splats: Vec<ProjectedSplat>,
    pub render_state: RenderState,
    pub halfblock_cells: Vec<HalfblockCell>,
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    pub scene_label: String,
    pub input_state: crate::input::state::InputState,
    pub show_hud: bool,
    pub hud_mode: HudMode,
    pub camera_mode: CameraMode,
    pub move_speed: f32,
    pub frame_count: u64,
    pub last_frame_time: Instant,
    pub fps: f32,
    pub visible_splat_count: usize,
    pub effective_render_path: &'static str,
    pub last_render_width: usize,
    pub last_render_height: usize,
    pub last_terminal_cols: usize,
    pub last_terminal_rows: usize,
    pub orbit_angle: f32,
    pub orbit_radius: f32,
    pub orbit_height: f32,
    pub orbit_target: Vec3,
    pub supersample_factor: u32,
    pub render_mode: RenderMode,
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    pub backend: Backend,
    pub use_truecolor: bool,
    #[cfg(feature = "metal")]
    pub metal_lod_mode: MetalLodMode,
    #[cfg(feature = "metal")]
    pub metal_lod_requested_splat_count: Option<usize>,
    #[cfg(feature = "metal")]
    pub metal_active_splat_count: usize,
    #[cfg(feature = "metal")]
    pub metal_lod_order: MetalLodOrder,
    #[cfg(feature = "metal")]
    pub kitty_image_id: u32,
    #[cfg(feature = "metal")]
    pub kitty_visible_image_id: u32,
    #[cfg(feature = "metal")]
    pub kitty_pixel_hud_active: bool,
    #[cfg(feature = "metal")]
    pub kitty_payload_bytes: usize,
    #[cfg(feature = "metal")]
    pub kitty_base64_bytes: usize,
    #[cfg(feature = "metal")]
    pub kitty_chunks: usize,
    #[cfg(feature = "metal")]
    pub kitty_write_ms: f32,
    #[cfg(feature = "metal")]
    pub kitty_convert_ms: f32,
    #[cfg(feature = "metal")]
    pub kitty_encode_ms: f32,
    pub last_flush_ms: f32,
    pub last_telemetry_write_ms: f32,
    #[cfg(feature = "metal")]
    pub metal_backend: Option<crate::render::metal::MetalBackend>,
    #[cfg(feature = "metal")]
    pub last_gpu_error: Option<String>,
    #[cfg(feature = "metal")]
    pub gpu_fallback_active: bool,
    pub live_telemetry: live_telemetry::LiveTelemetryState,
}
