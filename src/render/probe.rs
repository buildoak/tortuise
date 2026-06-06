#![cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]

use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::camera::{self, Camera};
use crate::math::Vec3;
use crate::sort::sort_by_depth;
use crate::splat::{ProjectedSplat, Splat};

use super::RenderState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeBackendSelection {
    Cpu,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "metal")]
    Both,
}

impl fmt::Display for ProbeBackendSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            #[cfg(feature = "metal")]
            Self::Metal => f.write_str("metal"),
            #[cfg(feature = "metal")]
            Self::Both => f.write_str("both"),
        }
    }
}

impl FromStr for ProbeBackendSelection {
    type Err = ProbeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            #[cfg(feature = "metal")]
            "metal" => Ok(Self::Metal),
            #[cfg(feature = "metal")]
            "both" => Ok(Self::Both),
            #[cfg(not(feature = "metal"))]
            "metal" | "both" => Err(ProbeParseError::unavailable_backend(value)),
            _ => Err(ProbeParseError::new("backend", value)),
        }
    }
}

impl ProbeBackendSelection {
    fn renders_cpu(self) -> bool {
        match self {
            Self::Cpu => true,
            #[cfg(feature = "metal")]
            Self::Metal => false,
            #[cfg(feature = "metal")]
            Self::Both => true,
        }
    }

    #[cfg(feature = "metal")]
    fn renders_metal(self) -> bool {
        match self {
            Self::Cpu => false,
            Self::Metal | Self::Both => true,
        }
    }

    #[cfg(feature = "metal")]
    fn compares_cpu_to_metal(self) -> bool {
        match self {
            Self::Both => true,
            Self::Cpu | Self::Metal => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeCase {
    Blank,
    Channels,
    Depth,
    TileBoundary,
    Loaded,
}

impl ProbeCase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Blank => "blank",
            Self::Channels => "channels",
            Self::Depth => "depth",
            Self::TileBoundary => "tile-boundary",
            Self::Loaded => "loaded",
        }
    }

    fn splats(self, loaded_splats: &[Splat]) -> Vec<Splat> {
        match self {
            Self::Blank => Vec::new(),
            Self::Channels => synthetic_channels_splats(),
            Self::Depth => synthetic_depth_splats(),
            Self::TileBoundary => synthetic_tile_boundary_splats(),
            Self::Loaded => loaded_splats.to_vec(),
        }
    }
}

impl fmt::Display for ProbeCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProbeCase {
    type Err = ProbeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "blank" => Ok(Self::Blank),
            "channels" => Ok(Self::Channels),
            "depth" => Ok(Self::Depth),
            "tile-boundary" | "tile_boundary" | "tileboundary" => Ok(Self::TileBoundary),
            "loaded" => Ok(Self::Loaded),
            _ => Err(ProbeParseError::new("case", value)),
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct ProbeParseError {
    field: &'static str,
    value: String,
    kind: ProbeParseErrorKind,
}

#[derive(Debug, Clone, PartialEq)]
enum ProbeParseErrorKind {
    Unknown,
    #[cfg(not(feature = "metal"))]
    UnavailableBackend,
}

impl ProbeParseError {
    fn new(field: &'static str, value: &str) -> Self {
        Self {
            field,
            value: value.to_string(),
            kind: ProbeParseErrorKind::Unknown,
        }
    }

    #[cfg(not(feature = "metal"))]
    fn unavailable_backend(value: &str) -> Self {
        Self {
            field: "backend",
            value: value.to_string(),
            kind: ProbeParseErrorKind::UnavailableBackend,
        }
    }
}

impl fmt::Display for ProbeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ProbeParseErrorKind::Unknown => {
                write!(f, "unknown probe {} '{}'", self.field, self.value)
            }
            #[cfg(not(feature = "metal"))]
            ProbeParseErrorKind::UnavailableBackend => write!(
                f,
                "probe backend '{}' requires building tortuise with --features metal",
                self.value
            ),
        }
    }
}

impl fmt::Debug for ProbeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for ProbeParseError {}

#[derive(Debug, Clone)]
pub struct ProbeCameraSpec {
    pub position: Vec3,
    pub target: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for ProbeCameraSpec {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 4.0),
            target: Vec3::ZERO,
            fov: std::f32::consts::PI / 3.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl ProbeCameraSpec {
    pub fn to_camera(&self) -> Camera {
        let mut camera = Camera::new(self.position, -std::f32::consts::FRAC_PI_2, 0.0);
        camera.fov = self.fov;
        camera.near = self.near;
        camera.far = self.far;
        camera::look_at_target(&mut camera, self.target);
        camera
    }
}

#[derive(Debug, Clone)]
pub struct ProbeConfig {
    pub width: usize,
    pub height: usize,
    pub inspect_scale: usize,
    pub out_dir: PathBuf,
    pub case: ProbeCase,
    pub backend: ProbeBackendSelection,
    pub camera: ProbeCameraSpec,
    pub frames: usize,
    pub warmup_frames: usize,
    pub terminal_artifacts: bool,
}

impl ProbeConfig {
    pub fn new(out_dir: impl Into<PathBuf>) -> Self {
        Self {
            width: 64,
            height: 48,
            inspect_scale: 1,
            out_dir: out_dir.into(),
            case: ProbeCase::Channels,
            backend: ProbeBackendSelection::Cpu,
            camera: ProbeCameraSpec::default(),
            frames: 1,
            warmup_frames: 0,
            terminal_artifacts: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeBoundingBox {
    pub min_x: usize,
    pub min_y: usize,
    pub max_x: usize,
    pub max_y: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeFrameStats {
    pub width: usize,
    pub height: usize,
    pub nonblack_pixels: usize,
    pub sum_r: u64,
    pub sum_g: u64,
    pub sum_b: u64,
    pub luma_min: u8,
    pub luma_max: u8,
    pub luma_mean: f64,
    pub luma_p95: u8,
    pub bbox: Option<ProbeBoundingBox>,
    pub checksum: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeFrameArtifact {
    pub frame_path: PathBuf,
    pub inspect_png_path: PathBuf,
    pub stats_path: PathBuf,
    pub stats: ProbeFrameStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeMetalFrameArtifact {
    pub frame_path: PathBuf,
    pub inspect_png_path: PathBuf,
    pub stats_path: PathBuf,
    pub packed_u32le_path: PathBuf,
    pub stats: ProbeFrameStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeDiffClassification {
    Pass,
    Blank,
    ChannelSwap,
    Mismatch,
}

impl ProbeDiffClassification {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Blank => "blank",
            Self::ChannelSwap => "channel_swap",
            Self::Mismatch => "mismatch",
        }
    }
}

impl fmt::Display for ProbeDiffClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeDiffMetrics {
    pub width: usize,
    pub height: usize,
    pub mean_abs: f64,
    pub max_abs: u8,
    pub p95_abs: u8,
    pub mismatch_pixels: usize,
    pub mismatch_ratio: f64,
    pub sum_abs_r: u64,
    pub sum_abs_g: u64,
    pub sum_abs_b: u64,
    pub classification: ProbeDiffClassification,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeDiffFrameArtifact {
    pub frame_path: PathBuf,
    pub inspect_png_path: PathBuf,
    pub metrics: ProbeDiffMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeRunResult {
    pub manifest_path: PathBuf,
    pub contact_sheet_path: PathBuf,
    pub inspect_contact_sheet_path: PathBuf,
    pub cpu_frames: Vec<ProbeFrameArtifact>,
    pub metal_frames: Vec<ProbeMetalFrameArtifact>,
    pub diff_frames: Vec<ProbeDiffFrameArtifact>,
    pub diff_summary_path: Option<PathBuf>,
}

pub enum ProbeError {
    Io(io::Error),
    InvalidConfig(String),
    Render(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => err.fmt(f),
            Self::InvalidConfig(msg) => f.write_str(msg),
            Self::Render(msg) => f.write_str(msg),
        }
    }
}

impl fmt::Debug for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for ProbeError {}

impl From<io::Error> for ProbeError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

#[cfg(feature = "metal")]
impl From<super::metal::MetalRenderError> for ProbeError {
    fn from(err: super::metal::MetalRenderError) -> Self {
        Self::Render(format!("Metal probe render failed: {err}"))
    }
}

pub fn run_probe(
    config: &ProbeConfig,
    loaded_splats: &[Splat],
) -> Result<ProbeRunResult, ProbeError> {
    validate_config(config)?;

    let splats = config.case.splats(loaded_splats);
    let camera = config.camera.to_camera();

    let mut cpu_frames = Vec::new();
    let mut cpu_framebuffers = Vec::new();
    if config.backend.renders_cpu() {
        let (artifacts, framebuffers) = render_cpu_probe_frames(config, &splats, &camera)?;
        cpu_frames = artifacts;
        cpu_framebuffers = framebuffers;
    }

    #[cfg(feature = "metal")]
    let (mut metal_frames, mut metal_framebuffers) = (Vec::new(), Vec::new());
    #[cfg(feature = "metal")]
    if config.backend.renders_metal() {
        let (artifacts, framebuffers) = render_metal_probe_frames(config, &splats, &camera)?;
        metal_frames = artifacts;
        metal_framebuffers = framebuffers;
    }
    #[cfg(not(feature = "metal"))]
    let metal_frames = Vec::new();
    #[cfg(not(feature = "metal"))]
    let metal_framebuffers: Vec<Vec<[u8; 3]>> = Vec::new();

    #[cfg(feature = "metal")]
    let (mut diff_frames, mut diff_summary_path) = (Vec::new(), None);
    #[cfg(feature = "metal")]
    if config.backend.compares_cpu_to_metal() {
        let diff_dir = config.out_dir.join("diff");
        let inspect_diff_dir = config.out_dir.join("inspect").join("diff");
        fs::create_dir_all(&diff_dir)?;
        fs::create_dir_all(&inspect_diff_dir)?;
        diff_frames = write_diff_artifacts(
            &diff_dir,
            &inspect_diff_dir,
            config.width,
            config.height,
            config.inspect_scale,
            &cpu_framebuffers,
            &metal_framebuffers,
        )?;
        let summary_path = diff_dir.join("summary.json");
        write_diff_summary_json(&summary_path, &diff_frames)?;
        diff_summary_path = Some(summary_path);
    }
    #[cfg(not(feature = "metal"))]
    let diff_frames = Vec::new();
    #[cfg(not(feature = "metal"))]
    let diff_summary_path = None;

    let contact_sheet_path = config.out_dir.join("contact_sheet.ppm");
    let inspect_contact_sheet_path = config
        .out_dir
        .join("inspect")
        .join(format!("contact_sheet.x{}.png", config.inspect_scale));
    let manifest_path = config.out_dir.join("probe_manifest.json");
    if let Some(frame) = cpu_framebuffers
        .first()
        .or_else(|| metal_framebuffers.first())
    {
        write_ppm(&contact_sheet_path, config.width, config.height, frame)?;
        if let Some(parent) = inspect_contact_sheet_path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_png_inspection(
            &inspect_contact_sheet_path,
            config.width,
            config.height,
            config.inspect_scale,
            frame,
        )?;
    }
    write_manifest_json(
        &manifest_path,
        config,
        &cpu_frames,
        &metal_frames,
        &diff_frames,
        diff_summary_path.as_deref(),
        &contact_sheet_path,
        &inspect_contact_sheet_path,
    )?;

    Ok(ProbeRunResult {
        manifest_path,
        contact_sheet_path,
        inspect_contact_sheet_path,
        cpu_frames,
        metal_frames,
        diff_frames,
        diff_summary_path,
    })
}

fn render_cpu_probe_frames(
    config: &ProbeConfig,
    splats: &[Splat],
    camera: &Camera,
) -> Result<(Vec<ProbeFrameArtifact>, Vec<Vec<[u8; 3]>>), ProbeError> {
    let cpu_dir = config.out_dir.join("cpu");
    let inspect_cpu_dir = config.out_dir.join("inspect").join("cpu");
    fs::create_dir_all(&cpu_dir)?;
    fs::create_dir_all(&inspect_cpu_dir)?;
    let terminal_dir = config.out_dir.join("terminal");
    if config.terminal_artifacts {
        fs::create_dir_all(&terminal_dir)?;
    }

    for _ in 0..config.warmup_frames {
        let _ = render_cpu_frame(splats, camera, config.width, config.height);
    }

    let mut artifacts = Vec::with_capacity(config.frames);
    let mut framebuffers = Vec::with_capacity(config.frames);
    for frame_idx in 0..config.frames {
        let frame = render_cpu_frame(splats, camera, config.width, config.height);
        let stats = compute_frame_stats(&frame, config.width, config.height);

        let frame_path = cpu_dir.join(format!("frame_{frame_idx:03}.ppm"));
        let inspect_png_path = inspect_cpu_dir.join(format!(
            "frame_{frame_idx:03}.x{}.png",
            config.inspect_scale
        ));
        let stats_path = cpu_dir.join(format!("frame_{frame_idx:03}.json"));
        write_ppm(&frame_path, config.width, config.height, &frame)?;
        write_png_inspection(
            &inspect_png_path,
            config.width,
            config.height,
            config.inspect_scale,
            &frame,
        )?;
        write_frame_stats_json(&stats_path, &stats)?;
        if config.terminal_artifacts {
            write_halfblock_ansi(
                &terminal_dir.join(format!("cpu_frame_{frame_idx:03}.ansi.txt")),
                config.width,
                config.height,
                &frame,
            )?;
        }

        artifacts.push(ProbeFrameArtifact {
            frame_path,
            inspect_png_path,
            stats_path,
            stats,
        });
        framebuffers.push(frame);
    }

    Ok((artifacts, framebuffers))
}

#[cfg(feature = "metal")]
fn render_metal_probe_frames(
    config: &ProbeConfig,
    splats: &[Splat],
    camera: &Camera,
) -> Result<(Vec<ProbeMetalFrameArtifact>, Vec<Vec<[u8; 3]>>), ProbeError> {
    let metal_dir = config.out_dir.join("metal");
    let inspect_metal_dir = config.out_dir.join("inspect").join("metal");
    fs::create_dir_all(&metal_dir)?;
    fs::create_dir_all(&inspect_metal_dir)?;
    let terminal_dir = config.out_dir.join("terminal");
    if config.terminal_artifacts {
        fs::create_dir_all(&terminal_dir)?;
    }

    let mut backend = super::metal::MetalBackend::new(splats.len())?;
    backend.upload_splats(splats)?;
    for _ in 0..config.warmup_frames {
        backend.render(camera, config.width, config.height, splats.len())?;
        let _ = backend.framebuffer_slice();
    }

    let mut artifacts = Vec::with_capacity(config.frames);
    let mut framebuffers = Vec::with_capacity(config.frames);
    for frame_idx in 0..config.frames {
        backend.render(camera, config.width, config.height, splats.len())?;
        let packed = backend.framebuffer_slice();
        let frame = normalize_metal_packed_pixels(packed);
        let stats = compute_frame_stats(&frame, config.width, config.height);

        let frame_path = metal_dir.join(format!("frame_{frame_idx:03}.ppm"));
        let inspect_png_path = inspect_metal_dir.join(format!(
            "frame_{frame_idx:03}.x{}.png",
            config.inspect_scale
        ));
        let stats_path = metal_dir.join(format!("frame_{frame_idx:03}.json"));
        let packed_u32le_path = metal_dir.join(format!("frame_{frame_idx:03}.packed_u32le.bin"));
        write_ppm(&frame_path, config.width, config.height, &frame)?;
        write_png_inspection(
            &inspect_png_path,
            config.width,
            config.height,
            config.inspect_scale,
            &frame,
        )?;
        write_frame_stats_json(&stats_path, &stats)?;
        write_packed_u32le(&packed_u32le_path, packed)?;
        if config.terminal_artifacts {
            write_halfblock_ansi(
                &terminal_dir.join(format!("metal_frame_{frame_idx:03}.ansi.txt")),
                config.width,
                config.height,
                &frame,
            )?;
        }

        artifacts.push(ProbeMetalFrameArtifact {
            frame_path,
            inspect_png_path,
            stats_path,
            packed_u32le_path,
            stats,
        });
        framebuffers.push(frame);
    }

    Ok((artifacts, framebuffers))
}

pub fn render_cpu_frame(
    splats: &[Splat],
    camera: &Camera,
    width: usize,
    height: usize,
) -> Vec<[u8; 3]> {
    let len = width.saturating_mul(height);
    let mut render_state = RenderState {
        framebuffer: vec![[0, 0, 0]; len],
        alpha_buffer: vec![0.0; len],
        depth_buffer: vec![f32::INFINITY; len],
        width,
        height,
    };
    let mut projected_splats = Vec::<ProjectedSplat>::new();
    let mut visible_count = 0usize;

    super::pipeline::resize_render_state(&mut render_state, width, height);
    super::pipeline::clear_framebuffer(&mut render_state);
    super::pipeline::project_and_cull_splats(
        splats,
        &mut projected_splats,
        camera,
        width,
        height,
        &mut visible_count,
    );
    sort_by_depth(&mut projected_splats);
    super::rasterizer::rasterize_splats(&projected_splats, &mut render_state, width, height);

    render_state.framebuffer
}

pub fn compute_frame_stats(frame: &[[u8; 3]], width: usize, height: usize) -> ProbeFrameStats {
    let mut nonblack_pixels = 0usize;
    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;
    let mut lumas = Vec::with_capacity(frame.len());
    let mut bbox: Option<ProbeBoundingBox> = None;
    let mut checksum = fnv1a64_update(FNV1A64_OFFSET, width as u64);
    checksum = fnv1a64_update(checksum, height as u64);

    for (idx, pixel) in frame.iter().enumerate() {
        let [r, g, b] = *pixel;
        sum_r += r as u64;
        sum_g += g as u64;
        sum_b += b as u64;

        let luma = luma_u8(r, g, b);
        lumas.push(luma);
        checksum = fnv1a64_update(checksum, r as u64);
        checksum = fnv1a64_update(checksum, g as u64);
        checksum = fnv1a64_update(checksum, b as u64);

        if r != 0 || g != 0 || b != 0 {
            nonblack_pixels += 1;
            let x = if width == 0 { 0 } else { idx % width };
            let y = if width == 0 { 0 } else { idx / width };
            match bbox.as_mut() {
                Some(existing) => {
                    existing.min_x = existing.min_x.min(x);
                    existing.min_y = existing.min_y.min(y);
                    existing.max_x = existing.max_x.max(x);
                    existing.max_y = existing.max_y.max(y);
                }
                None => {
                    bbox = Some(ProbeBoundingBox {
                        min_x: x,
                        min_y: y,
                        max_x: x,
                        max_y: y,
                    });
                }
            }
        }
    }

    lumas.sort_unstable();
    let luma_min = lumas.first().copied().unwrap_or(0);
    let luma_max = lumas.last().copied().unwrap_or(0);
    let luma_sum: u64 = lumas.iter().map(|&v| v as u64).sum();
    let luma_mean = if lumas.is_empty() {
        0.0
    } else {
        luma_sum as f64 / lumas.len() as f64
    };
    let luma_p95 = percentile_nearest_rank(&lumas, 95);

    ProbeFrameStats {
        width,
        height,
        nonblack_pixels,
        sum_r,
        sum_g,
        sum_b,
        luma_min,
        luma_max,
        luma_mean,
        luma_p95,
        bbox,
        checksum,
    }
}

pub fn write_ppm(path: &Path, width: usize, height: usize, frame: &[[u8; 3]]) -> io::Result<()> {
    if frame.len() != width.saturating_mul(height) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PPM frame length does not match dimensions",
        ));
    }

    let mut file = fs::File::create(path)?;
    write!(file, "P6\n{} {}\n255\n", width, height)?;
    for [r, g, b] in frame {
        file.write_all(&[*r, *g, *b])?;
    }
    Ok(())
}

pub fn write_png_inspection(
    path: &Path,
    width: usize,
    height: usize,
    inspect_scale: usize,
    frame: &[[u8; 3]],
) -> Result<(), ProbeError> {
    let source_pixels = checked_pixel_count(width, height, "source PNG inspection frame")?;
    if frame.len() != source_pixels {
        return Err(ProbeError::InvalidConfig(format!(
            "PNG inspection frame length {} does not match dimensions {}x{}",
            frame.len(),
            width,
            height
        )));
    }

    let (inspect_width, inspect_height) = checked_inspect_dimensions(width, height, inspect_scale)?;
    let inspect_pixels =
        checked_pixel_count(inspect_width, inspect_height, "scaled PNG inspection frame")?;
    let byte_len = inspect_pixels.checked_mul(3).ok_or_else(|| {
        ProbeError::InvalidConfig(format!(
            "PNG inspection artifact {}x{} exceeds addressable RGB byte length",
            inspect_width, inspect_height
        ))
    })?;

    let mut bytes = Vec::with_capacity(byte_len);
    for y in 0..height {
        let row_start = y * width;
        for _ in 0..inspect_scale {
            for x in 0..width {
                let pixel = frame[row_start + x];
                for _ in 0..inspect_scale {
                    bytes.extend_from_slice(&pixel);
                }
            }
        }
    }

    let file = fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, inspect_width as u32, inspect_height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|err| {
        ProbeError::Render(format!(
            "failed to write PNG inspection header '{}': {err}",
            path.display()
        ))
    })?;
    writer.write_image_data(&bytes).map_err(|err| {
        ProbeError::Render(format!(
            "failed to write PNG inspection data '{}': {err}",
            path.display()
        ))
    })?;
    Ok(())
}

pub fn write_halfblock_ansi(
    path: &Path,
    width: usize,
    height: usize,
    frame: &[[u8; 3]],
) -> io::Result<()> {
    if frame.len() != width.saturating_mul(height) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "ANSI frame length does not match dimensions",
        ));
    }

    let mut file = fs::File::create(path)?;
    let rows = height.div_ceil(2);
    for row in 0..rows {
        let top_y = row * 2;
        let bottom_y = top_y + 1;
        for x in 0..width {
            let top = frame[top_y * width + x];
            let bottom = if bottom_y < height {
                frame[bottom_y * width + x]
            } else {
                [0, 0, 0]
            };
            write!(
                file,
                "\x1b[48;2;{};{};{}m\x1b[38;2;{};{};{}m{}",
                top[0],
                top[1],
                top[2],
                bottom[0],
                bottom[1],
                bottom[2],
                super::HALF_BLOCK
            )?;
        }
        writeln!(file, "\x1b[0m")?;
    }
    Ok(())
}

pub fn normalize_metal_packed_pixels(packed: &[u32]) -> Vec<[u8; 3]> {
    packed
        .iter()
        .map(|&p| {
            [
                (p & 0xff) as u8,
                ((p >> 8) & 0xff) as u8,
                ((p >> 16) & 0xff) as u8,
            ]
        })
        .collect()
}

pub fn compute_diff_metrics(
    reference: &[[u8; 3]],
    candidate: &[[u8; 3]],
    width: usize,
    height: usize,
) -> ProbeDiffMetrics {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "diff frame lengths must match"
    );
    assert_eq!(
        reference.len(),
        width.saturating_mul(height),
        "diff frame length must match dimensions"
    );

    let mut all_abs = Vec::with_capacity(reference.len().saturating_mul(3));
    let mut mismatch_pixels = 0usize;
    let mut sum_abs_r = 0u64;
    let mut sum_abs_g = 0u64;
    let mut sum_abs_b = 0u64;
    let mut max_abs = 0u8;

    for (a, b) in reference.iter().zip(candidate.iter()) {
        let dr = abs_diff_u8(a[0], b[0]);
        let dg = abs_diff_u8(a[1], b[1]);
        let db = abs_diff_u8(a[2], b[2]);
        if dr != 0 || dg != 0 || db != 0 {
            mismatch_pixels += 1;
        }
        sum_abs_r += dr as u64;
        sum_abs_g += dg as u64;
        sum_abs_b += db as u64;
        max_abs = max_abs.max(dr).max(dg).max(db);
        all_abs.push(dr);
        all_abs.push(dg);
        all_abs.push(db);
    }

    all_abs.sort_unstable();
    let total_channels = all_abs.len();
    let total_abs = sum_abs_r + sum_abs_g + sum_abs_b;
    let mean_abs = if total_channels == 0 {
        0.0
    } else {
        total_abs as f64 / total_channels as f64
    };
    let pixel_count = reference.len();
    let mismatch_ratio = if pixel_count == 0 {
        0.0
    } else {
        mismatch_pixels as f64 / pixel_count as f64
    };
    let p95_abs = percentile_nearest_rank(&all_abs, 95);

    let mut metrics = ProbeDiffMetrics {
        width,
        height,
        mean_abs,
        max_abs,
        p95_abs,
        mismatch_pixels,
        mismatch_ratio,
        sum_abs_r,
        sum_abs_g,
        sum_abs_b,
        classification: ProbeDiffClassification::Mismatch,
    };
    metrics.classification = classify_diff(reference, candidate, &metrics);
    metrics
}

fn make_diff_frame(reference: &[[u8; 3]], candidate: &[[u8; 3]]) -> Vec<[u8; 3]> {
    reference
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| {
            [
                abs_diff_u8(a[0], b[0]),
                abs_diff_u8(a[1], b[1]),
                abs_diff_u8(a[2], b[2]),
            ]
        })
        .collect()
}

fn validate_config(config: &ProbeConfig) -> Result<(), ProbeError> {
    if config.width == 0 || config.height == 0 {
        return Err(ProbeError::InvalidConfig(
            "probe dimensions must be non-zero".to_string(),
        ));
    }
    if config.inspect_scale == 0 {
        return Err(ProbeError::InvalidConfig(
            "--probe-inspect-scale must be greater than 0".to_string(),
        ));
    }
    if !config.camera.fov.is_finite()
        || config.camera.fov <= 0.0
        || config.camera.fov >= std::f32::consts::PI
    {
        return Err(ProbeError::InvalidConfig(
            "--probe-fov-deg must be finite and greater than 0 and less than 180".to_string(),
        ));
    }
    checked_inspect_dimensions(config.width, config.height, config.inspect_scale)?;
    if config.frames == 0 {
        return Err(ProbeError::InvalidConfig(
            "probe frame count must be non-zero".to_string(),
        ));
    }
    Ok(())
}

fn checked_inspect_dimensions(
    width: usize,
    height: usize,
    inspect_scale: usize,
) -> Result<(usize, usize), ProbeError> {
    if inspect_scale == 0 {
        return Err(ProbeError::InvalidConfig(
            "--probe-inspect-scale must be greater than 0".to_string(),
        ));
    }

    let inspect_width = width.checked_mul(inspect_scale).ok_or_else(|| {
        ProbeError::InvalidConfig(format!(
            "inspect width overflow: --probe-size width {width} times --probe-inspect-scale {inspect_scale}"
        ))
    })?;
    let inspect_height = height.checked_mul(inspect_scale).ok_or_else(|| {
        ProbeError::InvalidConfig(format!(
            "inspect height overflow: --probe-size height {height} times --probe-inspect-scale {inspect_scale}"
        ))
    })?;
    if inspect_width > u32::MAX as usize || inspect_height > u32::MAX as usize {
        return Err(ProbeError::InvalidConfig(format!(
            "PNG inspection dimensions {}x{} exceed PNG u32 limits; reduce --probe-size or --probe-inspect-scale",
            inspect_width, inspect_height
        )));
    }
    Ok((inspect_width, inspect_height))
}

fn checked_pixel_count(width: usize, height: usize, label: &str) -> Result<usize, ProbeError> {
    width.checked_mul(height).ok_or_else(|| {
        ProbeError::InvalidConfig(format!(
            "{label} dimensions {width}x{height} overflow usize"
        ))
    })
}

fn write_packed_u32le(path: &Path, packed: &[u32]) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    for pixel in packed {
        file.write_all(&pixel.to_le_bytes())?;
    }
    Ok(())
}

fn write_diff_artifacts(
    diff_dir: &Path,
    inspect_diff_dir: &Path,
    width: usize,
    height: usize,
    inspect_scale: usize,
    cpu_framebuffers: &[Vec<[u8; 3]>],
    metal_framebuffers: &[Vec<[u8; 3]>],
) -> Result<Vec<ProbeDiffFrameArtifact>, ProbeError> {
    if cpu_framebuffers.len() != metal_framebuffers.len() {
        return Err(ProbeError::Render(format!(
            "cannot diff {} CPU frames against {} Metal frames",
            cpu_framebuffers.len(),
            metal_framebuffers.len()
        )));
    }

    let mut diff_frames = Vec::with_capacity(cpu_framebuffers.len());
    for (frame_idx, (cpu_frame, metal_frame)) in cpu_framebuffers
        .iter()
        .zip(metal_framebuffers.iter())
        .enumerate()
    {
        if cpu_frame.len() != metal_frame.len() {
            return Err(ProbeError::Render(format!(
                "cannot diff frame {frame_idx}: CPU has {} pixels, Metal has {} pixels",
                cpu_frame.len(),
                metal_frame.len()
            )));
        }
        let diff_frame = make_diff_frame(cpu_frame, metal_frame);
        let metrics = compute_diff_metrics(cpu_frame, metal_frame, width, height);
        let frame_path = diff_dir.join(format!("cpu_vs_metal_frame_{frame_idx:03}.ppm"));
        let inspect_png_path = inspect_diff_dir.join(format!(
            "cpu_vs_metal_frame_{frame_idx:03}.x{inspect_scale}.png"
        ));
        write_ppm(&frame_path, width, height, &diff_frame)?;
        write_png_inspection(&inspect_png_path, width, height, inspect_scale, &diff_frame)?;
        diff_frames.push(ProbeDiffFrameArtifact {
            frame_path,
            inspect_png_path,
            metrics,
        });
    }

    Ok(diff_frames)
}

fn write_frame_stats_json(path: &Path, stats: &ProbeFrameStats) -> io::Result<()> {
    fs::write(path, frame_stats_json(stats))
}

fn write_manifest_json(
    path: &Path,
    config: &ProbeConfig,
    cpu_frames: &[ProbeFrameArtifact],
    metal_frames: &[ProbeMetalFrameArtifact],
    diff_frames: &[ProbeDiffFrameArtifact],
    diff_summary_path: Option<&Path>,
    contact_sheet_path: &Path,
    inspect_contact_sheet_path: &Path,
) -> Result<(), ProbeError> {
    let (inspect_width, inspect_height) =
        checked_inspect_dimensions(config.width, config.height, config.inspect_scale)?;
    let frames_json = cpu_frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| {
            format!(
                concat!(
                    "{{",
                    "\"index\":{},",
                    "\"frame\":\"{}\",",
                    "\"inspect_png\":\"{}\",",
                    "\"stats\":\"{}\",",
                    "\"metrics\":{}",
                    "}}"
                ),
                idx,
                json_escape(&display_path(&frame.frame_path)),
                json_escape(&display_path(&frame.inspect_png_path)),
                json_escape(&display_path(&frame.stats_path)),
                frame_stats_json(&frame.stats)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n      ");
    let metal_frames_json = metal_frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| {
            format!(
                concat!(
                    "{{",
                    "\"index\":{},",
                    "\"frame\":\"{}\",",
                    "\"inspect_png\":\"{}\",",
                    "\"stats\":\"{}\",",
                    "\"packed_u32le\":\"{}\",",
                    "\"metrics\":{}",
                    "}}"
                ),
                idx,
                json_escape(&display_path(&frame.frame_path)),
                json_escape(&display_path(&frame.inspect_png_path)),
                json_escape(&display_path(&frame.stats_path)),
                json_escape(&display_path(&frame.packed_u32le_path)),
                frame_stats_json(&frame.stats)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n      ");
    let diff_frames_json = diff_frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| {
            format!(
                concat!(
                    "{{",
                    "\"index\":{},",
                    "\"frame\":\"{}\",",
                    "\"inspect_png\":\"{}\",",
                    "\"metrics\":{}",
                    "}}"
                ),
                idx,
                json_escape(&display_path(&frame.frame_path)),
                json_escape(&display_path(&frame.inspect_png_path)),
                diff_metrics_json(&frame.metrics)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n      ");
    let diff_summary = match diff_summary_path {
        Some(path) => format!("\"{}\"", json_escape(&display_path(path))),
        None => "null".to_string(),
    };
    let json = format!(
        concat!(
            "{{\n",
            "  \"version\": 1,\n",
            "  \"case\": \"{}\",\n",
            "  \"backend\": \"{}\",\n",
            "  \"width\": {},\n",
            "  \"height\": {},\n",
            "  \"inspect_scale\": {},\n",
            "  \"inspect_width\": {},\n",
            "  \"inspect_height\": {},\n",
            "  \"frames\": {},\n",
            "  \"warmup_frames\": {},\n",
            "  \"terminal_artifacts\": {},\n",
            "  \"artifacts\": {{\n",
            "    \"contact_sheet\": \"{}\",\n",
            "    \"inspect_contact_sheet\": \"{}\",\n",
            "    \"diff_summary\": {}\n",
            "  }},\n",
            "  \"cpu_frames\": [\n      {}\n  ],\n",
            "  \"metal_frames\": [\n      {}\n  ],\n",
            "  \"diff_frames\": [\n      {}\n  ]\n",
            "}}\n"
        ),
        config.case,
        config.backend,
        config.width,
        config.height,
        config.inspect_scale,
        inspect_width,
        inspect_height,
        config.frames,
        config.warmup_frames,
        config.terminal_artifacts,
        json_escape(&display_path(contact_sheet_path)),
        json_escape(&display_path(inspect_contact_sheet_path)),
        diff_summary,
        frames_json,
        metal_frames_json,
        diff_frames_json
    );
    fs::write(path, json)?;
    Ok(())
}

fn write_diff_summary_json(path: &Path, diff_frames: &[ProbeDiffFrameArtifact]) -> io::Result<()> {
    let frames_json = diff_frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| {
            format!(
                concat!(
                    "{{",
                    "\"index\":{},",
                    "\"frame\":\"{}\",",
                    "\"metrics\":{}",
                    "}}"
                ),
                idx,
                json_escape(&display_path(&frame.frame_path)),
                diff_metrics_json(&frame.metrics)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n    ");

    let overall = summarize_diff_frames(diff_frames);
    let json = format!(
        concat!(
            "{{\n",
            "  \"version\": 1,\n",
            "  \"comparison\": \"cpu_vs_metal\",\n",
            "  \"frame_count\": {},\n",
            "  \"overall\": {},\n",
            "  \"frames\": [\n    {}\n  ]\n",
            "}}\n"
        ),
        diff_frames.len(),
        diff_summary_metrics_json(&overall),
        frames_json
    );
    fs::write(path, json)
}

fn frame_stats_json(stats: &ProbeFrameStats) -> String {
    let bbox = match &stats.bbox {
        Some(bbox) => format!(
            "{{\"min_x\":{},\"min_y\":{},\"max_x\":{},\"max_y\":{}}}",
            bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
        ),
        None => "null".to_string(),
    };

    format!(
        concat!(
            "{{",
            "\"width\":{},",
            "\"height\":{},",
            "\"nonblack_pixels\":{},",
            "\"sum_r\":{},",
            "\"sum_g\":{},",
            "\"sum_b\":{},",
            "\"luma_min\":{},",
            "\"luma_max\":{},",
            "\"luma_mean\":{:.6},",
            "\"luma_p95\":{},",
            "\"bbox\":{},",
            "\"checksum\":\"{:016x}\"",
            "}}"
        ),
        stats.width,
        stats.height,
        stats.nonblack_pixels,
        stats.sum_r,
        stats.sum_g,
        stats.sum_b,
        stats.luma_min,
        stats.luma_max,
        stats.luma_mean,
        stats.luma_p95,
        bbox,
        stats.checksum
    )
}

fn diff_metrics_json(metrics: &ProbeDiffMetrics) -> String {
    format!(
        concat!(
            "{{",
            "\"width\":{},",
            "\"height\":{},",
            "\"mean_abs\":{:.6},",
            "\"max_abs\":{},",
            "\"p95_abs\":{},",
            "\"mismatch_pixels\":{},",
            "\"mismatch_ratio\":{:.6},",
            "\"sum_abs_r\":{},",
            "\"sum_abs_g\":{},",
            "\"sum_abs_b\":{},",
            "\"classification\":\"{}\"",
            "}}"
        ),
        metrics.width,
        metrics.height,
        metrics.mean_abs,
        metrics.max_abs,
        metrics.p95_abs,
        metrics.mismatch_pixels,
        metrics.mismatch_ratio,
        metrics.sum_abs_r,
        metrics.sum_abs_g,
        metrics.sum_abs_b,
        metrics.classification
    )
}

#[derive(Debug, Clone, PartialEq)]
struct ProbeDiffSummaryMetrics {
    mean_abs: f64,
    max_abs: u8,
    p95_abs: u8,
    mismatch_pixels: usize,
    mismatch_ratio: f64,
    sum_abs_r: u64,
    sum_abs_g: u64,
    sum_abs_b: u64,
    classification: ProbeDiffClassification,
}

fn summarize_diff_frames(diff_frames: &[ProbeDiffFrameArtifact]) -> ProbeDiffSummaryMetrics {
    let mut sum_abs_r = 0u64;
    let mut sum_abs_g = 0u64;
    let mut sum_abs_b = 0u64;
    let mut total_pixels = 0usize;
    let mut mismatch_pixels = 0usize;
    let mut max_abs = 0u8;
    let mut worst_p95_abs = 0u8;
    let mut classification = ProbeDiffClassification::Pass;

    for frame in diff_frames {
        let metrics = &frame.metrics;
        sum_abs_r += metrics.sum_abs_r;
        sum_abs_g += metrics.sum_abs_g;
        sum_abs_b += metrics.sum_abs_b;
        total_pixels += metrics.width.saturating_mul(metrics.height);
        mismatch_pixels += metrics.mismatch_pixels;
        max_abs = max_abs.max(metrics.max_abs);
        worst_p95_abs = worst_p95_abs.max(metrics.p95_abs);
        classification = worse_classification(classification, metrics.classification);
    }

    let total_channels = total_pixels.saturating_mul(3);
    let total_abs = sum_abs_r + sum_abs_g + sum_abs_b;
    ProbeDiffSummaryMetrics {
        mean_abs: if total_channels == 0 {
            0.0
        } else {
            total_abs as f64 / total_channels as f64
        },
        max_abs,
        p95_abs: worst_p95_abs,
        mismatch_pixels,
        mismatch_ratio: if total_pixels == 0 {
            0.0
        } else {
            mismatch_pixels as f64 / total_pixels as f64
        },
        sum_abs_r,
        sum_abs_g,
        sum_abs_b,
        classification,
    }
}

fn diff_summary_metrics_json(metrics: &ProbeDiffSummaryMetrics) -> String {
    format!(
        concat!(
            "{{",
            "\"mean_abs\":{:.6},",
            "\"max_abs\":{},",
            "\"p95_abs\":{},",
            "\"mismatch_pixels\":{},",
            "\"mismatch_ratio\":{:.6},",
            "\"sum_abs_r\":{},",
            "\"sum_abs_g\":{},",
            "\"sum_abs_b\":{},",
            "\"classification\":\"{}\"",
            "}}"
        ),
        metrics.mean_abs,
        metrics.max_abs,
        metrics.p95_abs,
        metrics.mismatch_pixels,
        metrics.mismatch_ratio,
        metrics.sum_abs_r,
        metrics.sum_abs_g,
        metrics.sum_abs_b,
        metrics.classification
    )
}

fn worse_classification(
    current: ProbeDiffClassification,
    next: ProbeDiffClassification,
) -> ProbeDiffClassification {
    if classification_rank(next) > classification_rank(current) {
        next
    } else {
        current
    }
}

fn classification_rank(classification: ProbeDiffClassification) -> u8 {
    match classification {
        ProbeDiffClassification::Pass => 0,
        ProbeDiffClassification::ChannelSwap => 1,
        ProbeDiffClassification::Blank => 2,
        ProbeDiffClassification::Mismatch => 3,
    }
}

fn classify_diff(
    reference: &[[u8; 3]],
    candidate: &[[u8; 3]],
    metrics: &ProbeDiffMetrics,
) -> ProbeDiffClassification {
    if metrics.max_abs == 0 {
        return ProbeDiffClassification::Pass;
    }
    if metrics.max_abs <= 2 && metrics.mean_abs <= 0.10 && metrics.p95_abs <= 1 {
        return ProbeDiffClassification::Pass;
    }

    let reference_nonblack = count_nonblack(reference);
    let candidate_nonblack = count_nonblack(candidate);
    if reference_nonblack == 0 || candidate_nonblack == 0 {
        return ProbeDiffClassification::Blank;
    }

    if is_exact_channel_swap(reference, candidate) {
        return ProbeDiffClassification::ChannelSwap;
    }

    ProbeDiffClassification::Mismatch
}

fn count_nonblack(frame: &[[u8; 3]]) -> usize {
    frame
        .iter()
        .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        .count()
}

fn is_exact_channel_swap(reference: &[[u8; 3]], candidate: &[[u8; 3]]) -> bool {
    const PERMUTATIONS: [[usize; 3]; 5] = [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    PERMUTATIONS.iter().any(|permutation| {
        reference.iter().zip(candidate.iter()).all(|(a, b)| {
            a[0] == b[permutation[0]] && a[1] == b[permutation[1]] && a[2] == b[permutation[2]]
        })
    })
}

fn abs_diff_u8(a: u8, b: u8) -> u8 {
    if a >= b {
        a - b
    } else {
        b - a
    }
}

fn synthetic_channels_splats() -> Vec<Splat> {
    vec![
        probe_splat(Vec3::new(-0.55, 0.0, 0.0), [255, 0, 0], 0.95, 0.18),
        probe_splat(Vec3::new(0.0, 0.0, 0.0), [0, 255, 0], 0.95, 0.18),
        probe_splat(Vec3::new(0.55, 0.0, 0.0), [0, 0, 255], 0.95, 0.18),
    ]
}

fn synthetic_depth_splats() -> Vec<Splat> {
    vec![
        probe_splat(Vec3::new(-0.35, 0.0, 0.9), [220, 50, 40], 0.85, 0.17),
        probe_splat(Vec3::new(0.0, 0.0, 0.0), [40, 220, 80], 0.85, 0.18),
        probe_splat(Vec3::new(0.35, 0.0, -0.9), [50, 90, 240], 0.85, 0.20),
    ]
}

fn synthetic_tile_boundary_splats() -> Vec<Splat> {
    let mut splats = Vec::new();
    let colors = [
        [255, 210, 40],
        [40, 210, 255],
        [255, 60, 160],
        [120, 255, 90],
    ];
    for (idx, x) in [-0.9, -0.3, 0.3, 0.9].iter().enumerate() {
        splats.push(probe_splat(
            Vec3::new(*x, -0.45, 0.0),
            colors[idx % colors.len()],
            0.8,
            0.16,
        ));
        splats.push(probe_splat(
            Vec3::new(*x, 0.45, 0.0),
            colors[(idx + 1) % colors.len()],
            0.8,
            0.16,
        ));
    }
    splats
}

fn probe_splat(position: Vec3, color: [u8; 3], opacity: f32, scale: f32) -> Splat {
    Splat {
        position,
        color,
        opacity,
        scale: Vec3::new(scale, scale, scale),
        rotation: [1.0, 0.0, 0.0, 0.0],
    }
}

fn percentile_nearest_rank(sorted_values: &[u8], percentile: usize) -> u8 {
    if sorted_values.is_empty() {
        return 0;
    }
    let len = sorted_values.len();
    let rank = (percentile * len).div_ceil(100).max(1);
    sorted_values[rank - 1]
}

fn luma_u8(r: u8, g: u8, b: u8) -> u8 {
    let value = 299u32 * r as u32 + 587u32 * g as u32 + 114u32 * b as u32;
    ((value + 500) / 1000) as u8
}

const FNV1A64_OFFSET: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x100000001b3;

fn fnv1a64_update(mut state: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        state ^= byte as u64;
        state = state.wrapping_mul(FNV1A64_PRIME);
    }
    state
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn probe_case_parses_and_formats() {
        assert_eq!("blank".parse::<ProbeCase>().unwrap(), ProbeCase::Blank);
        assert_eq!(
            "channels".parse::<ProbeCase>().unwrap(),
            ProbeCase::Channels
        );
        assert_eq!("depth".parse::<ProbeCase>().unwrap(), ProbeCase::Depth);
        assert_eq!(
            "tile_boundary".parse::<ProbeCase>().unwrap(),
            ProbeCase::TileBoundary
        );
        assert_eq!(ProbeCase::TileBoundary.to_string(), "tile-boundary");
        assert!("missing".parse::<ProbeCase>().is_err());
    }

    #[test]
    fn probe_backend_parses_and_formats() {
        assert_eq!(
            "CPU".parse::<ProbeBackendSelection>().unwrap(),
            ProbeBackendSelection::Cpu
        );
        assert_eq!(ProbeBackendSelection::Cpu.to_string(), "cpu");
        #[cfg(feature = "metal")]
        {
            assert_eq!(
                "metal".parse::<ProbeBackendSelection>().unwrap(),
                ProbeBackendSelection::Metal
            );
            assert_eq!(
                "BOTH".parse::<ProbeBackendSelection>().unwrap(),
                ProbeBackendSelection::Both
            );
            assert_eq!(ProbeBackendSelection::Metal.to_string(), "metal");
            assert_eq!(ProbeBackendSelection::Both.to_string(), "both");
        }
        #[cfg(not(feature = "metal"))]
        {
            let err = "metal".parse::<ProbeBackendSelection>().unwrap_err();
            assert!(err.to_string().contains("--features metal"));
            let err = "both".parse::<ProbeBackendSelection>().unwrap_err();
            assert!(err.to_string().contains("--features metal"));
        }
        assert!("missing".parse::<ProbeBackendSelection>().is_err());
    }

    #[test]
    fn probe_blank_stats_are_zero() {
        let frame = vec![[0, 0, 0]; 12];
        let stats = compute_frame_stats(&frame, 4, 3);

        assert_eq!(stats.width, 4);
        assert_eq!(stats.height, 3);
        assert_eq!(stats.nonblack_pixels, 0);
        assert_eq!(stats.sum_r, 0);
        assert_eq!(stats.sum_g, 0);
        assert_eq!(stats.sum_b, 0);
        assert_eq!(stats.luma_min, 0);
        assert_eq!(stats.luma_max, 0);
        assert_eq!(stats.luma_mean, 0.0);
        assert_eq!(stats.luma_p95, 0);
        assert_eq!(stats.bbox, None);
    }

    #[test]
    fn probe_channel_stats_capture_rgb_sums_and_bbox() {
        let camera = ProbeCameraSpec::default().to_camera();
        let frame = render_cpu_frame(&synthetic_channels_splats(), &camera, 64, 48);
        let stats = compute_frame_stats(&frame, 64, 48);

        assert_eq!(stats.width, 64);
        assert_eq!(stats.height, 48);
        assert!(stats.nonblack_pixels > 300);
        assert!(stats.sum_r > 0);
        assert!(stats.sum_g > 0);
        assert!(stats.sum_b > 0);
        assert!(stats.luma_max > stats.luma_min);
        assert!(stats.luma_p95 > 0);
        let bbox = stats.bbox.as_ref().unwrap();
        assert!(bbox.min_x < bbox.max_x);
        assert!(bbox.min_y < bbox.max_y);
        assert!(bbox.max_x < stats.width);
        assert!(bbox.max_y < stats.height);
    }

    #[test]
    fn probe_cpu_output_is_deterministic() {
        let camera = ProbeCameraSpec::default().to_camera();
        let splats = synthetic_tile_boundary_splats();
        let frame_a = render_cpu_frame(&splats, &camera, 64, 48);
        let frame_b = render_cpu_frame(&splats, &camera, 64, 48);
        let stats_a = compute_frame_stats(&frame_a, 64, 48);
        let stats_b = compute_frame_stats(&frame_b, 64, 48);

        assert_eq!(frame_a, frame_b);
        assert_eq!(stats_a, stats_b);
        assert!(stats_a.nonblack_pixels > 0);
    }

    #[test]
    fn probe_ppm_writer_writes_binary_p6() {
        let dir = unique_temp_dir("probe_ppm_writer");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.ppm");
        let frame = vec![[255, 0, 0], [0, 255, 0]];

        write_ppm(&path, 2, 1, &frame).unwrap();

        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..11], b"P6\n2 1\n255\n");
        assert_eq!(&bytes[11..], &[255, 0, 0, 0, 255, 0]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn probe_png_inspection_writer_scales_dimensions_and_writes_png_header() {
        let dir = unique_temp_dir("probe_png_writer");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.x3.png");
        let frame = vec![[255, 0, 0], [0, 255, 0]];

        write_png_inspection(&path, 2, 1, 3, &frame).unwrap();

        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&bytes[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes(bytes[16..20].try_into().unwrap()), 6);
        assert_eq!(u32::from_be_bytes(bytes[20..24].try_into().unwrap()), 3);
        assert_eq!(bytes[24], 8);
        assert_eq!(bytes[25], 2);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn probe_config_validation_rejects_invalid_fov_and_inspect_scale() {
        let mut config = ProbeConfig::new(unique_temp_dir("probe_invalid_config"));
        config.inspect_scale = 0;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("--probe-inspect-scale"));

        config.inspect_scale = 1;
        config.camera.fov = 0.0;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("--probe-fov-deg"));

        config.camera.fov = std::f32::consts::PI;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("--probe-fov-deg"));

        config.camera.fov = std::f32::consts::FRAC_PI_2;
        validate_config(&config).unwrap();
    }

    #[test]
    fn probe_inspection_dimensions_guard_overflow_and_png_u32_limit() {
        let err = checked_inspect_dimensions(usize::MAX, 1, 2).unwrap_err();
        assert!(err.to_string().contains("inspect width overflow"));

        let err = checked_inspect_dimensions(u32::MAX as usize + 1, 1, 1).unwrap_err();
        assert!(err.to_string().contains("PNG u32 limits"));
    }

    #[test]
    fn probe_normalizes_metal_packed_pixels_low_byte_red() {
        let packed = [0x44332211, 0xff00aa55, 0x000000ff];
        let frame = normalize_metal_packed_pixels(&packed);

        assert_eq!(
            frame,
            vec![[0x11, 0x22, 0x33], [0x55, 0xaa, 0x00], [0xff, 0x00, 0x00]]
        );
    }

    #[test]
    fn probe_diff_metrics_classify_pass_mismatch_blank_and_channel_swap() {
        let reference = vec![[10, 20, 30], [0, 0, 0], [200, 10, 5], [1, 2, 3]];

        let pass = compute_diff_metrics(&reference, &reference, 2, 2);
        assert_eq!(pass.classification, ProbeDiffClassification::Pass);
        assert_eq!(pass.mean_abs, 0.0);
        assert_eq!(pass.max_abs, 0);
        assert_eq!(pass.mismatch_pixels, 0);

        let mismatch_candidate = vec![[10, 20, 30], [0, 1, 0], [190, 11, 5], [1, 2, 3]];
        let mismatch = compute_diff_metrics(&reference, &mismatch_candidate, 2, 2);
        assert_eq!(mismatch.classification, ProbeDiffClassification::Mismatch);
        assert_eq!(mismatch.max_abs, 10);
        assert_eq!(mismatch.mismatch_pixels, 2);
        assert_eq!(mismatch.sum_abs_r, 10);
        assert_eq!(mismatch.sum_abs_g, 2);
        assert_eq!(mismatch.sum_abs_b, 0);

        let blank_candidate = vec![[0, 0, 0]; 4];
        let blank = compute_diff_metrics(&reference, &blank_candidate, 2, 2);
        assert_eq!(blank.classification, ProbeDiffClassification::Blank);

        let swapped_candidate = reference
            .iter()
            .map(|pixel| [pixel[2], pixel[1], pixel[0]])
            .collect::<Vec<_>>();
        let swapped = compute_diff_metrics(&reference, &swapped_candidate, 2, 2);
        assert_eq!(swapped.classification, ProbeDiffClassification::ChannelSwap);
    }

    #[test]
    fn probe_diff_artifacts_write_ppm_and_summary_json() {
        let out_dir = unique_temp_dir("probe_diff_artifacts");
        let diff_dir = out_dir.join("diff");
        let inspect_diff_dir = out_dir.join("inspect").join("diff");
        fs::create_dir_all(&diff_dir).unwrap();
        fs::create_dir_all(&inspect_diff_dir).unwrap();
        let cpu_frames = vec![vec![[10, 20, 30], [0, 0, 0]]];
        let metal_frames = vec![vec![[10, 25, 30], [0, 0, 5]]];

        let diff_frames = write_diff_artifacts(
            &diff_dir,
            &inspect_diff_dir,
            2,
            1,
            2,
            &cpu_frames,
            &metal_frames,
        )
        .unwrap();
        let summary_path = diff_dir.join("summary.json");
        write_diff_summary_json(&summary_path, &diff_frames).unwrap();

        assert_eq!(diff_frames.len(), 1);
        assert!(diff_dir.join("cpu_vs_metal_frame_000.ppm").exists());
        assert!(inspect_diff_dir
            .join("cpu_vs_metal_frame_000.x2.png")
            .exists());
        let ppm = fs::read(diff_dir.join("cpu_vs_metal_frame_000.ppm")).unwrap();
        assert_eq!(&ppm[..11], b"P6\n2 1\n255\n");
        assert_eq!(&ppm[11..], &[0, 5, 0, 0, 0, 5]);

        let summary = fs::read_to_string(summary_path).unwrap();
        assert!(summary.contains("\"comparison\": \"cpu_vs_metal\""));
        assert!(summary.contains("\"mean_abs\""));
        assert!(summary.contains("\"classification\":\"mismatch\""));

        let _ = fs::remove_dir_all(out_dir);
    }

    #[test]
    fn probe_writes_packed_u32le_artifact() {
        let dir = unique_temp_dir("probe_packed_writer");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("packed.bin");

        write_packed_u32le(&path, &[0x11223344, 0xaabbccdd]).unwrap();

        let bytes = fs::read(path).unwrap();
        assert_eq!(bytes, vec![0x44, 0x33, 0x22, 0x11, 0xdd, 0xcc, 0xbb, 0xaa]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn probe_run_writes_cpu_artifacts() {
        let out_dir = unique_temp_dir("probe_run_writes");
        let mut config = ProbeConfig::new(&out_dir);
        config.width = 24;
        config.height = 16;
        config.case = ProbeCase::Blank;

        let result = run_probe(&config, &[]).unwrap();

        assert!(result.manifest_path.exists());
        assert!(result.contact_sheet_path.exists());
        assert!(result.inspect_contact_sheet_path.exists());
        assert_eq!(result.cpu_frames.len(), 1);
        assert!(result.cpu_frames[0].frame_path.exists());
        assert!(result.cpu_frames[0].inspect_png_path.exists());
        assert!(result.cpu_frames[0].stats_path.exists());
        assert_eq!(result.cpu_frames[0].stats.nonblack_pixels, 0);

        let manifest = fs::read_to_string(&result.manifest_path).unwrap();
        assert!(manifest.contains("\"case\": \"blank\""));
        assert!(manifest.contains("\"inspect_scale\": 1"));
        assert!(manifest.contains("\"inspect_width\": 24"));
        assert!(manifest.contains("\"inspect_height\": 16"));
        assert!(manifest.contains("\"inspect_contact_sheet\""));
        assert!(manifest.contains("\"inspect_png\""));
        assert!(manifest.contains("\"cpu_frames\""));

        let _ = fs::remove_dir_all(out_dir);
    }

    #[test]
    fn probe_run_honors_frame_count() {
        let out_dir = unique_temp_dir("probe_run_frames");
        let mut config = ProbeConfig::new(&out_dir);
        config.width = 24;
        config.height = 16;
        config.case = ProbeCase::Blank;
        config.frames = 2;
        config.warmup_frames = 1;

        let result = run_probe(&config, &[]).unwrap();

        assert_eq!(result.cpu_frames.len(), 2);
        assert!(out_dir.join("cpu/frame_000.ppm").exists());
        assert!(out_dir.join("cpu/frame_001.ppm").exists());
        assert!(out_dir.join("inspect/cpu/frame_000.x1.png").exists());
        assert!(out_dir.join("inspect/cpu/frame_001.x1.png").exists());
        let manifest = fs::read_to_string(&result.manifest_path).unwrap();
        assert!(manifest.contains("\"frames\": 2"));
        assert!(manifest.contains("\"warmup_frames\": 1"));

        let _ = fs::remove_dir_all(out_dir);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn probe_run_writes_both_blank_artifacts_when_device_is_available() {
        if metal::Device::system_default().is_none() {
            eprintln!("Skipping Metal probe test: no system-default Metal device.");
            return;
        }

        let out_dir = unique_temp_dir("probe_run_both_blank");
        let mut config = ProbeConfig::new(&out_dir);
        config.width = 8;
        config.height = 6;
        config.case = ProbeCase::Blank;
        config.backend = ProbeBackendSelection::Both;
        config.frames = 1;
        config.warmup_frames = 1;

        let result = run_probe(&config, &[]).unwrap();

        assert_eq!(result.cpu_frames.len(), 1);
        assert_eq!(result.metal_frames.len(), 1);
        assert_eq!(result.diff_frames.len(), 1);
        assert!(out_dir.join("cpu/frame_000.ppm").exists());
        assert!(out_dir.join("metal/frame_000.ppm").exists());
        assert!(out_dir.join("metal/frame_000.json").exists());
        assert!(out_dir.join("metal/frame_000.packed_u32le.bin").exists());
        assert!(out_dir.join("diff/cpu_vs_metal_frame_000.ppm").exists());
        assert!(out_dir.join("diff/summary.json").exists());
        assert_eq!(
            result.diff_frames[0].metrics.classification,
            ProbeDiffClassification::Pass
        );
        assert!(result.contact_sheet_path.exists());

        let _ = fs::remove_dir_all(out_dir);
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("tortuise_{}_{}", name, nanos))
    }
}
