#![cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]

use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant};

use crate::camera::{self, Camera};
use crate::math::Vec3;
use crate::sort::sort_by_depth;
use crate::splat::{ProjectedSplat, Splat};

use super::RenderState;

const PROBE_TILE_SIZE: usize = 16;
const PROBE_ALIGNMENT_WINDOW: i32 = 8;

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
    pub kitty_artifacts: bool,
    pub stage_telemetry: bool,
    pub timing: bool,
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
            kitty_artifacts: false,
            stage_telemetry: false,
            timing: false,
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
    pub kitty_rgba_path: Option<PathBuf>,
    pub kitty_metadata_path: Option<PathBuf>,
    pub stage_telemetry_path: Option<PathBuf>,
    pub stats: ProbeFrameStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeMetalFrameArtifact {
    pub frame_path: PathBuf,
    pub inspect_png_path: PathBuf,
    pub stats_path: PathBuf,
    pub packed_u32le_path: PathBuf,
    pub kitty_rgba_path: Option<PathBuf>,
    pub kitty_metadata_path: Option<PathBuf>,
    pub stage_telemetry_path: Option<PathBuf>,
    pub stats: ProbeFrameStats,
    pub telemetry: Option<ProbeMetalStageTelemetry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeDiffClassification {
    Pass,
    Blank,
    ChannelSwap,
    GlobalShift,
    StructuredMismatch,
}

impl ProbeDiffClassification {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Blank => "blank",
            Self::ChannelSwap => "channel_swap",
            Self::GlobalShift => "global_shift",
            Self::StructuredMismatch => "structured_mismatch",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeSignedBoundingBoxDelta {
    pub min_x: i64,
    pub min_y: i64,
    pub max_x: i64,
    pub max_y: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbePointF64 {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeCentroidDelta {
    pub reference: Option<ProbePointF64>,
    pub candidate: Option<ProbePointF64>,
    pub dx: Option<f64>,
    pub dy: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeTranslationAlignmentMetrics {
    pub dx: i32,
    pub dy: i32,
    pub overlap_pixels: usize,
    pub mean_abs: f64,
    pub max_abs: u8,
    pub mismatch_pixels: usize,
    pub mismatch_ratio: f64,
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
    pub reference_nonblack_pixels: usize,
    pub candidate_nonblack_pixels: usize,
    pub nonblack_delta: i64,
    pub bbox_delta: Option<ProbeSignedBoundingBoxDelta>,
    pub centroid_delta: ProbeCentroidDelta,
    pub best_translation: ProbeTranslationAlignmentMetrics,
    pub classification: ProbeDiffClassification,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeTileBounds {
    pub min_x: usize,
    pub min_y: usize,
    pub max_x: usize,
    pub max_y: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeProjectedSplatTelemetry {
    pub original_index: usize,
    pub screen_x: f32,
    pub screen_y: f32,
    pub depth: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub bbox: ProbeBoundingBox,
    pub tile_bounds: ProbeTileBounds,
    pub opacity: f32,
    pub color: [u8; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeMetalStageTelemetry {
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    pub num_tiles: usize,
    pub tile_capacity: usize,
    pub sort_capacity_before: usize,
    pub sort_capacity_after: usize,
    pub estimated_overlaps: usize,
    pub attempt_sort_count: usize,
    pub sort_path: &'static str,
    pub previous_total_overlaps: u32,
    pub actual_total_overlaps: u32,
    pub valid_count: u32,
    pub retry_count: u32,
    pub overflow_flag: u32,
    pub tile_density: ProbeMetalTileDensityTelemetry,
    pub stage_timings: Vec<ProbeMetalStageTiming>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeMetalStageTiming {
    pub stage: &'static str,
    pub ok: bool,
    pub encode_ms: f64,
    pub wait_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeMetalFrameStageTiming {
    pub frame: usize,
    pub stages: Vec<ProbeMetalStageTiming>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProbeMetalTileDensityTelemetry {
    pub total_tile_entries: u32,
    pub max_tile_range: u32,
    pub p50_tile_range: u32,
    pub p90_tile_range: u32,
    pub p95_tile_range: u32,
    pub p99_tile_range: u32,
    pub tile_ranges_ge_512: u32,
    pub tile_ranges_ge_1024: u32,
    pub tile_ranges_ge_2048: u32,
    pub tile_ranges_ge_4096: u32,
    pub tile_ranges_ge_8192: u32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProbeBackendTiming {
    pub frames: usize,
    pub warmup_frames: usize,
    pub warmup_ms: f64,
    pub render_ms: f64,
    pub readback_normalize_ms: f64,
    pub artifact_write_ms: f64,
    pub stage_timings: Vec<ProbeMetalFrameStageTiming>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProbeTimingSummary {
    pub cpu: Option<ProbeBackendTiming>,
    pub metal: Option<ProbeBackendTiming>,
    pub diff_artifact_ms: f64,
    pub manifest_artifact_ms: f64,
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
    pub timing_path: Option<PathBuf>,
    pub timing: Option<ProbeTimingSummary>,
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
    let mut timing = ProbeTimingSummary::default();
    if config.backend.renders_cpu() {
        let (artifacts, framebuffers, cpu_timing) =
            render_cpu_probe_frames(config, &splats, &camera)?;
        cpu_frames = artifacts;
        cpu_framebuffers = framebuffers;
        timing.cpu = Some(cpu_timing);
    }

    #[cfg(feature = "metal")]
    let (mut metal_frames, mut metal_framebuffers) = (Vec::new(), Vec::new());
    #[cfg(feature = "metal")]
    if config.backend.renders_metal() {
        let (artifacts, framebuffers, metal_timing) =
            render_metal_probe_frames(config, &splats, &camera)?;
        metal_frames = artifacts;
        metal_framebuffers = framebuffers;
        timing.metal = Some(metal_timing);
    }
    #[cfg(not(feature = "metal"))]
    let metal_frames = Vec::new();
    #[cfg(not(feature = "metal"))]
    let metal_framebuffers: Vec<Vec<[u8; 3]>> = Vec::new();

    #[cfg(feature = "metal")]
    let (mut diff_frames, mut diff_summary_path) = (Vec::new(), None);
    #[cfg(feature = "metal")]
    if config.backend.compares_cpu_to_metal() {
        let started = Instant::now();
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
        timing.diff_artifact_ms += duration_ms(started.elapsed());
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
    let manifest_started = Instant::now();
    let timing_path = if config.timing {
        let path = config.out_dir.join("probe_timing.json");
        Some(path)
    } else {
        None
    };
    write_manifest_json(
        &manifest_path,
        config,
        &cpu_frames,
        &metal_frames,
        &diff_frames,
        diff_summary_path.as_deref(),
        timing_path.as_deref(),
        if config.timing { Some(&timing) } else { None },
        &contact_sheet_path,
        &inspect_contact_sheet_path,
    )?;
    timing.manifest_artifact_ms += duration_ms(manifest_started.elapsed());
    if let Some(path) = &timing_path {
        write_manifest_json(
            &manifest_path,
            config,
            &cpu_frames,
            &metal_frames,
            &diff_frames,
            diff_summary_path.as_deref(),
            timing_path.as_deref(),
            Some(&timing),
            &contact_sheet_path,
            &inspect_contact_sheet_path,
        )?;
        write_timing_json(path, &timing)?;
    }

    Ok(ProbeRunResult {
        manifest_path,
        contact_sheet_path,
        inspect_contact_sheet_path,
        cpu_frames,
        metal_frames,
        diff_frames,
        diff_summary_path,
        timing_path,
        timing: if config.timing { Some(timing) } else { None },
    })
}

fn render_cpu_probe_frames(
    config: &ProbeConfig,
    splats: &[Splat],
    camera: &Camera,
) -> Result<
    (
        Vec<ProbeFrameArtifact>,
        Vec<Vec<[u8; 3]>>,
        ProbeBackendTiming,
    ),
    ProbeError,
> {
    let cpu_dir = config.out_dir.join("cpu");
    let inspect_cpu_dir = config.out_dir.join("inspect").join("cpu");
    fs::create_dir_all(&cpu_dir)?;
    fs::create_dir_all(&inspect_cpu_dir)?;
    let terminal_dir = config.out_dir.join("terminal");
    if config.terminal_artifacts {
        fs::create_dir_all(&terminal_dir)?;
    }
    let kitty_dir = config.out_dir.join("kitty");
    if config.kitty_artifacts {
        fs::create_dir_all(&kitty_dir)?;
    }

    let mut timing = ProbeBackendTiming {
        frames: config.frames,
        warmup_frames: config.warmup_frames,
        ..ProbeBackendTiming::default()
    };

    for _ in 0..config.warmup_frames {
        let started = Instant::now();
        let _ = render_cpu_frame(splats, camera, config.width, config.height);
        timing.warmup_ms += duration_ms(started.elapsed());
    }

    let mut artifacts = Vec::with_capacity(config.frames);
    let mut framebuffers = Vec::with_capacity(config.frames);
    for frame_idx in 0..config.frames {
        let render_started = Instant::now();
        let (frame, projected_splats) = render_cpu_frame_with_projection(
            splats,
            camera,
            config.width,
            config.height,
            config.stage_telemetry,
        );
        timing.render_ms += duration_ms(render_started.elapsed());
        let stats = compute_frame_stats(&frame, config.width, config.height);

        let frame_path = cpu_dir.join(format!("frame_{frame_idx:03}.ppm"));
        let inspect_png_path = inspect_cpu_dir.join(format!(
            "frame_{frame_idx:03}.x{}.png",
            config.inspect_scale
        ));
        let stats_path = cpu_dir.join(format!("frame_{frame_idx:03}.json"));
        let (kitty_rgba_path, kitty_metadata_path) = if config.kitty_artifacts {
            (
                Some(kitty_dir.join(format!("cpu_frame_{frame_idx:03}.rgba"))),
                Some(kitty_dir.join(format!("cpu_frame_{frame_idx:03}.json"))),
            )
        } else {
            (None, None)
        };
        let stage_telemetry_path = if config.stage_telemetry {
            Some(cpu_dir.join(format!("stage_telemetry_frame_{frame_idx:03}.json")))
        } else {
            None
        };
        let artifact_started = Instant::now();
        write_ppm(&frame_path, config.width, config.height, &frame)?;
        write_png_inspection(
            &inspect_png_path,
            config.width,
            config.height,
            config.inspect_scale,
            &frame,
        )?;
        write_frame_stats_json(&stats_path, &stats)?;
        if let Some(path) = &stage_telemetry_path {
            write_cpu_stage_telemetry_json(
                path,
                splats.len(),
                config.width,
                config.height,
                projected_splats.as_deref().unwrap_or(&[]),
            )?;
        }
        if config.terminal_artifacts {
            write_halfblock_ansi(
                &terminal_dir.join(format!("cpu_frame_{frame_idx:03}.ansi.txt")),
                config.width,
                config.height,
                &frame,
            )?;
        }
        if let (Some(rgba_path), Some(metadata_path)) = (&kitty_rgba_path, &kitty_metadata_path) {
            let payload_bytes = write_kitty_rgba_from_rgb(rgba_path, &frame)?;
            write_kitty_payload_metadata(
                metadata_path,
                config.width,
                config.height,
                payload_bytes,
            )?;
        }
        timing.artifact_write_ms += duration_ms(artifact_started.elapsed());

        artifacts.push(ProbeFrameArtifact {
            frame_path,
            inspect_png_path,
            stats_path,
            kitty_rgba_path,
            kitty_metadata_path,
            stage_telemetry_path,
            stats,
        });
        framebuffers.push(frame);
    }

    Ok((artifacts, framebuffers, timing))
}

#[cfg(feature = "metal")]
fn render_metal_probe_frames(
    config: &ProbeConfig,
    splats: &[Splat],
    camera: &Camera,
) -> Result<
    (
        Vec<ProbeMetalFrameArtifact>,
        Vec<Vec<[u8; 3]>>,
        ProbeBackendTiming,
    ),
    ProbeError,
> {
    let metal_dir = config.out_dir.join("metal");
    let inspect_metal_dir = config.out_dir.join("inspect").join("metal");
    fs::create_dir_all(&metal_dir)?;
    fs::create_dir_all(&inspect_metal_dir)?;
    let terminal_dir = config.out_dir.join("terminal");
    if config.terminal_artifacts {
        fs::create_dir_all(&terminal_dir)?;
    }
    let kitty_dir = config.out_dir.join("kitty");
    if config.kitty_artifacts {
        fs::create_dir_all(&kitty_dir)?;
    }

    let mut backend = super::metal::MetalBackend::new(splats.len())?;
    backend.set_probe_stage_telemetry_enabled(config.stage_telemetry);
    backend.set_probe_stage_timing_enabled(config.stage_telemetry || config.timing);
    backend.upload_splats(splats)?;
    let mut timing = ProbeBackendTiming {
        frames: config.frames,
        warmup_frames: config.warmup_frames,
        ..ProbeBackendTiming::default()
    };
    for _ in 0..config.warmup_frames {
        let started = Instant::now();
        backend.render(camera, config.width, config.height, splats.len())?;
        let _ = backend.framebuffer_slice();
        timing.warmup_ms += duration_ms(started.elapsed());
    }

    let mut artifacts = Vec::with_capacity(config.frames);
    let mut framebuffers = Vec::with_capacity(config.frames);
    for frame_idx in 0..config.frames {
        let render_started = Instant::now();
        backend.render(camera, config.width, config.height, splats.len())?;
        timing.render_ms += duration_ms(render_started.elapsed());
        let readback_started = Instant::now();
        let packed = backend.framebuffer_slice();
        let frame = normalize_metal_packed_pixels(packed);
        timing.readback_normalize_ms += duration_ms(readback_started.elapsed());
        let stats = compute_frame_stats(&frame, config.width, config.height);
        let metal_telemetry = backend.probe_telemetry();
        if config.timing {
            timing
                .stage_timings
                .push(ProbeMetalFrameStageTiming::from_metal(
                    frame_idx,
                    &metal_telemetry,
                ));
        }
        let telemetry = if config.stage_telemetry {
            Some(ProbeMetalStageTelemetry::from(metal_telemetry))
        } else {
            None
        };

        let frame_path = metal_dir.join(format!("frame_{frame_idx:03}.ppm"));
        let inspect_png_path = inspect_metal_dir.join(format!(
            "frame_{frame_idx:03}.x{}.png",
            config.inspect_scale
        ));
        let stats_path = metal_dir.join(format!("frame_{frame_idx:03}.json"));
        let packed_u32le_path = metal_dir.join(format!("frame_{frame_idx:03}.packed_u32le.bin"));
        let (kitty_rgba_path, kitty_metadata_path) = if config.kitty_artifacts {
            (
                Some(kitty_dir.join(format!("metal_frame_{frame_idx:03}.rgba"))),
                Some(kitty_dir.join(format!("metal_frame_{frame_idx:03}.json"))),
            )
        } else {
            (None, None)
        };
        let stage_telemetry_path = if config.stage_telemetry {
            Some(metal_dir.join(format!("stage_telemetry_frame_{frame_idx:03}.json")))
        } else {
            None
        };
        let artifact_started = Instant::now();
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
        if let (Some(path), Some(telemetry)) = (&stage_telemetry_path, &telemetry) {
            write_metal_stage_telemetry_json(path, telemetry)?;
        }
        if config.terminal_artifacts {
            write_halfblock_ansi(
                &terminal_dir.join(format!("metal_frame_{frame_idx:03}.ansi.txt")),
                config.width,
                config.height,
                &frame,
            )?;
        }
        if let (Some(rgba_path), Some(metadata_path)) = (&kitty_rgba_path, &kitty_metadata_path) {
            let payload_bytes = write_kitty_rgba_from_packed(rgba_path, packed)?;
            write_kitty_payload_metadata(
                metadata_path,
                config.width,
                config.height,
                payload_bytes,
            )?;
        }
        timing.artifact_write_ms += duration_ms(artifact_started.elapsed());

        artifacts.push(ProbeMetalFrameArtifact {
            frame_path,
            inspect_png_path,
            stats_path,
            packed_u32le_path,
            kitty_rgba_path,
            kitty_metadata_path,
            stage_telemetry_path,
            stats,
            telemetry,
        });
        framebuffers.push(frame);
    }

    Ok((artifacts, framebuffers, timing))
}

pub fn render_cpu_frame(
    splats: &[Splat],
    camera: &Camera,
    width: usize,
    height: usize,
) -> Vec<[u8; 3]> {
    render_cpu_frame_with_projection(splats, camera, width, height, false).0
}

fn render_cpu_frame_with_projection(
    splats: &[Splat],
    camera: &Camera,
    width: usize,
    height: usize,
    keep_projection: bool,
) -> (Vec<[u8; 3]>, Option<Vec<ProjectedSplat>>) {
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

    let telemetry_projection = if keep_projection {
        Some(projected_splats)
    } else {
        None
    };
    (render_state.framebuffer, telemetry_projection)
}

#[cfg(feature = "metal")]
impl From<super::metal::MetalProbeTelemetry> for ProbeMetalStageTelemetry {
    fn from(value: super::metal::MetalProbeTelemetry) -> Self {
        Self {
            tile_count_x: value.tile_count_x,
            tile_count_y: value.tile_count_y,
            num_tiles: value.num_tiles,
            tile_capacity: value.tile_capacity,
            sort_capacity_before: value.sort_capacity_before,
            sort_capacity_after: value.sort_capacity_after,
            estimated_overlaps: value.estimated_overlaps,
            attempt_sort_count: value.attempt_sort_count,
            sort_path: value.sort_path,
            previous_total_overlaps: value.previous_total_overlaps,
            actual_total_overlaps: value.actual_total_overlaps,
            valid_count: value.valid_count,
            retry_count: value.retry_count,
            overflow_flag: value.overflow_flag,
            tile_density: ProbeMetalTileDensityTelemetry::from(value.tile_density),
            stage_timings: value.stage_timings[..value.stage_timing_count]
                .iter()
                .copied()
                .map(ProbeMetalStageTiming::from)
                .collect(),
        }
    }
}

#[cfg(feature = "metal")]
impl From<super::metal::MetalStageTimingTelemetry> for ProbeMetalStageTiming {
    fn from(value: super::metal::MetalStageTimingTelemetry) -> Self {
        Self {
            stage: value.stage,
            ok: value.ok,
            encode_ms: value.encode_ms,
            wait_ms: value.wait_ms,
        }
    }
}

#[cfg(feature = "metal")]
impl ProbeMetalFrameStageTiming {
    fn from_metal(frame: usize, telemetry: &super::metal::MetalProbeTelemetry) -> Self {
        Self {
            frame,
            stages: telemetry.stage_timings[..telemetry.stage_timing_count]
                .iter()
                .copied()
                .map(ProbeMetalStageTiming::from)
                .collect(),
        }
    }
}

#[cfg(feature = "metal")]
impl From<super::metal::MetalTileDensityTelemetry> for ProbeMetalTileDensityTelemetry {
    fn from(value: super::metal::MetalTileDensityTelemetry) -> Self {
        Self {
            total_tile_entries: value.total_tile_entries,
            max_tile_range: value.max_tile_range,
            p50_tile_range: value.p50_tile_range,
            p90_tile_range: value.p90_tile_range,
            p95_tile_range: value.p95_tile_range,
            p99_tile_range: value.p99_tile_range,
            tile_ranges_ge_512: value.tile_ranges_ge_512,
            tile_ranges_ge_1024: value.tile_ranges_ge_1024,
            tile_ranges_ge_2048: value.tile_ranges_ge_2048,
            tile_ranges_ge_4096: value.tile_ranges_ge_4096,
            tile_ranges_ge_8192: value.tile_ranges_ge_8192,
        }
    }
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
    let reference_stats = compute_frame_stats(reference, width, height);
    let candidate_stats = compute_frame_stats(candidate, width, height);
    let reference_nonblack_pixels = reference_stats.nonblack_pixels;
    let candidate_nonblack_pixels = candidate_stats.nonblack_pixels;
    let nonblack_delta = candidate_nonblack_pixels as i64 - reference_nonblack_pixels as i64;
    let bbox_delta = bbox_delta(reference_stats.bbox.as_ref(), candidate_stats.bbox.as_ref());
    let centroid_delta = centroid_delta(
        frame_nonblack_centroid(reference, width),
        frame_nonblack_centroid(candidate, width),
    );
    let best_translation =
        best_translation_alignment(reference, candidate, width, height, PROBE_ALIGNMENT_WINDOW);

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
        reference_nonblack_pixels,
        candidate_nonblack_pixels,
        nonblack_delta,
        bbox_delta,
        centroid_delta,
        best_translation,
        classification: ProbeDiffClassification::StructuredMismatch,
    };
    metrics.classification = classify_diff(reference, candidate, &metrics);
    metrics
}

fn bbox_delta(
    reference: Option<&ProbeBoundingBox>,
    candidate: Option<&ProbeBoundingBox>,
) -> Option<ProbeSignedBoundingBoxDelta> {
    let reference = reference?;
    let candidate = candidate?;
    Some(ProbeSignedBoundingBoxDelta {
        min_x: candidate.min_x as i64 - reference.min_x as i64,
        min_y: candidate.min_y as i64 - reference.min_y as i64,
        max_x: candidate.max_x as i64 - reference.max_x as i64,
        max_y: candidate.max_y as i64 - reference.max_y as i64,
    })
}

fn frame_nonblack_centroid(frame: &[[u8; 3]], width: usize) -> Option<ProbePointF64> {
    let mut count = 0usize;
    let mut sum_x = 0f64;
    let mut sum_y = 0f64;
    for (idx, pixel) in frame.iter().enumerate() {
        if pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0 {
            continue;
        }
        count += 1;
        let x = if width == 0 { 0 } else { idx % width };
        let y = if width == 0 { 0 } else { idx / width };
        sum_x += x as f64;
        sum_y += y as f64;
    }

    if count == 0 {
        None
    } else {
        Some(ProbePointF64 {
            x: sum_x / count as f64,
            y: sum_y / count as f64,
        })
    }
}

fn centroid_delta(
    reference: Option<ProbePointF64>,
    candidate: Option<ProbePointF64>,
) -> ProbeCentroidDelta {
    let (dx, dy) = match (&reference, &candidate) {
        (Some(reference), Some(candidate)) => (
            Some(candidate.x - reference.x),
            Some(candidate.y - reference.y),
        ),
        _ => (None, None),
    };
    ProbeCentroidDelta {
        reference,
        candidate,
        dx,
        dy,
    }
}

fn best_translation_alignment(
    reference: &[[u8; 3]],
    candidate: &[[u8; 3]],
    width: usize,
    height: usize,
    window: i32,
) -> ProbeTranslationAlignmentMetrics {
    let mut best = translation_alignment(reference, candidate, width, height, 0, 0);
    for dy in -window..=window {
        for dx in -window..=window {
            let candidate_metrics =
                translation_alignment(reference, candidate, width, height, dx, dy);
            if is_better_translation(&candidate_metrics, &best) {
                best = candidate_metrics;
            }
        }
    }
    best
}

fn translation_alignment(
    reference: &[[u8; 3]],
    candidate: &[[u8; 3]],
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
) -> ProbeTranslationAlignmentMetrics {
    let mut sum_abs = 0u64;
    let mut max_abs = 0u8;
    let mut mismatch_pixels = 0usize;
    let mut overlap_pixels = 0usize;

    for y in 0..height {
        let shifted_y = y as i32 + dy;
        if shifted_y < 0 || shifted_y >= height as i32 {
            continue;
        }
        for x in 0..width {
            let shifted_x = x as i32 + dx;
            if shifted_x < 0 || shifted_x >= width as i32 {
                continue;
            }
            let reference_pixel = reference[y * width + x];
            let candidate_pixel = candidate[shifted_y as usize * width + shifted_x as usize];
            let dr = abs_diff_u8(reference_pixel[0], candidate_pixel[0]);
            let dg = abs_diff_u8(reference_pixel[1], candidate_pixel[1]);
            let db = abs_diff_u8(reference_pixel[2], candidate_pixel[2]);
            if dr != 0 || dg != 0 || db != 0 {
                mismatch_pixels += 1;
            }
            sum_abs += dr as u64 + dg as u64 + db as u64;
            max_abs = max_abs.max(dr).max(dg).max(db);
            overlap_pixels += 1;
        }
    }

    let total_channels = overlap_pixels.saturating_mul(3);
    ProbeTranslationAlignmentMetrics {
        dx,
        dy,
        overlap_pixels,
        mean_abs: if total_channels == 0 {
            f64::INFINITY
        } else {
            sum_abs as f64 / total_channels as f64
        },
        max_abs,
        mismatch_pixels,
        mismatch_ratio: if overlap_pixels == 0 {
            1.0
        } else {
            mismatch_pixels as f64 / overlap_pixels as f64
        },
    }
}

fn is_better_translation(
    candidate: &ProbeTranslationAlignmentMetrics,
    best: &ProbeTranslationAlignmentMetrics,
) -> bool {
    candidate
        .mismatch_ratio
        .total_cmp(&best.mismatch_ratio)
        .then(candidate.mean_abs.total_cmp(&best.mean_abs))
        .then(candidate.max_abs.cmp(&best.max_abs))
        .then(best.overlap_pixels.cmp(&candidate.overlap_pixels))
        .then((candidate.dx.abs() + candidate.dy.abs()).cmp(&(best.dx.abs() + best.dy.abs())))
        .is_lt()
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

fn kitty_base64_bytes(payload_bytes: usize) -> usize {
    payload_bytes.div_ceil(3) * 4
}

fn kitty_chunk_count(base64_bytes: usize, chunk_size: usize) -> usize {
    if base64_bytes == 0 {
        0
    } else {
        base64_bytes.div_ceil(chunk_size.max(1))
    }
}

fn write_kitty_payload_metadata(
    path: &Path,
    width: usize,
    height: usize,
    payload_bytes: usize,
) -> io::Result<()> {
    let base64_bytes = kitty_base64_bytes(payload_bytes);
    let chunks_4096 = kitty_chunk_count(base64_bytes, 4096);
    fs::write(
        path,
        format!(
            concat!(
                "{{",
                "\"format\":\"rgba8\",",
                "\"width\":{},",
                "\"height\":{},",
                "\"payload_bytes\":{},",
                "\"base64_bytes\":{},",
                "\"chunks_4096\":{}",
                "}}\n"
            ),
            width, height, payload_bytes, base64_bytes, chunks_4096
        ),
    )
}

fn write_kitty_rgba_from_rgb(path: &Path, frame: &[[u8; 3]]) -> io::Result<usize> {
    let mut file = fs::File::create(path)?;
    let mut bytes_written = 0usize;
    for pixel in frame {
        file.write_all(&[pixel[0], pixel[1], pixel[2], 255])?;
        bytes_written += 4;
    }
    Ok(bytes_written)
}

fn write_kitty_rgba_from_packed(path: &Path, packed: &[u32]) -> io::Result<usize> {
    let mut file = fs::File::create(path)?;
    let mut bytes_written = 0usize;
    for pixel in packed {
        let r = (pixel & 0xFF) as u8;
        let g = ((pixel >> 8) & 0xFF) as u8;
        let b = ((pixel >> 16) & 0xFF) as u8;
        let a = ((pixel >> 24) & 0xFF) as u8;
        file.write_all(&[r, g, b, a])?;
        bytes_written += 4;
    }
    Ok(bytes_written)
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
    timing_path: Option<&Path>,
    timing: Option<&ProbeTimingSummary>,
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
                    "\"kitty_rgba\":{},",
                    "\"kitty_metadata\":{},",
                    "\"stage_telemetry\":{},",
                    "\"metrics\":{}",
                    "}}"
                ),
                idx,
                json_escape(&display_path(&frame.frame_path)),
                json_escape(&display_path(&frame.inspect_png_path)),
                json_escape(&display_path(&frame.stats_path)),
                optional_path_json(frame.kitty_rgba_path.as_deref()),
                optional_path_json(frame.kitty_metadata_path.as_deref()),
                optional_path_json(frame.stage_telemetry_path.as_deref()),
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
                    "\"kitty_rgba\":{},",
                    "\"kitty_metadata\":{},",
                    "\"stage_telemetry\":{},",
                    "\"telemetry\":{},",
                    "\"metrics\":{}",
                    "}}"
                ),
                idx,
                json_escape(&display_path(&frame.frame_path)),
                json_escape(&display_path(&frame.inspect_png_path)),
                json_escape(&display_path(&frame.stats_path)),
                json_escape(&display_path(&frame.packed_u32le_path)),
                optional_path_json(frame.kitty_rgba_path.as_deref()),
                optional_path_json(frame.kitty_metadata_path.as_deref()),
                optional_path_json(frame.stage_telemetry_path.as_deref()),
                frame
                    .telemetry
                    .as_ref()
                    .map(metal_stage_telemetry_json)
                    .unwrap_or_else(|| "null".to_string()),
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
    let timing_artifact = optional_path_json(timing_path);
    let timing_summary = timing
        .map(probe_timing_summary_json)
        .unwrap_or_else(|| "null".to_string());
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
            "  \"kitty_artifacts\": {},\n",
            "  \"stage_telemetry\": {},\n",
            "  \"timing_enabled\": {},\n",
            "  \"artifacts\": {{\n",
            "    \"contact_sheet\": \"{}\",\n",
            "    \"inspect_contact_sheet\": \"{}\",\n",
            "    \"diff_summary\": {},\n",
            "    \"timing\": {}\n",
            "  }},\n",
            "  \"timing\": {},\n",
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
        config.kitty_artifacts,
        config.stage_telemetry,
        config.timing,
        json_escape(&display_path(contact_sheet_path)),
        json_escape(&display_path(inspect_contact_sheet_path)),
        diff_summary,
        timing_artifact,
        timing_summary,
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
    let bbox_delta = metrics
        .bbox_delta
        .as_ref()
        .map(bbox_delta_json)
        .unwrap_or_else(|| "null".to_string());
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
            "\"reference_nonblack_pixels\":{},",
            "\"candidate_nonblack_pixels\":{},",
            "\"nonblack_delta\":{},",
            "\"bbox_delta\":{},",
            "\"centroid_delta\":{},",
            "\"best_translation\":{},",
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
        metrics.reference_nonblack_pixels,
        metrics.candidate_nonblack_pixels,
        metrics.nonblack_delta,
        bbox_delta,
        centroid_delta_json(&metrics.centroid_delta),
        translation_alignment_json(&metrics.best_translation),
        metrics.classification
    )
}

fn optional_path_json(path: Option<&Path>) -> String {
    match path {
        Some(path) => format!("\"{}\"", json_escape(&display_path(path))),
        None => "null".to_string(),
    }
}

fn bbox_delta_json(delta: &ProbeSignedBoundingBoxDelta) -> String {
    format!(
        "{{\"min_x\":{},\"min_y\":{},\"max_x\":{},\"max_y\":{}}}",
        delta.min_x, delta.min_y, delta.max_x, delta.max_y
    )
}

fn point_json(point: &ProbePointF64) -> String {
    format!("{{\"x\":{:.6},\"y\":{:.6}}}", point.x, point.y)
}

fn optional_f64_json(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "null".to_string())
}

fn centroid_delta_json(delta: &ProbeCentroidDelta) -> String {
    let reference = delta
        .reference
        .as_ref()
        .map(point_json)
        .unwrap_or_else(|| "null".to_string());
    let candidate = delta
        .candidate
        .as_ref()
        .map(point_json)
        .unwrap_or_else(|| "null".to_string());
    format!(
        "{{\"reference\":{},\"candidate\":{},\"dx\":{},\"dy\":{}}}",
        reference,
        candidate,
        optional_f64_json(delta.dx),
        optional_f64_json(delta.dy)
    )
}

fn translation_alignment_json(metrics: &ProbeTranslationAlignmentMetrics) -> String {
    format!(
        concat!(
            "{{",
            "\"dx\":{},",
            "\"dy\":{},",
            "\"overlap_pixels\":{},",
            "\"mean_abs\":{:.6},",
            "\"max_abs\":{},",
            "\"mismatch_pixels\":{},",
            "\"mismatch_ratio\":{:.6}",
            "}}"
        ),
        metrics.dx,
        metrics.dy,
        metrics.overlap_pixels,
        metrics.mean_abs,
        metrics.max_abs,
        metrics.mismatch_pixels,
        metrics.mismatch_ratio
    )
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn backend_timing_json(timing: &ProbeBackendTiming) -> String {
    let avg_render_ms = if timing.frames == 0 {
        0.0
    } else {
        timing.render_ms / timing.frames as f64
    };
    let stage_timings = metal_frame_stage_timings_json(&timing.stage_timings);
    format!(
        concat!(
            "{{",
            "\"frames\":{},",
            "\"warmup_frames\":{},",
            "\"warmup_ms\":{:.6},",
            "\"render_ms\":{:.6},",
            "\"render_avg_ms\":{:.6},",
            "\"readback_normalize_ms\":{:.6},",
            "\"artifact_write_ms\":{:.6},",
            "\"stage_timings\":{}",
            "}}"
        ),
        timing.frames,
        timing.warmup_frames,
        timing.warmup_ms,
        timing.render_ms,
        avg_render_ms,
        timing.readback_normalize_ms,
        timing.artifact_write_ms,
        stage_timings
    )
}

fn probe_timing_summary_json(timing: &ProbeTimingSummary) -> String {
    let cpu = timing
        .cpu
        .as_ref()
        .map(backend_timing_json)
        .unwrap_or_else(|| "null".to_string());
    let metal = timing
        .metal
        .as_ref()
        .map(backend_timing_json)
        .unwrap_or_else(|| "null".to_string());
    format!(
        concat!(
            "{{",
            "\"cpu\":{},",
            "\"metal\":{},",
            "\"diff_artifact_ms\":{:.6},",
            "\"manifest_artifact_ms\":{:.6}",
            "}}"
        ),
        cpu, metal, timing.diff_artifact_ms, timing.manifest_artifact_ms
    )
}

fn write_timing_json(path: &Path, timing: &ProbeTimingSummary) -> io::Result<()> {
    fs::write(path, format!("{}\n", probe_timing_summary_json(timing)))
}

fn metal_stage_telemetry_json(telemetry: &ProbeMetalStageTelemetry) -> String {
    let stage_timings = metal_stage_timings_json(&telemetry.stage_timings);
    format!(
        concat!(
            "{{",
            "\"tile_count_x\":{},",
            "\"tile_count_y\":{},",
            "\"num_tiles\":{},",
            "\"tile_capacity\":{},",
            "\"sort_capacity_before\":{},",
            "\"sort_capacity_after\":{},",
            "\"estimated_overlaps\":{},",
            "\"attempt_sort_count\":{},",
            "\"sort_path\":\"{}\",",
            "\"previous_total_overlaps\":{},",
            "\"actual_total_overlaps\":{},",
            "\"valid_count\":{},",
            "\"retry_count\":{},",
            "\"overflow_flag\":{},",
            "\"tile_density\":{},",
            "\"stage_timings\":{}",
            "}}"
        ),
        telemetry.tile_count_x,
        telemetry.tile_count_y,
        telemetry.num_tiles,
        telemetry.tile_capacity,
        telemetry.sort_capacity_before,
        telemetry.sort_capacity_after,
        telemetry.estimated_overlaps,
        telemetry.attempt_sort_count,
        telemetry.sort_path,
        telemetry.previous_total_overlaps,
        telemetry.actual_total_overlaps,
        telemetry.valid_count,
        telemetry.retry_count,
        telemetry.overflow_flag,
        metal_tile_density_telemetry_json(&telemetry.tile_density),
        stage_timings
    )
}

fn metal_stage_timings_json(timings: &[ProbeMetalStageTiming]) -> String {
    let stages = timings
        .iter()
        .map(|timing| {
            format!(
                concat!(
                    "{{",
                    "\"stage\":\"{}\",",
                    "\"ok\":{},",
                    "\"encode_ms\":{:.6},",
                    "\"wait_ms\":{:.6}",
                    "}}"
                ),
                json_escape(timing.stage),
                timing.ok,
                timing.encode_ms,
                timing.wait_ms
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", stages)
}

fn metal_frame_stage_timings_json(timings: &[ProbeMetalFrameStageTiming]) -> String {
    let frames = timings
        .iter()
        .map(|timing| {
            format!(
                "{{\"frame\":{},\"stages\":{}}}",
                timing.frame,
                metal_stage_timings_json(&timing.stages)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", frames)
}

fn metal_tile_density_telemetry_json(telemetry: &ProbeMetalTileDensityTelemetry) -> String {
    format!(
        concat!(
            "{{",
            "\"total_tile_entries\":{},",
            "\"max_tile_range\":{},",
            "\"p50_tile_range\":{},",
            "\"p90_tile_range\":{},",
            "\"p95_tile_range\":{},",
            "\"p99_tile_range\":{},",
            "\"tile_ranges_ge_512\":{},",
            "\"tile_ranges_ge_1024\":{},",
            "\"tile_ranges_ge_2048\":{},",
            "\"tile_ranges_ge_4096\":{},",
            "\"tile_ranges_ge_8192\":{}",
            "}}"
        ),
        telemetry.total_tile_entries,
        telemetry.max_tile_range,
        telemetry.p50_tile_range,
        telemetry.p90_tile_range,
        telemetry.p95_tile_range,
        telemetry.p99_tile_range,
        telemetry.tile_ranges_ge_512,
        telemetry.tile_ranges_ge_1024,
        telemetry.tile_ranges_ge_2048,
        telemetry.tile_ranges_ge_4096,
        telemetry.tile_ranges_ge_8192
    )
}

#[cfg(feature = "metal")]
fn write_metal_stage_telemetry_json(
    path: &Path,
    telemetry: &ProbeMetalStageTelemetry,
) -> io::Result<()> {
    fs::write(path, format!("{}\n", metal_stage_telemetry_json(telemetry)))
}

fn projected_splat_telemetry_json(telemetry: &ProbeProjectedSplatTelemetry) -> String {
    format!(
        concat!(
            "{{",
            "\"original_index\":{},",
            "\"screen_center\":{{\"x\":{:.6},\"y\":{:.6}}},",
            "\"depth\":{:.6},",
            "\"radius\":{{\"x\":{:.6},\"y\":{:.6}}},",
            "\"bbox\":{{\"min_x\":{},\"min_y\":{},\"max_x\":{},\"max_y\":{}}},",
            "\"tile_bounds\":{{\"min_x\":{},\"min_y\":{},\"max_x\":{},\"max_y\":{}}},",
            "\"opacity\":{:.6},",
            "\"color\":[{},{},{}]",
            "}}"
        ),
        telemetry.original_index,
        telemetry.screen_x,
        telemetry.screen_y,
        telemetry.depth,
        telemetry.radius_x,
        telemetry.radius_y,
        telemetry.bbox.min_x,
        telemetry.bbox.min_y,
        telemetry.bbox.max_x,
        telemetry.bbox.max_y,
        telemetry.tile_bounds.min_x,
        telemetry.tile_bounds.min_y,
        telemetry.tile_bounds.max_x,
        telemetry.tile_bounds.max_y,
        telemetry.opacity,
        telemetry.color[0],
        telemetry.color[1],
        telemetry.color[2]
    )
}

fn projected_splat_telemetry(
    splat: &ProjectedSplat,
    width: usize,
    height: usize,
) -> ProbeProjectedSplatTelemetry {
    let max_x = width.saturating_sub(1);
    let max_y = height.saturating_sub(1);
    let min_x = (splat.screen_x - splat.radius_x).floor().max(0.0) as usize;
    let min_y = (splat.screen_y - splat.radius_y).floor().max(0.0) as usize;
    let bbox_max_x = (splat.screen_x + splat.radius_x).ceil().max(0.0) as usize;
    let bbox_max_y = (splat.screen_y + splat.radius_y).ceil().max(0.0) as usize;
    let bbox = ProbeBoundingBox {
        min_x: min_x.min(max_x),
        min_y: min_y.min(max_y),
        max_x: bbox_max_x.min(max_x),
        max_y: bbox_max_y.min(max_y),
    };
    let tile_bounds = ProbeTileBounds {
        min_x: bbox.min_x / PROBE_TILE_SIZE,
        min_y: bbox.min_y / PROBE_TILE_SIZE,
        max_x: bbox.max_x / PROBE_TILE_SIZE,
        max_y: bbox.max_y / PROBE_TILE_SIZE,
    };

    ProbeProjectedSplatTelemetry {
        original_index: splat.original_index,
        screen_x: splat.screen_x,
        screen_y: splat.screen_y,
        depth: splat.depth,
        radius_x: splat.radius_x,
        radius_y: splat.radius_y,
        bbox,
        tile_bounds,
        opacity: splat.opacity,
        color: splat.color,
    }
}

fn write_cpu_stage_telemetry_json(
    path: &Path,
    input_splat_count: usize,
    width: usize,
    height: usize,
    projected_splats: &[ProjectedSplat],
) -> io::Result<()> {
    let splats_json = projected_splats
        .iter()
        .map(|splat| {
            projected_splat_telemetry_json(&projected_splat_telemetry(splat, width, height))
        })
        .collect::<Vec<_>>()
        .join(",\n    ");
    let json = format!(
        concat!(
            "{{\n",
            "  \"source\": \"cpu_project_and_cull_splats\",\n",
            "  \"input_splat_count\": {},\n",
            "  \"projected_splat_count\": {},\n",
            "  \"sorted_by_depth\": true,\n",
            "  \"width\": {},\n",
            "  \"height\": {},\n",
            "  \"tile_size\": {},\n",
            "  \"splats\": [\n    {}\n  ]\n",
            "}}\n"
        ),
        input_splat_count,
        projected_splats.len(),
        width,
        height,
        PROBE_TILE_SIZE,
        splats_json
    );
    fs::write(path, json)
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
        ProbeDiffClassification::GlobalShift => 3,
        ProbeDiffClassification::StructuredMismatch => 4,
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

    let best = &metrics.best_translation;
    let best_is_shift = best.dx != 0 || best.dy != 0;
    let strongly_improved = best.mismatch_ratio <= metrics.mismatch_ratio * 0.25
        || (metrics.mismatch_ratio - best.mismatch_ratio) >= 0.20;
    let low_residual = best.mismatch_ratio <= 0.05 || best.mean_abs <= metrics.mean_abs * 0.20;
    if best_is_shift && strongly_improved && low_residual {
        return ProbeDiffClassification::GlobalShift;
    }

    ProbeDiffClassification::StructuredMismatch
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
    fn probe_diff_metrics_classify_pass_structured_blank_and_channel_swap() {
        let reference = vec![[10, 20, 30], [0, 0, 0], [200, 10, 5], [1, 2, 3]];

        let pass = compute_diff_metrics(&reference, &reference, 2, 2);
        assert_eq!(pass.classification, ProbeDiffClassification::Pass);
        assert_eq!(pass.mean_abs, 0.0);
        assert_eq!(pass.max_abs, 0);
        assert_eq!(pass.mismatch_pixels, 0);

        let mismatch_candidate = vec![[10, 20, 30], [0, 1, 0], [190, 11, 5], [1, 2, 3]];
        let mismatch = compute_diff_metrics(&reference, &mismatch_candidate, 2, 2);
        assert_eq!(
            mismatch.classification,
            ProbeDiffClassification::StructuredMismatch
        );
        assert_eq!(mismatch.max_abs, 10);
        assert_eq!(mismatch.mismatch_pixels, 2);
        assert_eq!(mismatch.sum_abs_r, 10);
        assert_eq!(mismatch.sum_abs_g, 2);
        assert_eq!(mismatch.sum_abs_b, 0);
        assert_eq!(mismatch.reference_nonblack_pixels, 3);
        assert_eq!(mismatch.candidate_nonblack_pixels, 4);
        assert_eq!(mismatch.nonblack_delta, 1);
        assert!(mismatch.bbox_delta.is_some());
        assert_eq!(mismatch.best_translation.dx, 0);
        assert_eq!(mismatch.best_translation.dy, 0);

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
    fn probe_diff_metrics_detect_global_shift_and_centroid_delta() {
        let mut reference = vec![[0, 0, 0]; 15];
        let mut candidate = vec![[0, 0, 0]; 15];
        reference[1 + 5] = [240, 10, 20];
        candidate[2 + 5] = [240, 10, 20];

        let metrics = compute_diff_metrics(&reference, &candidate, 5, 3);

        assert_eq!(metrics.classification, ProbeDiffClassification::GlobalShift);
        assert_eq!(metrics.nonblack_delta, 0);
        assert_eq!(metrics.bbox_delta.as_ref().unwrap().min_x, 1);
        assert_eq!(metrics.centroid_delta.dx.unwrap(), 1.0);
        assert_eq!(metrics.centroid_delta.dy.unwrap(), 0.0);
        assert_eq!(metrics.best_translation.dx, 1);
        assert_eq!(metrics.best_translation.dy, 0);
        assert_eq!(metrics.best_translation.mismatch_pixels, 0);
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
        assert!(summary.contains("\"classification\":\"structured_mismatch\""));

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
    fn probe_writes_kitty_rgba_from_rgb() {
        let dir = unique_temp_dir("probe_kitty_rgb_writer");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.rgba");

        let bytes_written = write_kitty_rgba_from_rgb(&path, &[[1, 2, 3], [4, 5, 6]]).unwrap();

        assert_eq!(bytes_written, 8);
        assert_eq!(fs::read(path).unwrap(), vec![1, 2, 3, 255, 4, 5, 6, 255]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn probe_writes_kitty_rgba_from_packed() {
        let dir = unique_temp_dir("probe_kitty_packed_writer");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.rgba");

        let bytes_written = write_kitty_rgba_from_packed(&path, &[0x44332211]).unwrap();

        assert_eq!(bytes_written, 4);
        assert_eq!(fs::read(path).unwrap(), vec![0x11, 0x22, 0x33, 0x44]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn probe_writes_kitty_payload_metadata() {
        let dir = unique_temp_dir("probe_kitty_metadata");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.json");

        write_kitty_payload_metadata(&path, 2, 1, 8).unwrap();

        let json = fs::read_to_string(path).unwrap();
        assert!(json.contains("\"format\":\"rgba8\""));
        assert!(json.contains("\"payload_bytes\":8"));
        assert!(json.contains("\"base64_bytes\":12"));
        assert!(json.contains("\"chunks_4096\":1"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn probe_metal_stage_timing_json_is_structured() {
        let telemetry = ProbeMetalStageTelemetry {
            tile_count_x: 2,
            tile_count_y: 3,
            num_tiles: 6,
            tile_capacity: 6,
            sort_capacity_before: 16,
            sort_capacity_after: 16,
            estimated_overlaps: 12,
            attempt_sort_count: 12,
            sort_path: "fused",
            previous_total_overlaps: 0,
            actual_total_overlaps: 4,
            valid_count: 2,
            retry_count: 0,
            overflow_flag: 0,
            tile_density: ProbeMetalTileDensityTelemetry::default(),
            stage_timings: vec![ProbeMetalStageTiming {
                stage: "fused_render_attempt",
                ok: true,
                encode_ms: 1.25,
                wait_ms: 2.5,
            }],
        };
        let json = metal_stage_telemetry_json(&telemetry);

        assert!(json.contains("\"stage_timings\""));
        assert!(json.contains("\"stage\":\"fused_render_attempt\""));
        assert!(json.contains("\"ok\":true"));
        assert!(json.contains("\"encode_ms\":1.250000"));
        assert!(json.contains("\"wait_ms\":2.500000"));

        let timing = ProbeBackendTiming {
            frames: 1,
            stage_timings: vec![ProbeMetalFrameStageTiming {
                frame: 0,
                stages: telemetry.stage_timings,
            }],
            ..ProbeBackendTiming::default()
        };
        let timing_json = backend_timing_json(&timing);
        assert!(timing_json.contains("\"stage_timings\""));
        assert!(timing_json.contains("\"frame\":0"));
        assert!(timing_json.contains("\"stage\":\"fused_render_attempt\""));
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

    #[test]
    fn probe_run_writes_cpu_stage_telemetry_and_timing_when_requested() {
        let out_dir = unique_temp_dir("probe_run_stage_timing");
        let mut config = ProbeConfig::new(&out_dir);
        config.width = 24;
        config.height = 16;
        config.case = ProbeCase::Channels;
        config.stage_telemetry = true;
        config.timing = true;

        let result = run_probe(&config, &[]).unwrap();

        let stage_path = result.cpu_frames[0]
            .stage_telemetry_path
            .as_ref()
            .expect("stage telemetry path");
        assert!(stage_path.exists());
        assert!(result.timing_path.as_ref().unwrap().exists());
        assert!(result.timing.as_ref().unwrap().cpu.is_some());

        let stage = fs::read_to_string(stage_path).unwrap();
        assert!(stage.contains("\"source\": \"cpu_project_and_cull_splats\""));
        assert!(stage.contains("\"original_index\""));
        assert!(stage.contains("\"tile_bounds\""));

        let manifest = fs::read_to_string(&result.manifest_path).unwrap();
        assert!(manifest.contains("\"stage_telemetry\": true"));
        assert!(manifest.contains("\"timing_enabled\": true"));
        assert!(manifest.contains("probe_timing.json"));

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
        config.timing = true;

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
        let timing_path = result
            .timing_path
            .as_ref()
            .expect("probe timing path should be captured");
        let timing_json = fs::read_to_string(timing_path).unwrap();
        assert!(timing_json.contains("\"stage_timings\""));
        assert!(timing_json.contains("\"frame\":0"));
        assert_eq!(
            result.diff_frames[0].metrics.classification,
            ProbeDiffClassification::Pass
        );
        assert!(result.contact_sheet_path.exists());

        let _ = fs::remove_dir_all(out_dir);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn probe_run_tile_boundary_matches_metal_when_device_is_available() {
        if metal::Device::system_default().is_none() {
            eprintln!("Skipping Metal probe test: no system-default Metal device.");
            return;
        }

        let out_dir = unique_temp_dir("probe_run_tile_boundary");
        let mut config = ProbeConfig::new(&out_dir);
        config.width = 96;
        config.height = 72;
        config.case = ProbeCase::TileBoundary;
        config.backend = ProbeBackendSelection::Both;

        let result = run_probe(&config, &[]).unwrap();

        assert_eq!(result.diff_frames.len(), 1);
        assert_eq!(
            result.diff_frames[0].metrics.classification,
            ProbeDiffClassification::Pass
        );
        assert_eq!(result.diff_frames[0].metrics.max_abs, 0);

        let _ = fs::remove_dir_all(out_dir);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn probe_run_edge_bounds_overlap_count_matches_cpu_when_device_is_available() {
        if metal::Device::system_default().is_none() {
            eprintln!("Skipping Metal probe test: no system-default Metal device.");
            return;
        }

        let out_dir = unique_temp_dir("probe_run_edge_bounds");
        let splats = vec![probe_splat(
            Vec3::new(0.014, 0.0, 0.0),
            [255, 210, 40],
            0.9,
            0.18,
        )];
        let mut config = ProbeConfig::new(&out_dir);
        config.width = 96;
        config.height = 72;
        config.case = ProbeCase::Loaded;
        config.backend = ProbeBackendSelection::Both;
        config.stage_telemetry = true;

        let camera = config.camera.to_camera();
        let (_, projected) =
            render_cpu_frame_with_projection(&splats, &camera, config.width, config.height, true);
        let projected = projected.expect("CPU projection should be retained");
        let cpu_overlaps =
            cpu_equivalent_total_tile_overlaps(&projected, config.width, config.height);
        let truncating_overlaps =
            truncating_total_tile_overlaps(&projected, config.width, config.height);
        assert!(
            cpu_overlaps > truncating_overlaps,
            "synthetic edge-bound case must catch max-bound truncation"
        );

        let result = run_probe(&config, &splats).unwrap();
        let telemetry = result.metal_frames[0]
            .telemetry
            .as_ref()
            .expect("stage telemetry should be captured");

        assert_eq!(telemetry.valid_count, projected.len() as u32);
        assert_eq!(telemetry.actual_total_overlaps, cpu_overlaps);
        assert_eq!(telemetry.tile_density.total_tile_entries, cpu_overlaps);
        assert!(telemetry.tile_density.max_tile_range > 0);
        assert!(telemetry.tile_density.p95_tile_range <= telemetry.tile_density.max_tile_range);

        let stage_path = result.metal_frames[0]
            .stage_telemetry_path
            .as_ref()
            .expect("Metal stage telemetry path should be captured");
        let stage_json = fs::read_to_string(stage_path).unwrap();
        assert!(stage_json.contains("\"tile_density\""));
        assert!(stage_json.contains("\"total_tile_entries\""));
        assert!(stage_json.contains("\"p99_tile_range\""));
        assert!(stage_json.contains("\"tile_ranges_ge_512\""));
        assert!(stage_json.contains("\"stage_timings\""));
        assert!(
            stage_json.contains("\"stage\":\"fused_render_attempt\"")
                || stage_json.contains("\"stage\":\"project_count_scan\"")
        );
        assert!(stage_json.contains("\"encode_ms\""));
        assert!(stage_json.contains("\"wait_ms\""));
        assert_eq!(
            result.diff_frames[0].metrics.classification,
            ProbeDiffClassification::Pass
        );
        assert_eq!(result.diff_frames[0].metrics.max_abs, 0);

        let _ = fs::remove_dir_all(out_dir);
    }

    #[cfg(feature = "metal")]
    fn cpu_equivalent_total_tile_overlaps(
        projected_splats: &[ProjectedSplat],
        width: usize,
        height: usize,
    ) -> u32 {
        projected_splats
            .iter()
            .map(|splat| {
                let telemetry = projected_splat_telemetry(splat, width, height);
                let width_tiles = telemetry.tile_bounds.max_x - telemetry.tile_bounds.min_x + 1;
                let height_tiles = telemetry.tile_bounds.max_y - telemetry.tile_bounds.min_y + 1;
                (width_tiles * height_tiles) as u32
            })
            .sum()
    }

    #[cfg(feature = "metal")]
    fn truncating_total_tile_overlaps(
        projected_splats: &[ProjectedSplat],
        width: usize,
        height: usize,
    ) -> u32 {
        let max_x = width.saturating_sub(1) as f32;
        let max_y = height.saturating_sub(1) as f32;
        projected_splats
            .iter()
            .map(|splat| {
                let min_x = (splat.screen_x - splat.radius_x).max(0.0) as usize / PROBE_TILE_SIZE;
                let min_y = (splat.screen_y - splat.radius_y).max(0.0) as usize / PROBE_TILE_SIZE;
                let max_x = (splat.screen_x + splat.radius_x).min(max_x) as usize / PROBE_TILE_SIZE;
                let max_y = (splat.screen_y + splat.radius_y).min(max_y) as usize / PROBE_TILE_SIZE;
                ((max_x - min_x + 1) * (max_y - min_y + 1)) as u32
            })
            .sum()
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("tortuise_{}_{}", name, nanos))
    }
}
