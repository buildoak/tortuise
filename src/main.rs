use clap::{CommandFactory, Parser};
use crossterm::{
    cursor,
    event::{KeyboardEnhancementFlags, PushKeyboardEnhancementFlags},
    execute,
    terminal::{self, ClearType, EnterAlternateScreen},
};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

mod camera;
mod demo;
mod input;
mod math;
mod parser;
mod render;
mod sort;
mod splat;
mod terminal_setup;

use camera::Camera;
use math::Vec3;
use render::frame::run_app_loop;
use render::{AppState, Backend, CameraMode, RenderMode, RenderState};
use terminal_setup::{cleanup_terminal, install_panic_hook};

pub type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Parser)]
#[command(
    name = "tortuise",
    version,
    about = "Terminal-native 3D Gaussian Splatting viewer"
)]
struct Cli {
    /// Path to a .ply or .splat scene file
    input: Option<PathBuf>,
    #[arg(
        long,
        value_name = "OUT_DIR",
        help = "Render deterministic probe artifacts and exit"
    )]
    render_probe: Option<PathBuf>,
    #[arg(
        long,
        value_name = "cpu|metal|both",
        default_value = "cpu",
        help = "Probe backends; metal/both require building with --features metal"
    )]
    probe_backends: String,
    #[arg(
        long,
        value_name = "WxH",
        default_value = "64x48",
        help = "Probe framebuffer size"
    )]
    probe_size: String,
    #[arg(
        long,
        value_name = "x,y,z",
        help = "Probe camera position; required with --render-probe"
    )]
    probe_camera_pos: Option<String>,
    #[arg(
        long,
        value_name = "x,y,z",
        default_value = "0,0,0",
        help = "Probe camera look-at target"
    )]
    probe_look_at: String,
    #[arg(
        long,
        value_name = "yaw,pitch",
        conflicts_with = "probe_look_at",
        help = "Probe camera yaw/pitch in radians"
    )]
    probe_yaw_pitch: Option<String>,
    #[arg(
        long,
        default_value_t = 60.0,
        help = "Probe camera field of view in degrees"
    )]
    probe_fov_deg: f32,
    #[arg(
        long,
        value_name = "N",
        default_value_t = 1,
        help = "Nearest-neighbor scale for PNG inspection probe artifacts"
    )]
    probe_inspect_scale: usize,
    #[arg(long, default_value_t = 1, help = "Probe frames to capture")]
    probe_frames: usize,
    #[arg(long, default_value_t = 0, help = "Probe warmup frames to discard")]
    probe_warmup: usize,
    #[arg(
        long,
        value_name = "N",
        help = "Enable probe timing and capture N benchmark frames"
    )]
    probe_benchmark_frames: Option<usize>,
    #[arg(long, help = "Write probe timing JSON")]
    probe_timing: bool,
    #[arg(long, help = "Write CPU/Metal probe stage telemetry JSON")]
    probe_stage_telemetry: bool,
    #[arg(
        long,
        value_name = "loaded|channels|depth|blank|tile-boundary",
        default_value = "channels",
        help = "Probe scene case"
    )]
    probe_case: String,
    #[arg(long, help = "Also emit terminal-cell probe artifacts when supported")]
    probe_terminal: bool,
    #[arg(long, help = "Also emit Kitty-ready raw RGBA probe payload artifacts")]
    probe_kitty_payload: bool,
    #[arg(long, help = "Exit nonzero when CPU/Metal probe comparison mismatches")]
    probe_fail_on_mismatch: bool,
    #[cfg(feature = "metal")]
    #[arg(long, help = "Force CPU rendering", conflicts_with = "metal")]
    cpu: bool,
    #[cfg(feature = "metal")]
    #[arg(long, help = "Force Metal GPU rendering", conflicts_with = "cpu")]
    metal: bool,
    #[arg(long, help = "Flip Y axis")]
    flip_y: bool,
    #[arg(long, help = "Flip Z axis")]
    flip_z: bool,
    #[arg(long, help = "Run built-in demo scene", conflicts_with = "input")]
    demo: bool,
    #[arg(
        long,
        value_name = "N",
        default_value_t = 1,
        help = "Supersampling factor"
    )]
    supersample: u32,
}

fn find_luigi_ply() -> Option<PathBuf> {
    // 1. Check relative to cwd
    let cwd_candidate = PathBuf::from("scenes/luigi.ply");
    if cwd_candidate.exists() {
        return Some(cwd_candidate);
    }
    // 2. Check next to the executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let exe_candidate = exe_dir.join("scenes/luigi.ply");
            if exe_candidate.exists() {
                return Some(exe_candidate);
            }
        }
    }
    None
}

fn load_splats_from_cli(cli: &Cli) -> AppResult<Vec<splat::Splat>> {
    if cli.demo {
        // Try to load luigi.ply; fall back to procedural demo if not found
        if let Some(luigi_path) = find_luigi_ply() {
            let path_str = luigi_path.to_str().ok_or("luigi.ply path is non-UTF-8")?;
            return parser::ply::load_ply_file(path_str);
        }
        return Ok(demo::generate_demo_splats());
    }

    let path = cli
        .input
        .as_ref()
        .expect("input is Some; checked before dispatch");

    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let path_str = path.to_str().ok_or_else(|| {
        format!(
            "Input path contains non-UTF-8 characters: {}",
            path.display()
        )
    })?;

    match ext.as_str() {
        "ply" => parser::ply::load_ply_file(path_str),
        "splat" => parser::dot_splat::load_splat_file(path_str),
        _ => Err(format!(
            "Unsupported input '{}'. Use a .ply, .splat, or --demo",
            path.display()
        )
        .into()),
    }
}

fn parse_probe_size(raw: &str) -> AppResult<(usize, usize)> {
    let (w, h) = raw
        .split_once('x')
        .or_else(|| raw.split_once('X'))
        .ok_or_else(|| format!("Invalid --probe-size '{raw}'. Expected WxH, for example 64x48"))?;
    let width = w
        .parse::<usize>()
        .map_err(|_| format!("Invalid probe width '{w}' in --probe-size '{raw}'"))?;
    let height = h
        .parse::<usize>()
        .map_err(|_| format!("Invalid probe height '{h}' in --probe-size '{raw}'"))?;
    if width == 0 || height == 0 {
        return Err("Probe size must be non-zero".into());
    }
    Ok((width, height))
}

fn parse_vec3(raw: &str, flag: &str) -> AppResult<Vec3> {
    let parts = raw
        .split(',')
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("Invalid {flag} '{raw}'. Expected x,y,z"))?;
    if parts.len() != 3 {
        return Err(format!("Invalid {flag} '{raw}'. Expected x,y,z").into());
    }
    Ok(Vec3::new(parts[0], parts[1], parts[2]))
}

fn parse_yaw_pitch(raw: &str) -> AppResult<(f32, f32)> {
    let parts = raw
        .split(',')
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("Invalid --probe-yaw-pitch '{raw}'. Expected yaw,pitch"))?;
    if parts.len() != 2 {
        return Err(format!("Invalid --probe-yaw-pitch '{raw}'. Expected yaw,pitch").into());
    }
    Ok((parts[0], parts[1]))
}

fn validate_probe_fov_deg(deg: f32) -> AppResult<()> {
    if !deg.is_finite() || deg <= 0.0 || deg >= 180.0 {
        return Err("--probe-fov-deg must be finite and greater than 0 and less than 180".into());
    }
    Ok(())
}

fn validate_probe_inspect_scale(scale: usize) -> AppResult<()> {
    if scale == 0 {
        return Err("--probe-inspect-scale must be greater than 0".into());
    }
    Ok(())
}

fn run_render_probe(cli: &Cli, out_dir: PathBuf) -> AppResult<()> {
    let (width, height) = parse_probe_size(&cli.probe_size)?;
    validate_probe_fov_deg(cli.probe_fov_deg)?;
    validate_probe_inspect_scale(cli.probe_inspect_scale)?;
    let backend = cli
        .probe_backends
        .parse::<render::probe::ProbeBackendSelection>()?;
    let case = cli.probe_case.parse::<render::probe::ProbeCase>()?;
    let position_raw = cli
        .probe_camera_pos
        .as_deref()
        .ok_or("--probe-camera-pos is required with --render-probe")?;
    let position = parse_vec3(position_raw, "--probe-camera-pos")?;
    let target = parse_vec3(&cli.probe_look_at, "--probe-look-at")?;

    let mut camera_spec = render::probe::ProbeCameraSpec {
        position,
        target,
        fov: cli.probe_fov_deg.to_radians(),
        ..render::probe::ProbeCameraSpec::default()
    };
    if let Some(yaw_pitch) = cli.probe_yaw_pitch.as_deref() {
        let (yaw, pitch) = parse_yaw_pitch(yaw_pitch)?;
        let mut camera = Camera::new(position, yaw, pitch);
        camera.fov = camera_spec.fov;
        camera.near = camera_spec.near;
        camera.far = camera_spec.far;
        camera_spec.target = camera.position + camera.forward;
    }

    let loaded_splats = if case == render::probe::ProbeCase::Loaded {
        if cli.input.is_none() && !cli.demo {
            return Err("--probe-case loaded requires an input path or --demo".into());
        }
        let mut splats = load_splats_from_cli(cli)?;
        if cli.flip_y || cli.flip_z {
            for splat in &mut splats {
                if cli.flip_y {
                    splat.position.y = -splat.position.y;
                }
                if cli.flip_z {
                    splat.position.z = -splat.position.z;
                }
            }
        }
        splats
    } else {
        Vec::new()
    };

    let mut config = render::probe::ProbeConfig::new(out_dir);
    config.width = width;
    config.height = height;
    config.backend = backend;
    config.case = case;
    config.camera = camera_spec;
    config.frames = cli.probe_benchmark_frames.unwrap_or(cli.probe_frames);
    config.warmup_frames = cli.probe_warmup;
    config.terminal_artifacts = cli.probe_terminal;
    config.kitty_artifacts = cli.probe_kitty_payload;
    config.inspect_scale = cli.probe_inspect_scale;
    config.stage_telemetry = cli.probe_stage_telemetry;
    config.timing = cli.probe_timing || cli.probe_benchmark_frames.is_some();

    let result = render::probe::run_probe(&config, &loaded_splats)?;
    let mismatch_count = result
        .diff_frames
        .iter()
        .filter(|frame| {
            frame.metrics.classification != render::probe::ProbeDiffClassification::Pass
        })
        .count();
    println!(
        "{{\"status\":\"ok\",\"manifest\":\"{}\",\"contact_sheet\":\"{}\",\"timing\":\"{}\",\"frames\":{},\"mismatches\":{}}}",
        result.manifest_path.display(),
        result.contact_sheet_path.display(),
        result
            .timing_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        result.cpu_frames.len().max(result.metal_frames.len()),
        mismatch_count
    );

    if cli.probe_fail_on_mismatch {
        if mismatch_count > 0 {
            return Err(format!(
                "Probe comparison failed with {mismatch_count} mismatched frame(s)"
            )
            .into());
        }
    }

    Ok(())
}

fn main() -> AppResult<()> {
    install_panic_hook();
    let cli = Cli::parse();

    if let Some(out_dir) = cli.render_probe.clone() {
        return run_render_probe(&cli, out_dir);
    }

    if cli.input.is_none() && !cli.demo {
        Cli::command().print_help()?;
        println!();
        std::process::exit(0);
    }

    #[cfg(feature = "metal")]
    let mut backend = if cli.cpu {
        Backend::Cpu
    } else {
        Backend::Metal
    };
    #[cfg(not(feature = "metal"))]
    let backend = Backend::Cpu;

    let mut splats = load_splats_from_cli(&cli)?;
    if cli.flip_y || cli.flip_z {
        for splat in &mut splats {
            if cli.flip_y {
                splat.position.y = -splat.position.y;
            }
            if cli.flip_z {
                splat.position.z = -splat.position.z;
            }
        }
    }

    let use_truecolor = match std::env::var("COLORTERM") {
        Ok(val) => !val.is_empty() && (val == "truecolor" || val == "24bit"),
        Err(_) => match std::env::var("TERM_PROGRAM") {
            Ok(prog) => prog != "Apple_Terminal",
            Err(_) => match std::env::var("TERM") {
                Ok(term) => {
                    term.contains("ghostty") || term.contains("kitty") || term.contains("wezterm")
                }
                Err(_) => false,
            },
        },
    };

    let (cols, rows) = terminal::size().unwrap_or((120, 40));
    let width = cols.max(1) as usize;
    let height = rows.max(1) as usize * 2;

    let mut camera = Camera::new(Vec3::new(0.0, 0.0, 5.0), -std::f32::consts::FRAC_PI_2, 0.0);
    camera::look_at_target(&mut camera, Vec3::ZERO);

    #[cfg(feature = "metal")]
    let mut metal_backend = if backend == Backend::Metal {
        match render::metal::MetalBackend::new(splats.len()) {
            Ok(mut mb) => {
                mb.upload_splats(&splats)?;
                Some(mb)
            }
            Err(err) => {
                eprintln!(
                    "Warning: Metal initialization failed: {}. Falling back to CPU renderer.",
                    err
                );
                backend = Backend::Cpu;
                None
            }
        }
    } else {
        None
    };

    let mut app_state = AppState {
        camera,
        splats,
        projected_splats: Vec::with_capacity(32_768),
        render_state: RenderState {
            framebuffer: vec![[0, 0, 0]; width * height],
            alpha_buffer: vec![0.0; width * height],
            depth_buffer: vec![f32::INFINITY; width * height],
            width,
            height,
        },
        halfblock_cells: Vec::with_capacity(width * rows.max(1) as usize),
        hud_string_buf: String::with_capacity(512),
        input_state: input::state::InputState::default(),
        show_hud: true,
        camera_mode: CameraMode::Free,
        move_speed: 0.15,
        frame_count: 0,
        last_frame_time: Instant::now(),
        fps: 0.0,
        visible_splat_count: 0,
        orbit_angle: 0.0,
        orbit_radius: 5.0,
        orbit_height: 0.0,
        orbit_target: Vec3::ZERO,
        supersample_factor: cli.supersample.max(1),
        render_mode: RenderMode::Halfblock,
        backend,
        use_truecolor,
        #[cfg(feature = "metal")]
        metal_backend: metal_backend.take(),
        #[cfg(feature = "metal")]
        last_gpu_error: None,
        #[cfg(feature = "metal")]
        gpu_fallback_active: false,
    };

    crossterm::terminal::enable_raw_mode()?;
    let input_rx = input::thread::spawn_input_thread();
    let mut stdout = BufWriter::with_capacity(1024 * 1024, io::stdout());

    execute!(
        stdout,
        EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All)
    )?;
    // Request key event kinds so key releases are observable for held-key movement.
    let _ = execute!(
        stdout,
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
        )
    );
    stdout.flush()?;

    let run_result = run_app_loop(&mut app_state, &input_rx, &mut stdout);
    #[cfg(feature = "metal")]
    let cleanup_result = cleanup_terminal(&mut stdout, app_state.last_gpu_error.as_deref());
    #[cfg(not(feature = "metal"))]
    let cleanup_result = cleanup_terminal(&mut stdout, None);

    run_result?;
    cleanup_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_fov_deg_validation_rejects_invalid_values() {
        for value in [0.0, 180.0, -1.0, f32::NAN, f32::INFINITY] {
            let err = validate_probe_fov_deg(value).unwrap_err();
            assert!(err.to_string().contains("--probe-fov-deg"));
        }
        validate_probe_fov_deg(35.0).unwrap();
    }

    #[test]
    fn probe_inspect_scale_validation_rejects_zero() {
        let err = validate_probe_inspect_scale(0).unwrap_err();
        assert!(err.to_string().contains("--probe-inspect-scale"));
        validate_probe_inspect_scale(1).unwrap();
        validate_probe_inspect_scale(4).unwrap();
    }
}
