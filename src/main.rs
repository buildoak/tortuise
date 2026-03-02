use clap::{CommandFactory, Parser, Subcommand};
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
#[cfg(feature = "sharp")]
mod export;
mod input;
mod math;
mod parser;
mod render;
#[cfg(feature = "sharp")]
mod sharp;
mod sort;
mod splat;
mod terminal_setup;

use camera::Camera;
use math::Vec3;
use render::frame::run_app_loop;
use render::{AppState, Backend, CameraMode, RenderMode, RenderState};
use terminal_setup::{cleanup_terminal, install_panic_hook};

pub type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "sharp")]
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp"];

#[derive(Debug, Parser)]
#[command(
    name = "tortuise",
    version,
    about = "Terminal-native 3D Gaussian Splatting viewer"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    /// Path to a .ply or .splat scene file
    input: Option<PathBuf>,
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

#[derive(Debug, Subcommand)]
enum Commands {
    /// Convert a supported image into SHARP-generated .ply Gaussian splats
    Convert {
        /// Input image path (.jpg, .jpeg, .png, .webp)
        input: PathBuf,
        /// Output .ply path (defaults to input path with .ply extension)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
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
        #[cfg(feature = "sharp")]
        ext if IMAGE_EXTENSIONS.contains(&ext) => {
            eprintln!("Reconstructing 3D scene from image...");
            sharp::reconstruct_from_image(path).map_err(|e| e.into())
        }
        #[cfg(not(feature = "sharp"))]
        ext if ["jpg", "jpeg", "png", "webp"].contains(&ext) => {
            Err("Image input requires the 'sharp' feature. Rebuild with: cargo install tortuise --features sharp".into())
        }
        _ => Err(format!(
            "Unsupported input '{}'. Use a .ply, .splat, or --demo",
            path.display()
        )
        .into()),
    }
}

fn main() -> AppResult<()> {
    install_panic_hook();
    let cli = Cli::parse();

    #[cfg(feature = "sharp")]
    if let Some(Commands::Convert { input, output }) = &cli.command {
        eprintln!("Reconstructing 3D scene from image...");
        let splats = sharp::reconstruct_from_image(input)?;
        let output_path = output.clone().unwrap_or_else(|| {
            let mut out = input.clone();
            out.set_extension("ply");
            out
        });
        export::ply::save_ply(&splats, &output_path)?;
        println!(
            "Converted '{}' -> '{}' ({} splats)",
            input.display(),
            output_path.display(),
            splats.len()
        );
        return Ok(());
    }
    #[cfg(not(feature = "sharp"))]
    if matches!(cli.command, Some(Commands::Convert { .. })) {
        return Err(
            "The 'convert' subcommand requires the 'sharp' feature. Rebuild with: cargo install tortuise --features sharp"
                .into(),
        );
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
