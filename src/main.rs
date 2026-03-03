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
#[cfg(feature = "sharp")]
mod spinner;
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
    #[arg(long, help = "Flip X axis")]
    flip_x: bool,
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
        /// Input image path (.jpg, .jpeg, .png, .webp, .heic)
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
        "jpg" | "jpeg" | "png" | "webp" | "heic" => {
            let filename = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy();
            eprintln!("Image files require conversion to 3DGS first.\n");
            eprintln!("  tortuise convert {} -o scene.ply", filename);
            eprintln!("  tortuise scene.ply");
            std::process::exit(1);
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
        if !input.exists() {
            eprintln!("Error: file not found: {}", input.display());
            std::process::exit(1);
        }

        // If the input is HEIC, convert to PNG via macOS sips first.
        let heic_tmp: Option<PathBuf>;
        let effective_input: &std::path::Path;

        let ext = input
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if ext == "heic" {
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image");
            let tmp_path =
                std::env::temp_dir().join(format!("tortuise_heic_{}.png", stem));

            let sips_result = std::process::Command::new("sips")
                .args(["--setProperty", "format", "png"])
                .arg(input)
                .arg("--out")
                .arg(&tmp_path)
                .output();

            match sips_result {
                Ok(out) if out.status.success() => {
                    // iPhone HEIC files typically use the Display P3 color space.
                    // SHARP was trained on sRGB images, so feeding P3-gamut pixel
                    // values produces degenerate output (near-zero positions,
                    // uniform tiny scales → black screen).  Convert the pixel data
                    // to sRGB before inference.
                    let srgb_profile = "/System/Library/ColorSync/Profiles/sRGB Profile.icc";
                    if std::path::Path::new(srgb_profile).exists() {
                        let srgb = std::process::Command::new("sips")
                            .args(["--matchTo", srgb_profile])
                            .arg(&tmp_path)
                            .arg("--out")
                            .arg(&tmp_path)
                            .output();
                        if let Ok(s) = &srgb {
                            if !s.status.success() {
                                eprintln!(
                                    "sips sRGB conversion warning: {}",
                                    String::from_utf8_lossy(&s.stderr)
                                );
                            }
                        }
                    }

                    // HEIC files store landscape pixel data with EXIF orientation
                    // metadata (e.g. Orientation 6 = 90° CW for portrait photos).
                    // The format conversion above may preserve this tag without
                    // rotating the actual pixels.  Since the `image` crate does
                    // NOT auto-apply EXIF orientation from PNGs, we use `sips -r 0`
                    // to bake any pending orientation into the pixel data (a 0°
                    // rotation that forces sips to apply the EXIF transform).
                    let bake = std::process::Command::new("sips")
                        .args(["-r", "0"])
                        .arg(&tmp_path)
                        .output();
                    if let Ok(b) = &bake {
                        if !b.status.success() {
                            eprintln!(
                                "sips orientation bake warning: {}",
                                String::from_utf8_lossy(&b.stderr)
                            );
                        }
                    }
                    heic_tmp = Some(tmp_path);
                }
                Ok(out) => {
                    eprintln!(
                        "sips failed to convert HEIC: {}",
                        String::from_utf8_lossy(&out.stderr)
                    );
                    std::process::exit(1);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    eprintln!(
                        "HEIC conversion requires macOS 'sips' utility \
                         (not available on this platform). \
                         Convert your image to PNG or JPEG first."
                    );
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Failed to run sips: {}", e);
                    std::process::exit(1);
                }
            }
            effective_input = heic_tmp.as_deref().unwrap();
        } else {
            heic_tmp = None;
            effective_input = input;
        }

        // Wrap the rest in a closure so we can clean up the temp file on any exit path.
        let result = (|| -> AppResult<()> {
            // Phase 1: Ensure model is downloaded (shows its own progress UX).
            sharp::ensure_model_downloaded()?;

            // Phase 2: Inference with Matrix rain animation.
            let sp = spinner::Spinner::start("Reconstructing 3D scene from image...");
            let splats = sharp::reconstruct_from_image(effective_input)?;
            sp.finish(&format!("Reconstructed {} Gaussians", splats.len()));

            let output_path = output.clone().unwrap_or_else(|| {
                let mut out = input.clone();
                out.set_extension("ply");
                out
            });

            let sp = spinner::Spinner::start(&format!("Saving to {}...", output_path.display()));
            export::ply::save_ply(&splats, &output_path)?;
            sp.finish(&format!(
                "Saved {} splats to '{}'",
                splats.len(),
                output_path.display()
            ));
            Ok(())
        })();

        // Clean up temp HEIC-converted file regardless of success/failure.
        if let Some(ref tmp) = heic_tmp {
            let _ = std::fs::remove_file(tmp);
        }

        return result.map_err(Into::into);
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
    if cli.flip_x || cli.flip_y || cli.flip_z {
        for splat in &mut splats {
            if cli.flip_x {
                splat.position.x = -splat.position.x;
            }
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

    // Position camera to frame the loaded scene: place it at centroid + offset
    // along +Z, looking at the centroid. Falls back to default if scene is empty.
    let bounds = splat::compute_scene_bounds(&splats);
    let (scene_center, scene_radius) = match &bounds {
        Some(b) => (b.centroid, b.extent.max(0.1)),
        None => (Vec3::ZERO, 2.5),
    };
    let cam_distance = scene_radius * 2.5;
    let cam_start = Vec3::new(
        scene_center.x,
        scene_center.y,
        scene_center.z + cam_distance,
    );
    let mut camera = Camera::new(cam_start, -std::f32::consts::FRAC_PI_2, 0.0);
    camera::look_at_target(&mut camera, scene_center);

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
        orbit_radius: cam_distance,
        orbit_height: 0.0,
        orbit_target: scene_center,
        scene_center,
        initial_cam_distance: cam_distance,
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
