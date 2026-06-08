use crossterm::{queue, style::ResetColor, terminal};
use std::io::{self, Write};
use std::time::Instant;

use super::{AppResult, AppState, CameraMode, RenderMode, FRAME_TARGET};

const HALFBLOCK_FRAME_TARGET: std::time::Duration = std::time::Duration::from_millis(33);
#[cfg(feature = "metal")]
const DEFAULT_KITTY_FRAME_TARGET: std::time::Duration = std::time::Duration::from_millis(33);

#[cfg(feature = "metal")]
fn kitty_frame_target() -> std::time::Duration {
    std::env::var("TORTUISE_KITTY_FRAME_MS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .map(std::time::Duration::from_millis)
        .unwrap_or(DEFAULT_KITTY_FRAME_TARGET)
}

fn update_orbit(app_state: &mut AppState, delta_time: f32) {
    let orbit_speed = 0.9 * app_state.move_speed;
    app_state.orbit_angle += orbit_speed * delta_time;

    let target = app_state.orbit_target;
    app_state.camera.position.x = target.x + app_state.orbit_radius * app_state.orbit_angle.cos();
    app_state.camera.position.z = target.z + app_state.orbit_radius * app_state.orbit_angle.sin();
    app_state.camera.position.y = target.y + app_state.orbit_height;

    crate::camera::look_at_target(&mut app_state.camera, target);
}

pub fn render_frame(
    app_state: &mut AppState,
    terminal_size: (u16, u16),
    stdout: &mut impl Write,
) -> io::Result<()> {
    let cols = terminal_size.0.max(1);
    let rows = terminal_size.1.max(1);
    let term_cols = cols as usize;
    let term_rows = rows as usize;
    let ss = app_state.supersample_factor as usize;

    #[cfg(feature = "metal")]
    if app_state.render_mode != RenderMode::Kitty && app_state.kitty_payload_bytes > 0 {
        super::frame_kitty::delete_kitty_image(stdout, 1)?;
        super::frame_kitty::delete_kitty_image(stdout, 2)?;
        app_state.kitty_image_id = 1;
        app_state.kitty_visible_image_id = 0;
        app_state.kitty_payload_bytes = 0;
        app_state.kitty_base64_bytes = 0;
        app_state.kitty_chunks = 0;
        app_state.kitty_write_ms = 0.0;
    }

    match app_state.render_mode {
        RenderMode::Halfblock => {
            super::frame_halfblock::render_halfblock_frame(
                app_state, term_cols, term_rows, stdout,
            )?;
        }
        #[cfg(feature = "metal")]
        RenderMode::Kitty => {
            super::frame_kitty::render_kitty_frame(app_state, term_cols, term_rows, stdout)?;
        }
        RenderMode::PointCloud
        | RenderMode::Matrix
        | RenderMode::BlockDensity
        | RenderMode::Braille
        | RenderMode::AsciiClassic => {
            let proj_w = term_cols;
            let proj_h = term_rows * 2;
            super::pipeline::cpu_project_and_sort(app_state, proj_w, proj_h);

            match app_state.render_mode {
                RenderMode::PointCloud => super::modes::point_cloud::render_point_cloud(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                    app_state.use_truecolor,
                )?,
                RenderMode::Matrix => super::modes::matrix::render_matrix(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                    app_state.use_truecolor,
                )?,
                RenderMode::BlockDensity => super::modes::block_density::render_block_density(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                    app_state.use_truecolor,
                )?,
                RenderMode::Braille => super::modes::braille::render_braille(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                    app_state.use_truecolor,
                )?,
                RenderMode::AsciiClassic => super::modes::ascii::render_ascii_classic(
                    &app_state.projected_splats,
                    term_cols,
                    term_rows,
                    proj_h,
                    stdout,
                    app_state.show_hud,
                    app_state.use_truecolor,
                )?,
                _ => unreachable!(),
            }
        }
    }

    if app_state.show_hud {
        super::hud::draw_hud(app_state, cols, rows, ss, stdout)?;
    }

    queue!(stdout, ResetColor)?;
    stdout.flush()
}

pub fn run_app_loop(
    app_state: &mut AppState,
    input_rx: &crate::input::thread::InputReceiver,
    stdout: &mut io::BufWriter<io::Stdout>,
) -> AppResult<()> {
    loop {
        let frame_start = Instant::now();

        if crate::input::drain_input_events(app_state, input_rx)? {
            break;
        }

        let now = Instant::now();
        let delta_time = now
            .duration_since(app_state.last_frame_time)
            .as_secs_f32()
            .max(1e-6);
        app_state.last_frame_time = now;

        match app_state.camera_mode {
            CameraMode::Orbit => update_orbit(app_state, delta_time),
            CameraMode::Free => {
                crate::input::state::apply_movement_from_held_keys(app_state, delta_time);
            }
        }

        let terminal_size = terminal::size()?;
        render_frame(app_state, terminal_size, stdout)?;

        app_state.frame_count += 1;
        let instant_fps = 1.0 / delta_time;
        app_state.fps = if app_state.fps <= 0.01 {
            instant_fps
        } else {
            0.90 * app_state.fps + 0.10 * instant_fps
        };

        let spent = frame_start.elapsed();
        let target = match app_state.render_mode {
            RenderMode::Halfblock => HALFBLOCK_FRAME_TARGET,
            #[cfg(feature = "metal")]
            RenderMode::Kitty => kitty_frame_target(),
            _ => FRAME_TARGET,
        };
        if spent < target {
            std::thread::sleep(target - spent);
        }
    }

    Ok(())
}
