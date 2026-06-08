use crossterm::{queue, style::ResetColor, terminal};
use std::io::{self, Write};
use std::time::Instant;

use super::{
    live_telemetry::LiveFrameTelemetry, AppResult, AppState, CameraMode, RenderMode, FRAME_TARGET,
};

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

fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(feature = "metal")]
fn kitty_scale_divisor_for_telemetry() -> usize {
    std::env::var("TORTUISE_KITTY_SCALE_DIVISOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

#[cfg(feature = "metal")]
fn kitty_frame_ms_for_telemetry() -> u64 {
    std::env::var("TORTUISE_KITTY_FRAME_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(33)
}

#[cfg(feature = "metal")]
fn metal_gpu_wait_ms(app_state: &AppState) -> f64 {
    app_state
        .metal_backend
        .as_ref()
        .map(|mb| {
            let telemetry = mb.probe_telemetry();
            telemetry.stage_timings[..telemetry.stage_timing_count]
                .iter()
                .map(|stage| stage.wait_ms)
                .sum()
        })
        .unwrap_or(0.0)
        .max(0.0)
}

fn build_live_frame_telemetry(
    app_state: &AppState,
    input_stats: crate::input::InputDrainStats,
    target: std::time::Duration,
    frame_ms: f64,
    render_ms: f64,
    flush_ms: f64,
    sleep_ms: f64,
) -> LiveFrameTelemetry {
    #[allow(unused_mut)]
    let mut telemetry = LiveFrameTelemetry {
        frame: app_state.frame_count,
        frame_ms,
        target_ms: duration_ms(target),
        sleep_ms,
        input_events: input_stats.events,
        oldest_input_age_ms: input_stats.oldest_age_ms,
        render_ms,
        flush_ms,
        effective_path: app_state.effective_render_path,
        render_width: app_state.last_render_width,
        render_height: app_state.last_render_height,
        terminal_cols: app_state.last_terminal_cols,
        terminal_rows: app_state.last_terminal_rows,
        camera_x: app_state.camera.position.x as f64,
        camera_y: app_state.camera.position.y as f64,
        camera_z: app_state.camera.position.z as f64,
        camera_yaw: app_state.camera.yaw as f64,
        camera_pitch: app_state.camera.pitch as f64,
        camera_fov_deg: app_state.camera.fov.to_degrees() as f64,
        source_splat_count: app_state.splats.len(),
        active_splat_count: app_state.splats.len(),
        valid_count: app_state.visible_splat_count,
        previous_telemetry_write_ms: app_state.last_telemetry_write_ms as f64,
        ..LiveFrameTelemetry::default()
    };

    #[cfg(feature = "metal")]
    {
        telemetry.gpu_wait_ms = metal_gpu_wait_ms(app_state);
        telemetry.convert_ms = app_state.kitty_convert_ms as f64;
        telemetry.encode_ms = app_state.kitty_encode_ms as f64;
        telemetry.write_ms = app_state.kitty_write_ms as f64;
        telemetry.payload_bytes = app_state.kitty_payload_bytes;
        telemetry.base64_bytes = app_state.kitty_base64_bytes;
        telemetry.chunks = app_state.kitty_chunks;
        telemetry.kitty_format = match std::env::var("TORTUISE_KITTY_FORMAT") {
            Ok(value) if value.trim() == "rgb" => "rgb",
            _ => "rgba",
        };
        telemetry.kitty_scale_divisor = kitty_scale_divisor_for_telemetry();
        telemetry.kitty_frame_ms = kitty_frame_ms_for_telemetry();
        if app_state.effective_render_path.starts_with("metal_") {
            let Some(mb) = app_state.metal_backend.as_ref() else {
                return telemetry;
            };
            let metal = mb.probe_telemetry();
            telemetry.quality = match std::env::var("TORTUISE_METAL_QUALITY") {
                Ok(value) => match value.trim() {
                    "fast-preview" => "fast-preview",
                    "turbo" => "turbo",
                    _ => "exact",
                },
                Err(_) => "exact",
            };
            telemetry.sort_path = metal.sort_path;
            telemetry.lod_mode = metal.lod_mode;
            telemetry.lod_mapping = metal.lod_mapping;
            telemetry.source_splat_count = metal.source_splat_count;
            telemetry.active_splat_count = metal.active_splat_count;
            telemetry.valid_count = metal.valid_count as usize;
            telemetry.estimated_overlaps = metal.estimated_overlaps;
            telemetry.attempt_sort_count = metal.attempt_sort_count;
            telemetry.actual_total_overlaps = metal.actual_total_overlaps;
            telemetry.overflow_flag = metal.overflow_flag;
            telemetry.retry_count = metal.retry_count;
            telemetry.tile_entries = metal.tile_density.total_tile_entries;
            telemetry.max_tile_range = metal.tile_density.max_tile_range;
            telemetry.p95_tile_range = metal.tile_density.p95_tile_range;
            telemetry.p99_tile_range = metal.tile_density.p99_tile_range;
            telemetry.stage_timing_count = metal.stage_timing_count;
        }
    }

    telemetry
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
    app_state.last_terminal_cols = term_cols;
    app_state.last_terminal_rows = term_rows;

    #[cfg(feature = "metal")]
    if app_state.render_mode != RenderMode::Kitty
        && (app_state.kitty_payload_bytes > 0 || app_state.kitty_pixel_hud_active)
    {
        super::frame_kitty::delete_kitty_image(stdout, 1)?;
        super::frame_kitty::delete_kitty_image(stdout, 2)?;
        super::frame_kitty::delete_kitty_hud_images(stdout)?;
        app_state.kitty_image_id = 1;
        app_state.kitty_visible_image_id = 0;
        app_state.kitty_pixel_hud_active = false;
        app_state.kitty_payload_bytes = 0;
        app_state.kitty_base64_bytes = 0;
        app_state.kitty_chunks = 0;
        app_state.kitty_write_ms = 0.0;
        app_state.kitty_convert_ms = 0.0;
        app_state.kitty_encode_ms = 0.0;
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
            app_state.effective_render_path = "cpu_text";
            let proj_w = term_cols;
            let proj_h = term_rows * 2;
            app_state.last_render_width = proj_w;
            app_state.last_render_height = proj_h;
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

    let draw_terminal_hud = app_state.show_hud && {
        #[cfg(feature = "metal")]
        {
            app_state.render_mode != RenderMode::Kitty || !app_state.kitty_pixel_hud_active
        }
        #[cfg(not(feature = "metal"))]
        {
            true
        }
    };
    if draw_terminal_hud {
        super::hud::draw_hud(app_state, cols, rows, ss, stdout)?;
    }

    queue!(stdout, ResetColor)?;
    Ok(())
}

pub fn run_app_loop(
    app_state: &mut AppState,
    input_rx: &crate::input::thread::InputReceiver,
    stdout: &mut io::BufWriter<io::Stdout>,
) -> AppResult<()> {
    loop {
        let frame_start = Instant::now();

        let input_stats = crate::input::drain_input_events_with_stats(app_state, input_rx)?;
        if input_stats.quit_requested {
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
        let render_start = Instant::now();
        render_frame(app_state, terminal_size, stdout)?;
        let render_ms = duration_ms(render_start.elapsed());

        let flush_start = Instant::now();
        stdout.flush()?;
        let flush_ms = duration_ms(flush_start.elapsed());
        app_state.last_flush_ms = flush_ms as f32;

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
        let mut sleep_ms = 0.0;
        if spent < target {
            let sleep_for = target - spent;
            let sleep_start = Instant::now();
            std::thread::sleep(sleep_for);
            sleep_ms = duration_ms(sleep_start.elapsed());
        }
        let frame_ms = duration_ms(frame_start.elapsed());
        if app_state.live_telemetry.is_enabled() {
            let live_frame = build_live_frame_telemetry(
                app_state,
                input_stats,
                target,
                frame_ms,
                render_ms,
                flush_ms,
                sleep_ms,
            );
            app_state.last_telemetry_write_ms = app_state.live_telemetry.record(live_frame)? as f32;
        }
    }

    Ok(())
}
