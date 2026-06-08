use crossterm::{
    cursor,
    style::{SetBackgroundColor, SetForegroundColor},
    Command,
};
use std::io::{self, Write};

#[cfg(feature = "metal")]
use super::Backend;
use super::{make_color, AppState, HalfblockCell, HALF_BLOCK};

#[cfg(feature = "metal")]
fn rgb_from_packed_pixel(pixel: u32) -> [u8; 3] {
    [
        (pixel & 0xFF) as u8,
        ((pixel >> 8) & 0xFF) as u8,
        ((pixel >> 16) & 0xFF) as u8,
    ]
}

fn build_halfblock_cells_rgb(
    fb: &[[u8; 3]],
    ss_width: usize,
    ss_height: usize,
    term_cols: usize,
    term_rows: usize,
    ss: usize,
    out: &mut Vec<HalfblockCell>,
) {
    if ss == 1 {
        out.clear();
        out.resize(term_cols * term_rows, ([0u8; 3], [0u8; 3]));
        for term_row in 0..term_rows {
            let top_y = term_row * 2;
            let bot_y = top_y + 1;
            for x in 0..term_cols {
                let top = fb[top_y * ss_width + x];
                let bot = if bot_y < ss_height {
                    fb[bot_y * ss_width + x]
                } else {
                    [0, 0, 0]
                };
                out[term_row * term_cols + x] = (top, bot);
            }
        }
    } else {
        super::modes::halfblock::downsample_to_terminal_into(
            fb, ss_width, ss_height, term_cols, term_rows, ss, out,
        );
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::rgb_from_packed_pixel;

    #[test]
    fn packed_pixel_decode_matches_probe_low_byte_red_layout() {
        assert_eq!(rgb_from_packed_pixel(0x000000ff), [255, 0, 0]);
        assert_eq!(rgb_from_packed_pixel(0x0000ff00), [0, 255, 0]);
        assert_eq!(rgb_from_packed_pixel(0x00ff0000), [0, 0, 255]);
    }
}

#[cfg(feature = "metal")]
fn build_halfblock_cells_packed(
    packed_fb: &[u32],
    ss_width: usize,
    ss_height: usize,
    term_cols: usize,
    term_rows: usize,
    ss: usize,
    out: &mut Vec<HalfblockCell>,
) {
    if ss == 1 {
        out.clear();
        out.resize(term_cols * term_rows, ([0u8; 3], [0u8; 3]));
        for term_row in 0..term_rows {
            let top_y = term_row * 2;
            let bot_y = top_y + 1;
            for x in 0..term_cols {
                let top_idx = top_y * ss_width + x;
                let bot_idx = bot_y * ss_width + x;
                let top = packed_fb
                    .get(top_idx)
                    .copied()
                    .map(rgb_from_packed_pixel)
                    .unwrap_or([0, 0, 0]);
                let bot = if bot_y < ss_height {
                    packed_fb
                        .get(bot_idx)
                        .copied()
                        .map(rgb_from_packed_pixel)
                        .unwrap_or([0, 0, 0])
                } else {
                    [0, 0, 0]
                };
                out[term_row * term_cols + x] = (top, bot);
            }
        }
    } else {
        super::modes::halfblock::downsample_packed_to_terminal_into(
            packed_fb, ss_width, ss_height, term_cols, term_rows, ss, out,
        );
    }
}

fn write_ansi_command(buf: &mut String, command: impl Command) -> io::Result<()> {
    command
        .write_ansi(buf)
        .map_err(|_| io::Error::other("failed to encode ANSI command"))
}

fn write_halfblock_cells_ansi(
    cells: &[HalfblockCell],
    term_cols: usize,
    term_rows: usize,
    show_hud: bool,
    use_truecolor: bool,
    stdout: &mut impl Write,
) -> io::Result<()> {
    let mut last_bg: Option<(u8, u8, u8)> = None;
    let mut last_fg: Option<(u8, u8, u8)> = None;
    let mut row_buf = String::with_capacity(term_cols * 8 + 32);

    for term_row in 0..term_rows {
        if super::modes::is_hud_overlay_row(show_hud, term_row, term_rows) {
            last_bg = None;
            last_fg = None;
            continue;
        }

        row_buf.clear();
        write_ansi_command(&mut row_buf, cursor::MoveTo(0, term_row as u16))?;

        for x in 0..term_cols {
            let (top, bottom) = cells[term_row * term_cols + x];
            let bg = (top[0], top[1], top[2]);
            let fg = (bottom[0], bottom[1], bottom[2]);

            if last_bg != Some(bg) {
                write_ansi_command(
                    &mut row_buf,
                    SetBackgroundColor(make_color(bg.0, bg.1, bg.2, use_truecolor)),
                )?;
                last_bg = Some(bg);
            }
            if last_fg != Some(fg) {
                write_ansi_command(
                    &mut row_buf,
                    SetForegroundColor(make_color(fg.0, fg.1, fg.2, use_truecolor)),
                )?;
                last_fg = Some(fg);
            }
            row_buf.push(HALF_BLOCK);
        }

        stdout.write_all(row_buf.as_bytes())?;
    }

    Ok(())
}

#[cfg(feature = "metal")]
fn write_gpu_halfblock_cells_ansi(
    gpu_cells: &[crate::render::metal::GpuHalfblockCell],
    term_cols: usize,
    term_rows: usize,
    show_hud: bool,
    use_truecolor: bool,
    stdout: &mut impl Write,
) -> io::Result<()> {
    let mut last_bg: Option<(u8, u8, u8)> = None;
    let mut last_fg: Option<(u8, u8, u8)> = None;
    let mut row_buf = String::with_capacity(term_cols * 8 + 32);

    for term_row in 0..term_rows {
        if super::modes::is_hud_overlay_row(show_hud, term_row, term_rows) {
            last_bg = None;
            last_fg = None;
            continue;
        }

        row_buf.clear();
        write_ansi_command(&mut row_buf, cursor::MoveTo(0, term_row as u16))?;

        for x in 0..term_cols {
            let cell = gpu_cells[term_row * term_cols + x];
            let top = rgb_from_packed_pixel(cell.top_rgb);
            let bottom = rgb_from_packed_pixel(cell.bottom_rgb);
            let bg = (top[0], top[1], top[2]);
            let fg = (bottom[0], bottom[1], bottom[2]);

            if last_bg != Some(bg) {
                write_ansi_command(
                    &mut row_buf,
                    SetBackgroundColor(make_color(bg.0, bg.1, bg.2, use_truecolor)),
                )?;
                last_bg = Some(bg);
            }
            if last_fg != Some(fg) {
                write_ansi_command(
                    &mut row_buf,
                    SetForegroundColor(make_color(fg.0, fg.1, fg.2, use_truecolor)),
                )?;
                last_fg = Some(fg);
            }
            row_buf.push(HALF_BLOCK);
        }

        stdout.write_all(row_buf.as_bytes())?;
    }

    Ok(())
}

pub fn render_halfblock_frame(
    app_state: &mut AppState,
    term_cols: usize,
    term_rows: usize,
    stdout: &mut impl Write,
) -> io::Result<()> {
    let ss = app_state.supersample_factor as usize;
    let ss_width = term_cols * ss;
    let ss_height = term_rows * 2 * ss;

    super::pipeline::resize_render_state(&mut app_state.render_state, ss_width, ss_height);

    #[cfg(feature = "metal")]
    let gpu_rendered = if app_state.backend == Backend::Metal {
        gpu_render_to_framebuffer(app_state, ss_width, ss_height)
    } else {
        false
    };
    #[cfg(not(feature = "metal"))]
    let gpu_rendered = false;

    if !gpu_rendered {
        app_state.effective_render_path = "cpu_halfblock";
        super::pipeline::clear_framebuffer(&mut app_state.render_state);
        super::pipeline::cpu_project_and_sort(app_state, ss_width, ss_height);
        super::rasterizer::rasterize_splats(
            &app_state.projected_splats,
            &mut app_state.render_state,
            ss_width,
            ss_height,
        );
    }

    #[cfg(feature = "metal")]
    if gpu_rendered {
        app_state.effective_render_path = "metal_halfblock";
    }

    #[cfg(feature = "metal")]
    if gpu_rendered {
        let gpu_cells = app_state
            .metal_backend
            .as_ref()
            .map(|mb| mb.halfblock_cells_slice())
            .unwrap_or(&[]);
        if gpu_cells.len() == term_cols.saturating_mul(term_rows) {
            return write_gpu_halfblock_cells_ansi(
                gpu_cells,
                term_cols,
                term_rows,
                app_state.show_hud,
                app_state.use_truecolor,
                stdout,
            );
        } else {
            let packed = app_state
                .metal_backend
                .as_ref()
                .map(|mb| mb.framebuffer_slice())
                .unwrap_or(&[]);
            build_halfblock_cells_packed(
                packed,
                ss_width,
                ss_height,
                term_cols,
                term_rows,
                ss,
                &mut app_state.halfblock_cells,
            );
        }
    } else {
        build_halfblock_cells_rgb(
            &app_state.render_state.framebuffer,
            ss_width,
            ss_height,
            term_cols,
            term_rows,
            ss,
            &mut app_state.halfblock_cells,
        );
    }

    #[cfg(not(feature = "metal"))]
    build_halfblock_cells_rgb(
        &app_state.render_state.framebuffer,
        ss_width,
        ss_height,
        term_cols,
        term_rows,
        ss,
        &mut app_state.halfblock_cells,
    );

    let use_truecolor = app_state.use_truecolor;
    let show_hud = app_state.show_hud;
    let cells = &app_state.halfblock_cells;
    write_halfblock_cells_ansi(cells, term_cols, term_rows, show_hud, use_truecolor, stdout)
}

#[cfg(feature = "metal")]
fn gpu_render_to_framebuffer(app_state: &mut AppState, width: usize, height: usize) -> bool {
    let is_ready = match app_state.metal_backend.as_ref() {
        Some(mb) => mb.is_ready(),
        None => return false,
    };
    if !is_ready {
        return false;
    }

    let render_result: Result<(), crate::render::metal::MetalRenderError> =
        match app_state.metal_backend.as_mut() {
            Some(mb) => mb.render(
                &app_state.camera,
                width,
                height,
                app_state.metal_active_splat_count,
                app_state.splats.len(),
                app_state.metal_lod_mode,
                app_state.metal_lod_order,
                app_state.metal_lod_requested_splat_count,
            ),
            None => return false,
        };

    if let Err(err) = render_result {
        record_gpu_error(app_state, &err);
        if err.should_disable_gpu() {
            app_state.backend = Backend::Cpu;
            app_state.metal_backend = None;
            app_state.gpu_fallback_active = true;
            eprintln!("Metal disabled for remainder of session: {err}");
        }
        return false;
    }

    let downsample_result: Result<(), crate::render::metal::MetalRenderError> =
        match app_state.metal_backend.as_mut() {
            Some(mb) => mb.downsample_halfblock_cells(
                width,
                height,
                width / app_state.supersample_factor as usize,
                height / (2 * app_state.supersample_factor as usize),
                app_state.supersample_factor as usize,
            ),
            None => return false,
        };

    if let Err(err) = downsample_result {
        record_gpu_error(app_state, &err);
        if err.should_disable_gpu() {
            app_state.backend = Backend::Cpu;
            app_state.metal_backend = None;
            app_state.gpu_fallback_active = true;
            eprintln!("Metal disabled for remainder of session: {err}");
            return false;
        }
        eprintln!("Metal halfblock downsample failed; falling back to CPU downsample: {err}");
    }

    if let Some(mb) = app_state.metal_backend.as_ref() {
        let telemetry = mb.probe_telemetry();
        app_state.visible_splat_count = telemetry.valid_count as usize;
    }
    true
}

#[cfg(feature = "metal")]
fn record_gpu_error(app_state: &mut AppState, err: &dyn std::error::Error) {
    app_state.last_gpu_error = Some(err.to_string());
}
