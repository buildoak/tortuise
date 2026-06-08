use crossterm::{
    cursor, queue,
    style::{Print, SetBackgroundColor, SetForegroundColor},
};
use std::fmt::Write as _;
use std::io::{self, Write};

use super::{make_color, AppState, RenderMode};

fn truncate_and_pad_in_place(text: &mut String, width: usize) {
    if width == 0 {
        text.clear();
        return;
    }

    let mut seen_chars = 0usize;
    let mut truncate_byte = None;
    for (idx, _) in text.char_indices() {
        if seen_chars == width {
            truncate_byte = Some(idx);
            break;
        }
        seen_chars += 1;
    }

    if let Some(idx) = truncate_byte {
        text.truncate(idx);
    } else {
        for _ in seen_chars..width {
            text.push(' ');
        }
    }
}

pub fn draw_hud(
    app_state: &mut AppState,
    cols: u16,
    rows: u16,
    ss: usize,
    stdout: &mut impl Write,
) -> io::Result<()> {
    let width = cols as usize;
    let term_cols = cols as usize;
    let term_rows = rows as usize;
    let render_path = app_state.effective_render_path;
    let hud = &mut app_state.hud_string_buf;
    hud.clear();
    #[cfg(feature = "metal")]
    let splat_label = if app_state.effective_render_path.starts_with("metal_") {
        format!(
            "{}/{}/{}",
            app_state.visible_splat_count,
            app_state.metal_active_splat_count,
            app_state.splats.len()
        )
    } else {
        format!(
            "{}/{}",
            app_state.visible_splat_count,
            app_state.splats.len()
        )
    };
    #[cfg(not(feature = "metal"))]
    let splat_label = format!(
        "{}/{}",
        app_state.visible_splat_count,
        app_state.splats.len()
    );
    write!(
        hud,
        "FPS:{:>5.1}  Splats:{}  Pos:({:>6.2},{:>6.2},{:>6.2})  Speed:{:.2}  Cam:{}  Mode:{}  Render:{}  ",
        app_state.fps,
        splat_label,
        app_state.camera.position.x,
        app_state.camera.position.y,
        app_state.camera.position.z,
        app_state.move_speed,
        app_state.camera_mode.name(),
        app_state.render_mode.name(),
        render_path
    )
    .map_err(|_| io::Error::other("failed to format HUD"))?;

    let pixel_mode = app_state.render_mode == RenderMode::Halfblock || {
        #[cfg(feature = "metal")]
        {
            app_state.render_mode == RenderMode::Kitty
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    };

    if pixel_mode {
        write!(
            hud,
            "Canvas:{}x{} Detail:{}x",
            app_state.last_render_width.max(term_cols * ss),
            app_state.last_render_height.max(term_rows * 2 * ss),
            app_state.supersample_factor,
        )
        .map_err(|_| io::Error::other("failed to format HUD"))?;
    } else {
        hud.push_str("Canvas:N/A");
    }

    #[cfg(feature = "metal")]
    if app_state.render_mode == RenderMode::Kitty {
        write!(
            hud,
            "  Kitty:{}B/{}B {}ch flush{:.1}ms",
            app_state.kitty_payload_bytes,
            app_state.kitty_base64_bytes,
            app_state.kitty_chunks,
            app_state.last_flush_ms
        )
        .map_err(|_| io::Error::other("failed to format HUD"))?;
    }

    write!(hud, "  Cores:{}", rayon::current_num_threads())
        .map_err(|_| io::Error::other("failed to format HUD"))?;
    #[cfg(feature = "metal")]
    {
        hud.push_str("  GPU:");
        if let Some(err) = app_state.last_gpu_error.as_deref() {
            write!(hud, "ERR:{err}").map_err(|_| io::Error::other("failed to format HUD"))?;
        } else if app_state.gpu_fallback_active {
            hud.push_str("DISABLED");
        } else {
            hud.push_str("OK");
        }
    }
    truncate_and_pad_in_place(hud, width);

    let tc = app_state.use_truecolor;
    queue!(
        stdout,
        cursor::MoveTo(0, 0),
        SetBackgroundColor(make_color(0, 0, 0, tc)),
        SetForegroundColor(make_color(245, 245, 245, tc)),
        Print(hud.as_str())
    )?;

    let controls = match app_state.camera_mode {
        super::CameraMode::Free => {
            "WASD:Move  R/F:Up/Down  Arrows:Look  +/-:Speed  Space:Orbit  M:Mode  Tab:HUD  Z:Reset  Q/Esc:Quit"
        }
        super::CameraMode::Orbit => {
            "Arrows:Elevation/Nudge  +/-:Speed  Space:Free cam  M:Mode  Tab:HUD  Z:Reset  Q/Esc:Quit"
        }
    };
    hud.clear();
    hud.push_str(controls);
    truncate_and_pad_in_place(hud, width);

    queue!(
        stdout,
        cursor::MoveTo(0, rows - 1),
        SetBackgroundColor(make_color(0, 0, 0, tc)),
        SetForegroundColor(make_color(220, 220, 220, tc)),
        Print(hud.as_str())
    )?;

    Ok(())
}
