use crossterm::{
    cursor, queue,
    style::{Print, ResetColor, SetBackgroundColor, SetForegroundColor},
};
use std::io::{self, Write};

use super::{make_color, AppState, HALF_BLOCK};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreviewRect {
    x: u16,
    y: u16,
    cols: u16,
    rows: u16,
}

fn preview_rect(cols: u16, rows: u16, scale: f32) -> Option<PreviewRect> {
    if cols < 30 || rows < 12 {
        return None;
    }

    let scale = scale.clamp(0.05, 0.50);
    let width = ((cols as f32 * scale).round() as u16).clamp(12, cols.saturating_sub(2));
    let height = ((rows as f32 * scale).round() as u16).clamp(6, rows.saturating_sub(3));
    if width >= cols || height + 2 >= rows {
        return None;
    }

    Some(PreviewRect {
        x: cols.saturating_sub(width + 1),
        y: rows.saturating_sub(height + 2),
        cols: width,
        rows: height,
    })
}

fn sample_rgb(frame: &crate::hand::types::CameraPreviewFrame, x: usize, y: usize) -> [u8; 3] {
    let x = x.min(frame.width.saturating_sub(1));
    let y = y.min(frame.height.saturating_sub(1));
    let idx = (y * frame.width + x) * 3;
    [
        *frame.rgb.get(idx).unwrap_or(&0),
        *frame.rgb.get(idx + 1).unwrap_or(&0),
        *frame.rgb.get(idx + 2).unwrap_or(&0),
    ]
}

fn marker_rgb(
    app_state: &AppState,
    frame_width: usize,
    frame_height: usize,
    x: usize,
    y: usize,
) -> Option<[u8; 3]> {
    let radius = (frame_width.max(frame_height) as f32 / 24.0).max(2.0);
    let radius_sq = radius * radius;
    for hand in &app_state.hand_control.latest_hands {
        if hand.confidence < 0.25 {
            continue;
        }
        let hx = hand.x.clamp(0.0, 1.0) * frame_width.saturating_sub(1) as f32;
        let hy = (1.0 - hand.y.clamp(0.0, 1.0)) * frame_height.saturating_sub(1) as f32;
        let dx = x as f32 - hx;
        let dy = y as f32 - hy;
        if dx * dx + dy * dy <= radius_sq {
            return Some(if hand.pinch >= 0.72 {
                [80, 255, 140]
            } else {
                [70, 190, 255]
            });
        }
    }
    None
}

fn preview_pixel_rgb(
    app_state: &AppState,
    frame: &crate::hand::types::CameraPreviewFrame,
    x: usize,
    y: usize,
) -> [u8; 3] {
    marker_rgb(app_state, frame.width, frame.height, x, y)
        .unwrap_or_else(|| sample_rgb(frame, x, y))
}

pub fn draw_camera_preview(
    app_state: &AppState,
    cols: u16,
    rows: u16,
    stdout: &mut impl Write,
) -> io::Result<()> {
    if !app_state.hand_control.camera_preview_enabled || !app_state.hand_control.enabled {
        return Ok(());
    }
    let Some(frame) = app_state.hand_control.latest_preview.as_ref() else {
        if let Some(rect) = preview_rect(cols, rows, app_state.hand_control.camera_preview_scale) {
            draw_placeholder(app_state, rect, stdout)?;
        }
        return Ok(());
    };
    if frame.width == 0 || frame.height == 0 || frame.rgb.len() < frame.width * frame.height * 3 {
        return Ok(());
    }
    let Some(rect) = preview_rect(cols, rows, app_state.hand_control.camera_preview_scale) else {
        return Ok(());
    };

    let tc = app_state.use_truecolor;
    let draw_h = rect.rows as usize * 2;
    for row in 0..rect.rows as usize {
        queue!(stdout, cursor::MoveTo(rect.x, rect.y + row as u16))?;
        for col in 0..rect.cols as usize {
            let sx = col * frame.width / rect.cols as usize;
            let sy_top = (row * 2) * frame.height / draw_h;
            let sy_bottom = ((row * 2 + 1) * frame.height / draw_h).min(frame.height - 1);
            let top = preview_pixel_rgb(app_state, frame, sx, sy_top);
            let bottom = preview_pixel_rgb(app_state, frame, sx, sy_bottom);
            queue!(
                stdout,
                SetBackgroundColor(make_color(top[0], top[1], top[2], tc)),
                SetForegroundColor(make_color(bottom[0], bottom[1], bottom[2], tc)),
                crossterm::style::Print(HALF_BLOCK)
            )?;
        }
    }
    queue!(stdout, ResetColor)?;
    Ok(())
}

fn draw_placeholder(
    app_state: &AppState,
    rect: PreviewRect,
    stdout: &mut impl Write,
) -> io::Result<()> {
    let tc = app_state.use_truecolor;
    for row in 0..rect.rows {
        queue!(stdout, cursor::MoveTo(rect.x, rect.y + row))?;
        for _ in 0..rect.cols {
            queue!(
                stdout,
                SetBackgroundColor(make_color(18, 22, 28, tc)),
                SetForegroundColor(make_color(18, 22, 28, tc)),
                Print(" ")
            )?;
        }
    }

    let label = match app_state.hand_control.status {
        crate::hand::types::HandStatus::Error(code) => code,
        _ => "camera...",
    };
    let max_len = rect.cols.saturating_sub(2) as usize;
    let text = &label[..label.len().min(max_len)];
    queue!(
        stdout,
        cursor::MoveTo(rect.x + 1, rect.y + rect.rows / 2),
        SetBackgroundColor(make_color(18, 22, 28, tc)),
        SetForegroundColor(make_color(210, 220, 235, tc)),
        Print(text),
        ResetColor
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::preview_rect;

    #[test]
    fn preview_rect_stays_bottom_right_and_inside_terminal() {
        let rect = preview_rect(120, 40, 0.15).expect("rect");
        assert_eq!(rect.cols, 18);
        assert_eq!(rect.rows, 6);
        assert!(rect.x + rect.cols < 120);
        assert!(rect.y + rect.rows < 39);
    }

    #[test]
    fn preview_rect_hides_on_tiny_terminals() {
        assert!(preview_rect(20, 10, 0.15).is_none());
    }
}
