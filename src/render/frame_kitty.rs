use std::io::{self, Write};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use crossterm::{cursor, queue};

use super::{AppState, Backend};

const KITTY_DIRECT_CHUNK_SIZE: usize = 4096;

fn kitty_base64_len(payload_bytes: usize) -> usize {
    payload_bytes.div_ceil(3) * 4
}

fn packed_framebuffer_to_rgba(packed: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(packed.len().saturating_mul(4));
    for pixel in packed {
        out.push((pixel & 0xFF) as u8);
        out.push(((pixel >> 8) & 0xFF) as u8);
        out.push(((pixel >> 16) & 0xFF) as u8);
        out.push(((pixel >> 24) & 0xFF) as u8);
    }
}

fn write_kitty_rgba_direct(
    stdout: &mut impl Write,
    image_id: u32,
    width: usize,
    height: usize,
    term_cols: usize,
    term_rows: usize,
    rgba: &[u8],
) -> io::Result<(usize, usize)> {
    if rgba.is_empty() {
        return Ok((0, 0));
    }

    let encoded = BASE64_STANDARD.encode(rgba);
    debug_assert_eq!(encoded.len(), kitty_base64_len(rgba.len()));
    let mut chunks = 0usize;
    let mut offset = 0usize;
    while offset < encoded.len() {
        let end = (offset + KITTY_DIRECT_CHUNK_SIZE).min(encoded.len());
        let chunk = &encoded[offset..end];
        let more = if end < encoded.len() { 1 } else { 0 };
        if offset == 0 {
            write!(
                stdout,
                "\x1b_Ga=T,f=32,t=d,s={width},v={height},c={term_cols},r={term_rows},i={image_id},q=2,C=1,m={more};{chunk}\x1b\\"
            )?;
        } else {
            write!(stdout, "\x1b_Gq=2,m={more};{chunk}\x1b\\")?;
        }
        chunks += 1;
        offset = end;
    }
    Ok((encoded.len(), chunks))
}

fn render_metal_framebuffer(
    app_state: &mut AppState,
    width: usize,
    height: usize,
) -> Result<(), crate::render::metal::MetalRenderError> {
    match app_state.metal_backend.as_mut() {
        Some(mb) => mb.render(&app_state.camera, width, height, app_state.splats.len()),
        None => Err(crate::render::metal::MetalRenderError::GpuDisabled),
    }
}

fn record_kitty_gpu_error(app_state: &mut AppState, err: &crate::render::metal::MetalRenderError) {
    app_state.last_gpu_error = Some(err.to_string());
    if err.should_disable_gpu() {
        app_state.backend = Backend::Cpu;
        app_state.metal_backend = None;
        app_state.gpu_fallback_active = true;
    }
}

pub fn render_kitty_frame(
    app_state: &mut AppState,
    term_cols: usize,
    term_rows: usize,
    stdout: &mut impl Write,
) -> io::Result<()> {
    if app_state.backend != Backend::Metal || app_state.metal_backend.is_none() {
        return super::frame_halfblock::render_halfblock_frame(
            app_state, term_cols, term_rows, stdout,
        );
    }

    let ss = app_state.supersample_factor as usize;
    let width = term_cols.saturating_mul(ss).max(1);
    let height = term_rows.saturating_mul(2).saturating_mul(ss).max(1);
    super::pipeline::resize_render_state(&mut app_state.render_state, width, height);

    if let Err(err) = render_metal_framebuffer(app_state, width, height) {
        record_kitty_gpu_error(app_state, &err);
        return super::frame_halfblock::render_halfblock_frame(
            app_state, term_cols, term_rows, stdout,
        );
    }

    let mut rgba = Vec::with_capacity(width.saturating_mul(height).saturating_mul(4));
    if let Some(mb) = app_state.metal_backend.as_ref() {
        packed_framebuffer_to_rgba(mb.framebuffer_slice(), &mut rgba);
    }

    queue!(stdout, cursor::MoveTo(0, 0))?;
    let (base64_bytes, chunks) = write_kitty_rgba_direct(
        stdout,
        app_state.kitty_image_id,
        width,
        height,
        term_cols,
        term_rows,
        &rgba,
    )?;
    app_state.kitty_payload_bytes = rgba.len();
    app_state.kitty_base64_bytes = base64_bytes;
    app_state.kitty_chunks = chunks;
    app_state.visible_splat_count = app_state.splats.len();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_framebuffer_to_rgba_uses_low_byte_red_layout() {
        let mut out = Vec::new();
        packed_framebuffer_to_rgba(&[0x44332211], &mut out);
        assert_eq!(out, vec![0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn kitty_direct_writer_chunks_payload() {
        let rgba = vec![7u8; 4096];
        let mut out = Vec::new();

        let (base64_bytes, chunks) =
            write_kitty_rgba_direct(&mut out, 1, 32, 32, 16, 16, &rgba).unwrap();

        assert_eq!(base64_bytes, kitty_base64_len(rgba.len()));
        assert_eq!(chunks, 2);
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("\x1b_Ga=T,f=32,t=d,s=32,v=32,c=16,r=16,i=1,q=2,C=1,m=1;"));
        assert!(text.contains("\x1b_Gq=2,m=0;"));
    }
}
