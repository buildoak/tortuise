use crate::render::HalfblockCell;

// --- Downsample ---

pub fn downsample_to_terminal_into(
    fb: &[[u8; 3]],
    ss_width: usize,
    ss_height: usize,
    term_cols: usize,
    term_rows: usize,
    ss: usize,
    out: &mut Vec<HalfblockCell>,
) {
    out.clear();
    out.resize(term_cols * term_rows, ([0u8; 3], [0u8; 3]));

    for term_row in 0..term_rows {
        for term_col in 0..term_cols {
            // top pixel block: x in [col*ss .. (col+1)*ss], y in [row*2*ss .. (row*2+1)*ss]
            let x0 = term_col * ss;
            let x1 = ((term_col + 1) * ss).min(ss_width);
            let top_y0 = term_row * 2 * ss;
            let top_y1 = (term_row * 2 * ss + ss).min(ss_height);
            let bot_y0 = (term_row * 2 * ss + ss).min(ss_height);
            let bot_y1 = ((term_row + 1) * 2 * ss).min(ss_height);

            let mut tr = 0u32;
            let mut tg = 0u32;
            let mut tb = 0u32;
            let mut t_count = 0u32;

            for y in top_y0..top_y1 {
                for x in x0..x1 {
                    let p = fb[y * ss_width + x];
                    tr += p[0] as u32;
                    tg += p[1] as u32;
                    tb += p[2] as u32;
                    t_count += 1;
                }
            }

            let mut br = 0u32;
            let mut bg_g = 0u32;
            let mut bb = 0u32;
            let mut b_count = 0u32;

            for y in bot_y0..bot_y1 {
                for x in x0..x1 {
                    let p = fb[y * ss_width + x];
                    br += p[0] as u32;
                    bg_g += p[1] as u32;
                    bb += p[2] as u32;
                    b_count += 1;
                }
            }

            let bg_color = if t_count > 0 {
                [
                    (tr / t_count) as u8,
                    (tg / t_count) as u8,
                    (tb / t_count) as u8,
                ]
            } else {
                [0, 0, 0]
            };

            let fg_color = if b_count > 0 {
                [
                    (br / b_count) as u8,
                    (bg_g / b_count) as u8,
                    (bb / b_count) as u8,
                ]
            } else {
                [0, 0, 0]
            };

            out[term_row * term_cols + term_col] = (bg_color, fg_color);
        }
    }
}

#[cfg(feature = "metal")]
pub fn downsample_packed_to_terminal_into(
    fb: &[u32],
    ss_width: usize,
    ss_height: usize,
    term_cols: usize,
    term_rows: usize,
    ss: usize,
    out: &mut Vec<HalfblockCell>,
) {
    out.clear();
    out.resize(term_cols * term_rows, ([0u8; 3], [0u8; 3]));

    for term_row in 0..term_rows {
        for term_col in 0..term_cols {
            let x0 = term_col * ss;
            let x1 = ((term_col + 1) * ss).min(ss_width);
            let top_y0 = term_row * 2 * ss;
            let top_y1 = (term_row * 2 * ss + ss).min(ss_height);
            let bot_y0 = (term_row * 2 * ss + ss).min(ss_height);
            let bot_y1 = ((term_row + 1) * 2 * ss).min(ss_height);

            let mut tr = 0u32;
            let mut tg = 0u32;
            let mut tb = 0u32;
            let mut t_count = 0u32;

            for y in top_y0..top_y1 {
                for x in x0..x1 {
                    let p = fb[y * ss_width + x];
                    tr += p & 0xFF;
                    tg += (p >> 8) & 0xFF;
                    tb += (p >> 16) & 0xFF;
                    t_count += 1;
                }
            }

            let mut br = 0u32;
            let mut bg = 0u32;
            let mut bb = 0u32;
            let mut b_count = 0u32;

            for y in bot_y0..bot_y1 {
                for x in x0..x1 {
                    let p = fb[y * ss_width + x];
                    br += p & 0xFF;
                    bg += (p >> 8) & 0xFF;
                    bb += (p >> 16) & 0xFF;
                    b_count += 1;
                }
            }

            let bg_color = if t_count > 0 {
                [
                    (tr / t_count) as u8,
                    (tg / t_count) as u8,
                    (tb / t_count) as u8,
                ]
            } else {
                [0, 0, 0]
            };

            let fg_color = if b_count > 0 {
                [
                    (br / b_count) as u8,
                    (bg / b_count) as u8,
                    (bb / b_count) as u8,
                ]
            } else {
                [0, 0, 0]
            };

            out[term_row * term_cols + term_col] = (bg_color, fg_color);
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::downsample_packed_to_terminal_into;

    #[test]
    fn packed_downsample_matches_probe_low_byte_red_layout() {
        let fb = [
            0x000000ff, 0x0000ff00, 0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000, 0x00ff0000,
            0x00000000,
        ];
        let mut out = Vec::new();

        downsample_packed_to_terminal_into(&fb, 2, 4, 1, 1, 2, &mut out);

        assert_eq!(out, vec![([127, 127, 0], [0, 0, 127])]);
    }
}
