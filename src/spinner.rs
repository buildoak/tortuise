//! Matrix-style terminal progress animation for long-running operations.
//!
//! Displays a multi-line Matrix digital rain effect with an embedded progress bar
//! while SHARP neural network inference runs in the background.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

// ── Constants ──────────────────────────────────────────────────────────────────

/// Animation frame interval (~30fps).
const TICK: Duration = Duration::from_millis(33);

/// Total height of the animation region in terminal lines.
const ANIM_HEIGHT: usize = 10;

/// Number of rain rows above the progress bar.
const RAIN_ROWS_ABOVE: usize = 4;

/// Number of rain rows below the status text.
const RAIN_ROWS_BELOW: usize = 4;

/// Progress bar width as a fraction of terminal width.
const BAR_WIDTH_FRAC: f64 = 0.65;

/// Minimum progress bar width in characters.
const BAR_MIN_WIDTH: usize = 30;

/// Maximum progress bar width in characters.
const BAR_MAX_WIDTH: usize = 80;

// ── ANSI escape helpers ────────────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const WHITE: &str = "\x1b[97m";
const GREEN_BRIGHT: &str = "\x1b[92m";
const GREEN: &str = "\x1b[32m";
const GREEN_DIM: &str = "\x1b[2;32m";
const HIDE_CURSOR: &str = "\x1b[?25l";
const SHOW_CURSOR: &str = "\x1b[?25h";

// ── Character set ──────────────────────────────────────────────────────────────

/// Half-width katakana + digits + symbols for the Matrix rain effect.
const RAIN_CHARS: &[char] = &[
    // Half-width katakana
    'ｱ', 'ｲ', 'ｳ', 'ｴ', 'ｵ', 'ｶ', 'ｷ', 'ｸ', 'ｹ', 'ｺ',
    'ｻ', 'ｼ', 'ｽ', 'ｾ', 'ｿ', 'ﾀ', 'ﾁ', 'ﾂ', 'ﾃ', 'ﾄ',
    'ﾅ', 'ﾆ', 'ﾇ', 'ﾈ', 'ﾉ', 'ﾊ', 'ﾋ', 'ﾌ', 'ﾍ', 'ﾎ',
    'ﾏ', 'ﾐ', 'ﾑ', 'ﾒ', 'ﾓ', 'ﾔ', 'ﾕ', 'ﾖ', 'ﾗ', 'ﾘ',
    'ﾙ', 'ﾚ', 'ﾛ', 'ﾜ', 'ﾝ',
    // Digits
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    // Symbols
    ':', '.', '"', '=', '*', '+', '-', '<', '>', '|',
];

// ── Rain column state ──────────────────────────────────────────────────────────

/// Tracks a single column of the Matrix rain.
struct RainColumn {
    /// Current head position (row index, can be negative = not yet on screen).
    head: i32,
    /// Speed: how many rows to advance per tick.
    speed: i32,
    /// Length of the bright trail behind the head.
    trail_len: i32,
    /// Tick counter for speed gating.
    tick: u32,
    /// Ticks between advances (lower = faster).
    tick_rate: u32,
    /// Characters in each row for this column (indexed by row).
    chars: Vec<char>,
}

impl RainColumn {
    fn new(rng: &mut impl Rng, total_rows: usize) -> Self {
        let trail_len = rng.random_range(3..=8);
        let tick_rate = rng.random_range(1..=3);
        let chars: Vec<char> = (0..total_rows)
            .map(|_| RAIN_CHARS[rng.random_range(0..RAIN_CHARS.len())])
            .collect();
        Self {
            head: -rng.random_range(0..12),
            speed: 1,
            trail_len,
            tick: 0,
            tick_rate,
            chars,
        }
    }

    /// Reset the column to start falling again from above.
    fn respawn(&mut self, rng: &mut impl Rng, total_rows: usize) {
        self.head = -rng.random_range(2..10);
        self.trail_len = rng.random_range(3..=8);
        self.tick_rate = rng.random_range(1..=3);
        self.tick = 0;
        // Randomize characters again.
        for ch in &mut self.chars {
            *ch = RAIN_CHARS[rng.random_range(0..RAIN_CHARS.len())];
        }
        // Occasionally mutate just a few for variety.
        let mutations = rng.random_range(1..=(total_rows / 2).max(1));
        for _ in 0..mutations {
            let idx = rng.random_range(0..total_rows);
            self.chars[idx] = RAIN_CHARS[rng.random_range(0..RAIN_CHARS.len())];
        }
    }

    /// Advance the rain drop by one logical step.
    fn advance(&mut self, rng: &mut impl Rng, total_rows: usize) {
        self.tick += 1;
        if self.tick % self.tick_rate != 0 {
            return;
        }
        self.head += self.speed;

        // If the entire trail is off screen, respawn.
        if self.head - self.trail_len > total_rows as i32 + 2 {
            self.respawn(rng, total_rows);
        }

        // Occasionally mutate a character near the head for the "shifting" effect.
        if rng.random_range(0..4) == 0 {
            let mutate_row = self.head.saturating_sub(rng.random_range(0..3));
            if mutate_row >= 0 && (mutate_row as usize) < total_rows {
                self.chars[mutate_row as usize] =
                    RAIN_CHARS[rng.random_range(0..RAIN_CHARS.len())];
            }
        }
    }

    /// Return the ANSI-colored character for a given row, or None if this column
    /// has nothing visible at that row.
    fn render_at(&self, row: usize) -> Option<(&str, char)> {
        let row_i = row as i32;
        let distance_from_head = self.head - row_i;

        if distance_from_head < 0 {
            // Row is below the head — nothing here yet.
            return None;
        }

        let ch = self.chars.get(row)?;

        if distance_from_head == 0 {
            // Leading edge: bright white.
            Some((WHITE, *ch))
        } else if distance_from_head <= 1 {
            // Just behind head: bright green.
            Some((GREEN_BRIGHT, *ch))
        } else if distance_from_head <= self.trail_len {
            // Trail: normal green.
            Some((GREEN, *ch))
        } else if distance_from_head <= self.trail_len + 3 {
            // Fading tail: dim green.
            Some((GREEN_DIM, *ch))
        } else {
            None
        }
    }
}

// ── Pseudo-progress curve ──────────────────────────────────────────────────────

/// Maps elapsed seconds to a pseudo-progress value in [0.0, 0.85].
///
/// - 0-3s:  fast ramp to ~30%
/// - 3-18s: steady climb to ~80%
/// - 18s+:  asymptotically approaches 85%
///
/// When `finish()` is called, progress snaps to 100%.
fn pseudo_progress(elapsed_secs: f64) -> f64 {
    if elapsed_secs <= 3.0 {
        // Fast start: ease-out to 0.30.
        0.30 * (1.0 - (-elapsed_secs * 1.2).exp())
    } else if elapsed_secs <= 18.0 {
        // Middle: linear-ish from 0.30 to 0.80.
        let t = (elapsed_secs - 3.0) / 15.0;
        0.30 + 0.50 * ease_in_out(t)
    } else {
        // Tail: slowly approach 0.85.
        let overshoot = elapsed_secs - 18.0;
        0.80 + 0.05 * (1.0 - (-overshoot * 0.15).exp())
    }
}

/// Cubic ease-in-out.
fn ease_in_out(t: f64) -> f64 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let f = 2.0 * t - 2.0;
        0.5 * f * f * f + 1.0
    }
}

// ── Progress bar rendering ─────────────────────────────────────────────────────

/// Render the progress bar line into the buffer.
fn render_progress_bar(buf: &mut String, progress: f64, bar_width: usize, term_width: usize) {
    let filled = ((progress * bar_width as f64).round() as usize).min(bar_width);
    let empty = bar_width.saturating_sub(filled);

    // Center the bar.
    let total_bar_display = bar_width + 2; // brackets
    let pad_left = term_width.saturating_sub(total_bar_display) / 2;

    // Left padding.
    for _ in 0..pad_left {
        buf.push(' ');
    }

    // Opening bracket.
    buf.push_str(GREEN_DIM);
    buf.push('[');

    // Filled portion.
    if filled > 0 {
        // Main filled section.
        buf.push_str(GREEN_BRIGHT);
        buf.push_str(BOLD);
        for _ in 0..filled.saturating_sub(1) {
            buf.push('━');
        }
        // Leading edge: white bright.
        if progress < 1.0 {
            buf.push_str(RESET);
            buf.push_str(WHITE);
            buf.push_str(BOLD);
            buf.push('▓');
        } else {
            buf.push('━');
        }
    }

    // Empty portion.
    if empty > 0 {
        buf.push_str(RESET);
        buf.push_str(GREEN_DIM);
        for _ in 0..empty {
            buf.push('░');
        }
    }

    // Closing bracket.
    buf.push_str(RESET);
    buf.push_str(GREEN_DIM);
    buf.push(']');
    buf.push_str(RESET);

    // Percentage.
    let pct = (progress * 100.0).min(100.0) as u32;
    buf.push(' ');
    buf.push_str(GREEN);
    let pct_str = format!("{pct:>3}%");
    buf.push_str(&pct_str);
    buf.push_str(RESET);
}

/// Render the status message line.
fn render_status_line(buf: &mut String, message: &str, term_width: usize, tick: u32) {
    let display_msg = message;
    // Compute a pulsing dot animation for the ellipsis.
    let dots = match (tick / 8) % 4 {
        0 => "   ",
        1 => ".  ",
        2 => ".. ",
        _ => "...",
    };

    // Strip trailing "..." from message if present — we animate it ourselves.
    let base_msg = display_msg.trim_end_matches("...").trim_end_matches("..");

    let full_msg = format!("{base_msg}{dots}");
    let msg_len = full_msg.len();
    let pad_left = term_width.saturating_sub(msg_len) / 2;

    for _ in 0..pad_left {
        buf.push(' ');
    }

    buf.push_str(GREEN_BRIGHT);
    buf.push_str(&full_msg);
    buf.push_str(RESET);
}

// ── Rain row rendering ─────────────────────────────────────────────────────────

/// Render a single row of the Matrix rain into the buffer.
fn render_rain_row(
    buf: &mut String,
    columns: &[RainColumn],
    row: usize,
    term_width: usize,
    col_spacing: usize,
) {
    // We place rain characters at evenly spaced columns across the width.
    let mut pos = 0;
    for col in columns {
        if pos >= term_width {
            break;
        }
        // Render this column's character at the given row.
        if let Some((color, ch)) = col.render_at(row) {
            buf.push_str(color);
            buf.push(ch);
            buf.push_str(RESET);
        } else {
            buf.push(' ');
        }
        pos += 1;

        // Add spacing between columns.
        let spaces = col_spacing.saturating_sub(1);
        for _ in 0..spaces {
            if pos < term_width {
                buf.push(' ');
                pos += 1;
            }
        }
    }
    // Fill remaining width with spaces.
    while pos < term_width {
        buf.push(' ');
        pos += 1;
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// A Matrix-style progress animation that runs in a background thread.
pub struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Spinner {
    /// Starts the Matrix rain animation with an embedded progress bar.
    ///
    /// The animation runs on stderr in a background thread until [`Spinner::finish`]
    /// is called or the `Spinner` is dropped.
    pub fn start(message: &str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let msg = message.to_string();

        let handle = thread::spawn(move || {
            run_animation(&msg, &stop_clone);
        });

        Self {
            stop,
            handle: Some(handle),
        }
    }

    /// Stops the animation and prints a single completion line with a green checkmark.
    pub fn finish(mut self, message: &str) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        eprintln!("\x1b[32m✓\x1b[0m {message}");
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ── Animation loop ─────────────────────────────────────────────────────────────

fn run_animation(message: &str, stop: &AtomicBool) {
    let mut stderr = io::stderr();
    let mut rng = rand::rng();

    // Terminal dimensions.
    let term_width = crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(80)
        .max(40);

    // Rain column spacing and count.
    let col_spacing: usize = 3;
    let num_cols = (term_width / col_spacing).max(4);

    // Initialize rain columns — separate sets for above and below the bar.
    let mut columns_above: Vec<RainColumn> = (0..num_cols)
        .map(|_| RainColumn::new(&mut rng, RAIN_ROWS_ABOVE))
        .collect();
    let mut columns_below: Vec<RainColumn> = (0..num_cols)
        .map(|_| RainColumn::new(&mut rng, RAIN_ROWS_BELOW))
        .collect();

    // Progress bar width.
    let bar_width = ((term_width as f64 * BAR_WIDTH_FRAC) as usize)
        .clamp(BAR_MIN_WIDTH, BAR_MAX_WIDTH)
        .min(term_width.saturating_sub(8));

    let start_time = Instant::now();
    let mut tick: u32 = 0;
    let mut buf = String::with_capacity(4096);

    // Hide cursor and reserve space.
    let _ = write!(stderr, "{HIDE_CURSOR}");
    // Print blank lines to reserve space, then move back up.
    for _ in 0..ANIM_HEIGHT {
        let _ = writeln!(stderr);
    }
    let _ = write!(stderr, "\x1b[{}A", ANIM_HEIGHT);
    let _ = stderr.flush();

    while !stop.load(Ordering::Relaxed) {
        let elapsed = start_time.elapsed().as_secs_f64();
        let progress = pseudo_progress(elapsed);

        buf.clear();

        // Move cursor to top of our animation region.
        // (We stay in the same spot each frame.)

        // ── Rain rows above ──
        for row in 0..RAIN_ROWS_ABOVE {
            buf.push_str("\x1b[2K"); // Clear line.
            render_rain_row(&mut buf, &columns_above, row, term_width, col_spacing);
            buf.push_str("\r\n");
        }

        // ── Progress bar ──
        buf.push_str("\x1b[2K"); // Clear line.
        render_progress_bar(&mut buf, progress, bar_width, term_width);
        buf.push_str("\r\n");

        // ── Status message ──
        buf.push_str("\x1b[2K"); // Clear line.
        render_status_line(&mut buf, message, term_width, tick);
        buf.push_str("\r\n");

        // ── Rain rows below ──
        for row in 0..RAIN_ROWS_BELOW {
            buf.push_str("\x1b[2K"); // Clear line.
            render_rain_row(&mut buf, &columns_below, row, term_width, col_spacing);
            if row < RAIN_ROWS_BELOW - 1 {
                buf.push_str("\r\n");
            }
        }

        // Move cursor back up to top of animation region.
        // We printed ANIM_HEIGHT lines (4 above + 1 bar + 1 status + 4 below = 10).
        buf.push_str(&format!("\r\x1b[{}A", ANIM_HEIGHT - 1));

        let _ = write!(stderr, "{buf}");
        let _ = stderr.flush();

        // Advance rain state.
        for col in &mut columns_above {
            col.advance(&mut rng, RAIN_ROWS_ABOVE);
        }
        for col in &mut columns_below {
            col.advance(&mut rng, RAIN_ROWS_BELOW);
        }

        tick += 1;
        thread::sleep(TICK);
    }

    // ── Clean exit ──
    // Clear all animation lines.
    let mut cleanup = String::with_capacity(256);
    for i in 0..ANIM_HEIGHT {
        cleanup.push_str("\x1b[2K"); // Clear line.
        if i < ANIM_HEIGHT - 1 {
            cleanup.push_str("\r\n");
        }
    }
    // Move back to the top.
    cleanup.push_str(&format!("\r\x1b[{}A", ANIM_HEIGHT - 1));
    // Show cursor.
    cleanup.push_str(SHOW_CURSOR);

    let _ = write!(stderr, "{cleanup}");
    let _ = stderr.flush();
}
