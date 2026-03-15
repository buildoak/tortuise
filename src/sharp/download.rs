use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::SharpError;

const MODEL_URL: &str =
    "https://huggingface.co/buildoak/sharp-onnx/resolve/main/sharp_fp16.onnx";
const MODEL_DIR: &str = ".tortuise/models";
const MODEL_FILENAME: &str = "sharp_fp16.onnx";

/// Minimum plausible model size (10 MB). Anything smaller is almost
/// certainly a truncated download or placeholder.
const MIN_MODEL_SIZE: u64 = 10 * 1024 * 1024;

/// Approximate total download size for the consent prompt.
const APPROX_TOTAL_SIZE: &str = "~1.3 GB";

// ── ANSI escape helpers ────────────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const GREEN_BRIGHT: &str = "\x1b[92m";
const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";

// ── Public API ─────────────────────────────────────────────────────────────────

/// Ensures the SHARP FP16 ONNX model is present in the local cache directory.
///
/// If the model is not cached, prompts the user for consent before downloading.
/// Downloads show a clean progress bar (not the Matrix rain animation).
/// Returns the path to the model file once ready.
pub(super) fn ensure_model_available() -> Result<PathBuf, SharpError> {
    #[cfg(windows)]
    let home_var = "USERPROFILE";
    #[cfg(not(windows))]
    let home_var = "HOME";

    let home = std::env::var(home_var)
        .map_err(|e| SharpError::Download(format!("failed to resolve {}: {}", home_var, e)))?;

    let model_dir = PathBuf::from(home).join(MODEL_DIR);
    let model_path = model_dir.join(MODEL_FILENAME);

    // Check if model file already exists and is valid.
    if model_path.exists() {
        let model_size = fs::metadata(&model_path)
            .map(|m| m.len())
            .unwrap_or(0);

        if model_size >= MIN_MODEL_SIZE {
            return Ok(model_path);
        }
        eprintln!(
            "{}Cached model is too small ({} bytes), need to re-download.{}",
            DIM, model_size, RESET,
        );
        let _ = fs::remove_file(&model_path);
    }

    // ── Consent prompt ─────────────────────────────────────────────────────────

    prompt_download_consent(&model_dir)?;

    fs::create_dir_all(&model_dir).map_err(|e| {
        SharpError::Download(format!(
            "failed to create model cache directory '{}': {}",
            model_dir.display(),
            e
        ))
    })?;

    // ── Download model ─────────────────────────────────────────────────────────

    let total_bytes = download_file(MODEL_URL, &model_path, "model", MODEL_FILENAME)?;

    // ── Completion message ─────────────────────────────────────────────────────

    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "\n{}{}\u{2713} SHARP model ready ({:.1} GB){}",
        GREEN,
        BOLD,
        total_mb / 1024.0,
        RESET,
    );

    Ok(model_path)
}

// ── Consent prompt ─────────────────────────────────────────────────────────────

fn prompt_download_consent(model_dir: &Path) -> Result<(), SharpError> {
    eprintln!();
    eprintln!(
        "{}{}SHARP model not found locally.{}",
        YELLOW, BOLD, RESET,
    );
    eprintln!(
        "Download from HuggingFace? ({APPROX_TOTAL_SIZE})",
    );
    eprintln!(
        "{}Cache location: {}{}",
        DIM,
        model_dir.display(),
        RESET,
    );
    eprint!("[Y/n]: ");
    io::stderr().flush().map_err(|e| {
        SharpError::Download(format!("failed to flush stderr: {}", e))
    })?;

    let mut input = String::new();
    io::stdin()
        .lock()
        .read_line(&mut input)
        .map_err(|e| SharpError::Download(format!("failed to read stdin: {}", e)))?;

    let answer = input.trim().to_ascii_lowercase();
    if answer.is_empty() || answer == "y" || answer == "yes" {
        eprintln!();
        Ok(())
    } else {
        eprintln!();
        eprintln!(
            "Download declined. You can manually place the model file in:"
        );
        eprintln!("  {}", model_dir.display());
        eprintln!("  Required: {}", MODEL_FILENAME);
        std::process::exit(0);
    }
}

// ── Download with progress bar ─────────────────────────────────────────────────

/// Downloads a file with a real progress bar. Returns total bytes downloaded.
fn download_file(
    url: &str,
    dest: &Path,
    label: &str,
    filename: &str,
) -> Result<u64, SharpError> {
    let mut tmp_name = dest.as_os_str().to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = PathBuf::from(tmp_name);

    let mut response = ureq::get(url).call().map_err(|e| {
        SharpError::Download(format!("failed to request {} from {}: {}", label, url, e))
    })?;

    let content_length = response
        .headers()
        .get("Content-Length")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let mut file = fs::File::create(&tmp_path).map_err(|e| {
        SharpError::Download(format!(
            "failed to create temporary file '{}': {}",
            tmp_path.display(),
            e
        ))
    })?;

    // Print the header line for this file.
    let total_str = match content_length {
        Some(total) => format!("{:.1} MB", total as f64 / (1024.0 * 1024.0)),
        None => "unknown size".to_string(),
    };
    eprintln!(
        "{}Downloading SHARP model ({})...{} [{}]",
        GREEN, label, RESET, total_str,
    );

    let mut reader = response.body_mut().as_reader();
    let mut buffer = [0_u8; 64 * 1024];
    let mut downloaded: u64 = 0;
    let start_time = Instant::now();
    let mut last_render = Instant::now() - Duration::from_secs(1); // force first render

    // Hide cursor during the download so it does not flicker at the line start.
    let _ = write!(io::stderr(), "\x1b[?25l");
    let _ = io::stderr().flush();

    let write_result: Result<(), SharpError> = (|| {
        loop {
            let read = reader.read(&mut buffer).map_err(|e| {
                SharpError::Download(format!(
                    "failed reading download stream from {}: {}",
                    url, e
                ))
            })?;

            if read == 0 {
                break;
            }

            file.write_all(&buffer[..read]).map_err(|e| {
                SharpError::Download(format!(
                    "failed writing to temporary file '{}': {}",
                    tmp_path.display(),
                    e
                ))
            })?;

            downloaded += read as u64;

            // Throttle redraws to at most once per 100 ms to eliminate flicker.
            let now = Instant::now();
            if now.duration_since(last_render) >= Duration::from_millis(100) {
                render_download_progress(filename, downloaded, content_length, &start_time);
                last_render = now;
            }
        }

        // Final progress line (100%) — always render regardless of throttle.
        render_download_progress(filename, downloaded, content_length, &start_time);
        // Move to the next line and restore the cursor.
        let _ = write!(io::stderr(), "\n\x1b[?25h");
        let _ = io::stderr().flush();

        file.flush().map_err(|e| {
            SharpError::Download(format!(
                "failed flushing temporary file '{}': {}",
                tmp_path.display(),
                e
            ))
        })?;

        if let Some(expected) = content_length {
            if downloaded != expected {
                return Err(SharpError::Download(format!(
                    "incomplete download of {}: expected {} bytes, got {}",
                    label, expected, downloaded
                )));
            }
        }

        Ok(())
    })();

    if write_result.is_err() {
        // Restore the cursor even when the download fails.
        let _ = write!(io::stderr(), "\x1b[?25h");
        let _ = io::stderr().flush();
        let _ = fs::remove_file(&tmp_path);
        return write_result.map(|_| 0);
    }

    fs::rename(&tmp_path, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        SharpError::Download(format!(
            "failed to finalize {} at '{}': {}",
            label,
            dest.display(),
            e
        ))
    })?;

    Ok(downloaded)
}

// ── Progress bar rendering ─────────────────────────────────────────────────────

/// Renders a single-line download progress bar using `\r` to overwrite in place.
///
/// Format:
/// ```text
///   [████████████████░░░░░░░░░░░░░░░░░░░░░░░░]  412.3 / 1340.0 MB  (24.1 MB/s)
/// ```
fn render_download_progress(
    filename: &str,
    downloaded: u64,
    total: Option<u64>,
    start_time: &Instant,
) {
    let mut stderr = io::stderr();
    let dl_mb = downloaded as f64 / (1024.0 * 1024.0);
    let elapsed = start_time.elapsed().as_secs_f64();

    // Speed calculation (avoid division by zero).
    let speed_mb_s = if elapsed > 0.1 {
        dl_mb / elapsed
    } else {
        0.0
    };

    match total {
        Some(total_bytes) => {
            let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
            let fraction = if total_bytes > 0 {
                (downloaded as f64 / total_bytes as f64).min(1.0)
            } else {
                0.0
            };

            // Progress bar: 40 chars wide.
            let bar_width: usize = 40;
            let filled = (fraction * bar_width as f64).round() as usize;
            let empty = bar_width.saturating_sub(filled);

            let mut bar = String::with_capacity(bar_width + 20);
            bar.push_str(GREEN_BRIGHT);
            for _ in 0..filled {
                bar.push('\u{2588}'); // Full block.
            }
            bar.push_str(RESET);
            bar.push_str(DIM);
            for _ in 0..empty {
                bar.push('\u{2591}'); // Light shade.
            }
            bar.push_str(RESET);

            let _ = write!(
                stderr,
                "\r\x1b[2K  [{}]  {:.1} / {:.1} MB  ({:.1} MB/s)  {}{}{}",
                bar, dl_mb, total_mb, speed_mb_s, DIM, filename, RESET,
            );
        }
        None => {
            let _ = write!(
                stderr,
                "\r\x1b[2K  {:.1} MB downloaded  ({:.1} MB/s)  {}{}{}",
                dl_mb, speed_mb_s, DIM, filename, RESET,
            );
        }
    }

    let _ = stderr.flush();
}
