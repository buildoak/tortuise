use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use super::SharpError;

const MODEL_URL: &str = "https://huggingface.co/apple/Sharp/resolve/main/sharp.onnx";
const MODEL_DIR: &str = ".tortuise/models";
const MODEL_FILENAME: &str = "sharp.onnx";

/// Minimum plausible model file size (10 MB). Anything smaller is almost
/// certainly a truncated download or placeholder.
const MIN_MODEL_SIZE: u64 = 10 * 1024 * 1024;

/// Ensures the SHARP ONNX model is present in the local cache directory.
pub(super) fn ensure_model_available() -> Result<PathBuf, SharpError> {
    #[cfg(windows)]
    let home_var = "USERPROFILE";
    #[cfg(not(windows))]
    let home_var = "HOME";

    let home = std::env::var(home_var)
        .map_err(|e| SharpError::Download(format!("failed to resolve {}: {}", home_var, e)))?;

    let model_path = PathBuf::from(home).join(MODEL_DIR).join(MODEL_FILENAME);

    if model_path.exists() {
        // Validate cached file is not truncated or empty.
        // ONNX models >2GB use external data: sharp.onnx (structure) + sharp.onnx.data (weights).
        // Count both files when checking the minimum size.
        let meta = fs::metadata(&model_path).map_err(|e| {
            SharpError::Download(format!(
                "failed to stat cached model '{}': {}",
                model_path.display(),
                e
            ))
        })?;
        let mut total_size = meta.len();

        let data_path = {
            let mut p = model_path.as_os_str().to_os_string();
            p.push(".data");
            PathBuf::from(p)
        };
        if let Ok(data_meta) = fs::metadata(&data_path) {
            total_size += data_meta.len();
        }

        if total_size >= MIN_MODEL_SIZE {
            return Ok(model_path);
        }
        eprintln!(
            "Cached model is too small ({} bytes), re-downloading...",
            total_size
        );
        let _ = fs::remove_file(&model_path);
    }

    download_model(&model_path)?;
    Ok(model_path)
}

fn download_model(dest: &Path) -> Result<(), SharpError> {
    let parent = dest.parent().ok_or_else(|| {
        SharpError::Download(format!(
            "failed to resolve destination parent for '{}'",
            dest.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|e| {
        SharpError::Download(format!(
            "failed to create model cache directory '{}': {}",
            parent.display(),
            e
        ))
    })?;

    let mut response = ureq::get(MODEL_URL).call().map_err(|e| {
        SharpError::Download(format!("failed to request model from {}: {}", MODEL_URL, e))
    })?;

    let content_length = response
        .headers()
        .get("Content-Length")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let mut tmp_name = dest.as_os_str().to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = PathBuf::from(tmp_name);

    let mut file = fs::File::create(&tmp_path).map_err(|e| {
        SharpError::Download(format!(
            "failed to create temporary model file '{}': {}",
            tmp_path.display(),
            e
        ))
    })?;

    let mut reader = response.body_mut().as_reader();
    let mut buffer = [0_u8; 8 * 1024];
    let mut downloaded: u64 = 0;

    let write_result: Result<(), SharpError> = (|| {
        loop {
            let read = reader.read(&mut buffer).map_err(|e| {
                SharpError::Download(format!(
                    "failed reading model download stream from {}: {}",
                    MODEL_URL, e
                ))
            })?;

            if read == 0 {
                break;
            }

            file.write_all(&buffer[..read]).map_err(|e| {
                SharpError::Download(format!(
                    "failed writing to temporary model file '{}': {}",
                    tmp_path.display(),
                    e
                ))
            })?;

            downloaded += read as u64;
            match content_length {
                Some(total) => {
                    eprint!(
                        "\rDownloading SHARP model... {:.1} MB / {:.1} MB",
                        downloaded as f64 / (1024.0 * 1024.0),
                        total as f64 / (1024.0 * 1024.0)
                    );
                }
                None => {
                    eprint!(
                        "\rDownloading SHARP model... {:.1} MB / unknown",
                        downloaded as f64 / (1024.0 * 1024.0)
                    );
                }
            }
        }
        eprintln!();

        file.flush().map_err(|e| {
            SharpError::Download(format!(
                "failed flushing temporary model file '{}': {}",
                tmp_path.display(),
                e
            ))
        })?;

        // Verify download completeness when Content-Length was provided.
        if let Some(expected) = content_length {
            if downloaded != expected {
                return Err(SharpError::Download(format!(
                    "incomplete download: expected {} bytes, got {}",
                    expected, downloaded
                )));
            }
        }

        // Sanity-check minimum size regardless of Content-Length.
        if downloaded < MIN_MODEL_SIZE {
            return Err(SharpError::Download(format!(
                "downloaded file is suspiciously small ({} bytes)",
                downloaded
            )));
        }

        Ok(())
    })();

    // Clean up temp file on any error.
    if write_result.is_err() {
        let _ = fs::remove_file(&tmp_path);
        return write_result;
    }

    fs::rename(&tmp_path, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        SharpError::Download(format!(
            "failed to finalize model at '{}': {}",
            dest.display(),
            e
        ))
    })?;

    Ok(())
}
