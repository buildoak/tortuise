use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct LiveFrameTelemetry {
    pub schema_version: u32,
    pub frame: u64,
    pub frame_ms: f64,
    pub target_ms: f64,
    pub sleep_ms: f64,
    pub input_events: usize,
    pub oldest_input_age_ms: f64,
    pub render_ms: f64,
    pub gpu_wait_ms: f64,
    pub convert_ms: f64,
    pub encode_ms: f64,
    pub write_ms: f64,
    pub flush_ms: f64,
    pub payload_bytes: usize,
    pub base64_bytes: usize,
    pub chunks: usize,
    pub effective_path: &'static str,
    pub quality: &'static str,
    pub sort_path: &'static str,
    pub lod_mode: &'static str,
    pub lod_mapping: &'static str,
    pub source_splat_count: usize,
    pub active_splat_count: usize,
    pub valid_count: usize,
    pub actual_total_overlaps: u32,
    pub overflow_flag: u32,
    pub retry_count: u32,
    pub previous_telemetry_write_ms: f64,
}

impl Default for LiveFrameTelemetry {
    fn default() -> Self {
        Self {
            schema_version: 1,
            frame: 0,
            frame_ms: 0.0,
            target_ms: 0.0,
            sleep_ms: 0.0,
            input_events: 0,
            oldest_input_age_ms: 0.0,
            render_ms: 0.0,
            gpu_wait_ms: 0.0,
            convert_ms: 0.0,
            encode_ms: 0.0,
            write_ms: 0.0,
            flush_ms: 0.0,
            payload_bytes: 0,
            base64_bytes: 0,
            chunks: 0,
            effective_path: "unknown",
            quality: "cpu",
            sort_path: "cpu",
            lod_mode: "off",
            lod_mapping: "identity",
            source_splat_count: 0,
            active_splat_count: 0,
            valid_count: 0,
            actual_total_overlaps: 0,
            overflow_flag: 0,
            retry_count: 0,
            previous_telemetry_write_ms: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct LiveTelemetryState {
    writer: Option<BufWriter<File>>,
    pub last: LiveFrameTelemetry,
}

impl LiveTelemetryState {
    pub fn disabled() -> Self {
        Self {
            writer: None,
            last: LiveFrameTelemetry::default(),
        }
    }

    pub fn to_path(path: Option<&Path>) -> io::Result<Self> {
        let Some(path) = path else {
            return Ok(Self::disabled());
        };
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(Self {
            writer: Some(BufWriter::new(File::create(path)?)),
            last: LiveFrameTelemetry::default(),
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_enabled(&self) -> bool {
        self.writer.is_some()
    }

    pub fn record(&mut self, frame: LiveFrameTelemetry) -> io::Result<f64> {
        let started = std::time::Instant::now();
        self.last = frame;
        let Some(writer) = self.writer.as_mut() else {
            return Ok(0.0);
        };
        writeln!(
            writer,
            concat!(
                "{{",
                "\"schema_version\":{},",
                "\"frame\":{},",
                "\"frame_ms\":{:.3},",
                "\"target_ms\":{:.3},",
                "\"sleep_ms\":{:.3},",
                "\"input_events\":{},",
                "\"oldest_input_age_ms\":{:.3},",
                "\"render_ms\":{:.3},",
                "\"gpu_wait_ms\":{:.3},",
                "\"convert_ms\":{:.3},",
                "\"encode_ms\":{:.3},",
                "\"write_ms\":{:.3},",
                "\"flush_ms\":{:.3},",
                "\"payload_bytes\":{},",
                "\"base64_bytes\":{},",
                "\"chunks\":{},",
                "\"effective_path\":\"{}\",",
                "\"quality\":\"{}\",",
                "\"sort_path\":\"{}\",",
                "\"lod_mode\":\"{}\",",
                "\"lod_mapping\":\"{}\",",
                "\"source_splat_count\":{},",
                "\"active_splat_count\":{},",
                "\"valid_count\":{},",
                "\"actual_total_overlaps\":{},",
                "\"overflow_flag\":{},",
                "\"retry_count\":{},",
                "\"previous_telemetry_write_ms\":{:.3}",
                "}}"
            ),
            self.last.schema_version,
            self.last.frame,
            self.last.frame_ms,
            self.last.target_ms,
            self.last.sleep_ms,
            self.last.input_events,
            self.last.oldest_input_age_ms,
            self.last.render_ms,
            self.last.gpu_wait_ms,
            self.last.convert_ms,
            self.last.encode_ms,
            self.last.write_ms,
            self.last.flush_ms,
            self.last.payload_bytes,
            self.last.base64_bytes,
            self.last.chunks,
            self.last.effective_path,
            self.last.quality,
            self.last.sort_path,
            self.last.lod_mode,
            self.last.lod_mapping,
            self.last.source_splat_count,
            self.last.active_splat_count,
            self.last.valid_count,
            self.last.actual_total_overlaps,
            self.last.overflow_flag,
            self.last.retry_count,
            self.last.previous_telemetry_write_ms
        )?;
        writer.flush()?;
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_telemetry_keeps_last_frame_without_io() {
        let mut state = LiveTelemetryState::disabled();
        let frame = LiveFrameTelemetry {
            frame: 7,
            effective_path: "metal_kitty",
            ..LiveFrameTelemetry::default()
        };

        state.record(frame).unwrap();

        assert_eq!(state.last.frame, 7);
        assert_eq!(state.last.effective_path, "metal_kitty");
        assert!(!state.is_enabled());
    }

    #[test]
    fn telemetry_writer_emits_jsonl() {
        let path = std::env::temp_dir().join(format!(
            "tortuise-live-telemetry-{}-{}.jsonl",
            std::process::id(),
            17
        ));
        let _ = std::fs::remove_file(&path);
        {
            let mut state = LiveTelemetryState::to_path(Some(&path)).unwrap();
            let frame = LiveFrameTelemetry {
                frame: 3,
                effective_path: "metal_kitty",
                payload_bytes: 12,
                valid_count: 5,
                active_splat_count: 7,
                source_splat_count: 9,
                ..LiveFrameTelemetry::default()
            };

            state.record(frame).unwrap();
        }

        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(text.contains("\"schema_version\":1"));
        assert!(text.contains("\"frame\":3"));
        assert!(text.contains("\"effective_path\":\"metal_kitty\""));
        assert!(text.contains("\"valid_count\":5"));
        assert!(text.contains("\"previous_telemetry_write_ms\":0.000"));
        assert!(text.ends_with('\n'));
    }
}
