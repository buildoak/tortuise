use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct LiveFrameTelemetry {
    pub schema_version: u32,
    pub frame: u64,
    pub render_width: usize,
    pub render_height: usize,
    pub terminal_cols: usize,
    pub terminal_rows: usize,
    pub kitty_format: &'static str,
    pub kitty_scale_divisor: usize,
    pub kitty_frame_ms: u64,
    pub camera_x: f64,
    pub camera_y: f64,
    pub camera_z: f64,
    pub camera_yaw: f64,
    pub camera_pitch: f64,
    pub camera_fov_deg: f64,
    pub frame_ms: f64,
    pub target_ms: f64,
    pub sleep_ms: f64,
    pub input_events: usize,
    pub oldest_input_age_ms: f64,
    pub input_drain_ms: f64,
    pub interaction_latency_ms: f64,
    pub render_ms: f64,
    pub terminal_ms: f64,
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
    pub estimated_overlaps: usize,
    pub attempt_sort_count: usize,
    pub actual_total_overlaps: u32,
    pub overflow_flag: u32,
    pub retry_count: u32,
    pub tile_entries: u32,
    pub max_tile_range: u32,
    pub p95_tile_range: u32,
    pub p99_tile_range: u32,
    pub stage_timing_count: usize,
    pub previous_telemetry_write_ms: f64,
}

impl Default for LiveFrameTelemetry {
    fn default() -> Self {
        Self {
            schema_version: 3,
            frame: 0,
            render_width: 0,
            render_height: 0,
            terminal_cols: 0,
            terminal_rows: 0,
            kitty_format: "none",
            kitty_scale_divisor: 0,
            kitty_frame_ms: 0,
            camera_x: 0.0,
            camera_y: 0.0,
            camera_z: 0.0,
            camera_yaw: 0.0,
            camera_pitch: 0.0,
            camera_fov_deg: 0.0,
            frame_ms: 0.0,
            target_ms: 0.0,
            sleep_ms: 0.0,
            input_events: 0,
            oldest_input_age_ms: 0.0,
            input_drain_ms: 0.0,
            interaction_latency_ms: 0.0,
            render_ms: 0.0,
            terminal_ms: 0.0,
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
            estimated_overlaps: 0,
            attempt_sort_count: 0,
            actual_total_overlaps: 0,
            overflow_flag: 0,
            retry_count: 0,
            tile_entries: 0,
            max_tile_range: 0,
            p95_tile_range: 0,
            p99_tile_range: 0,
            stage_timing_count: 0,
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
                "\"render_width\":{},",
                "\"render_height\":{},",
                "\"terminal_cols\":{},",
                "\"terminal_rows\":{},",
                "\"kitty_format\":\"{}\",",
                "\"kitty_scale_divisor\":{},",
                "\"kitty_frame_ms\":{},",
                "\"camera\":{{",
                "\"position\":[{:.6},{:.6},{:.6}],",
                "\"yaw\":{:.6},",
                "\"pitch\":{:.6},",
                "\"fov_deg\":{:.6}",
                "}},",
                "\"frame_ms\":{:.3},",
                "\"target_ms\":{:.3},",
                "\"sleep_ms\":{:.3},",
                "\"input_events\":{},",
                "\"oldest_input_age_ms\":{:.3},",
                "\"input_drain_ms\":{:.3},",
                "\"interaction_latency_ms\":{:.3},",
                "\"render_ms\":{:.3},",
                "\"terminal_ms\":{:.3},",
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
                "\"estimated_overlaps\":{},",
                "\"attempt_sort_count\":{},",
                "\"actual_total_overlaps\":{},",
                "\"overflow_flag\":{},",
                "\"retry_count\":{},",
                "\"tile_entries\":{},",
                "\"max_tile_range\":{},",
                "\"p95_tile_range\":{},",
                "\"p99_tile_range\":{},",
                "\"stage_timing_count\":{},",
                "\"previous_telemetry_write_ms\":{:.3}",
                "}}"
            ),
            self.last.schema_version,
            self.last.frame,
            self.last.render_width,
            self.last.render_height,
            self.last.terminal_cols,
            self.last.terminal_rows,
            self.last.kitty_format,
            self.last.kitty_scale_divisor,
            self.last.kitty_frame_ms,
            self.last.camera_x,
            self.last.camera_y,
            self.last.camera_z,
            self.last.camera_yaw,
            self.last.camera_pitch,
            self.last.camera_fov_deg,
            self.last.frame_ms,
            self.last.target_ms,
            self.last.sleep_ms,
            self.last.input_events,
            self.last.oldest_input_age_ms,
            self.last.input_drain_ms,
            self.last.interaction_latency_ms,
            self.last.render_ms,
            self.last.terminal_ms,
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
            self.last.estimated_overlaps,
            self.last.attempt_sort_count,
            self.last.actual_total_overlaps,
            self.last.overflow_flag,
            self.last.retry_count,
            self.last.tile_entries,
            self.last.max_tile_range,
            self.last.p95_tile_range,
            self.last.p99_tile_range,
            self.last.stage_timing_count,
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
                input_drain_ms: 0.4,
                interaction_latency_ms: 12.5,
                terminal_ms: 1.2,
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
        assert!(text.contains("\"schema_version\":3"));
        assert!(text.contains("\"frame\":3"));
        assert!(text.contains("\"camera\""));
        assert!(text.contains("\"input_drain_ms\":0.400"));
        assert!(text.contains("\"interaction_latency_ms\":12.500"));
        assert!(text.contains("\"terminal_ms\":1.200"));
        assert!(text.contains("\"effective_path\":\"metal_kitty\""));
        assert!(text.contains("\"valid_count\":5"));
        assert!(text.contains("\"previous_telemetry_write_ms\":0.000"));
        assert!(text.ends_with('\n'));
    }
}
