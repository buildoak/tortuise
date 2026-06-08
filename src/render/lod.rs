use crate::math::Vec3;
use crate::splat::Splat;

use super::{MetalLodMode, MetalLodOrder};

pub fn build_metal_lod_indices(
    splats: &[Splat],
    mode: MetalLodMode,
    order: MetalLodOrder,
    active_count: usize,
) -> Result<Vec<u32>, String> {
    let source_count = splats.len();
    let mut indices = match (mode, order) {
        (MetalLodMode::Off, _) => (0..source_count).collect::<Vec<_>>(),
        (MetalLodMode::Fixed, MetalLodOrder::FloorEven) => (0..active_count)
            .map(|idx| floor_even_source_index(idx, active_count, source_count))
            .collect::<Vec<_>>(),
        (MetalLodMode::Fixed, MetalLodOrder::Voxel) => build_voxel_lod_order(splats),
    };
    if mode == MetalLodMode::Fixed {
        indices.truncate(active_count.min(source_count));
    }
    indices
        .into_iter()
        .map(|idx| u32::try_from(idx).map_err(|_| "LoD index exceeds u32 range".to_string()))
        .collect()
}

fn floor_even_source_index(out_idx: usize, active_count: usize, source_count: usize) -> usize {
    if active_count == 0 || source_count == 0 {
        return 0;
    }
    out_idx.saturating_mul(source_count) / active_count
}

fn build_voxel_lod_order(splats: &[Splat]) -> Vec<usize> {
    const GRID: usize = 32;
    const BUCKETS: usize = GRID * GRID * GRID;
    if splats.is_empty() {
        return Vec::new();
    }

    let mut min = splats[0].position;
    let mut max = splats[0].position;
    for splat in splats {
        min.x = min.x.min(splat.position.x);
        min.y = min.y.min(splat.position.y);
        min.z = min.z.min(splat.position.z);
        max.x = max.x.max(splat.position.x);
        max.y = max.y.max(splat.position.y);
        max.z = max.z.max(splat.position.z);
    }
    let extent = Vec3::new(
        (max.x - min.x).max(1e-6),
        (max.y - min.y).max(1e-6),
        (max.z - min.z).max(1e-6),
    );

    let mut buckets: Vec<Vec<(u32, usize)>> = (0..BUCKETS).map(|_| Vec::new()).collect();
    for (idx, splat) in splats.iter().enumerate() {
        let qx = quantize_axis(splat.position.x, min.x, extent.x, GRID);
        let qy = quantize_axis(splat.position.y, min.y, extent.y, GRID);
        let qz = quantize_axis(splat.position.z, min.z, extent.z, GRID);
        let voxel = (qz * GRID + qy) * GRID + qx;
        buckets[voxel].push((splat_importance_key(splat), idx));
    }

    let mut max_bucket_len = 0usize;
    for bucket in &mut buckets {
        bucket.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        max_bucket_len = max_bucket_len.max(bucket.len());
    }

    let mut order = Vec::with_capacity(splats.len());
    for layer in 0..max_bucket_len {
        for bucket in &buckets {
            if let Some((_, idx)) = bucket.get(layer) {
                order.push(*idx);
            }
        }
    }
    order
}

fn quantize_axis(value: f32, min: f32, extent: f32, grid: usize) -> usize {
    let normalized = ((value - min) / extent).clamp(0.0, 0.999_999);
    (normalized * grid as f32) as usize
}

fn splat_importance_key(splat: &Splat) -> u32 {
    let max_scale = splat
        .scale
        .x
        .max(splat.scale.y)
        .max(splat.scale.z)
        .max(1e-6);
    let importance = (splat.opacity.max(0.0) * max_scale * 1_000_000.0).clamp(0.0, u32::MAX as f32);
    importance as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splat::Splat;

    fn splat_at(x: f32, y: f32, z: f32, opacity: f32, scale: f32) -> Splat {
        Splat {
            position: Vec3::new(x, y, z),
            color: [255, 255, 255],
            opacity,
            scale: Vec3::new(scale, scale, scale),
            rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn floor_even_matches_previous_fixed_mapping() {
        let splats = vec![splat_at(0.0, 0.0, 0.0, 1.0, 1.0); 10];
        let indices =
            build_metal_lod_indices(&splats, MetalLodMode::Fixed, MetalLodOrder::FloorEven, 4)
                .unwrap();
        assert_eq!(indices, vec![0, 2, 5, 7]);
    }

    #[test]
    fn voxel_order_preserves_spatially_separate_splats_first() {
        let splats = vec![
            splat_at(0.0, 0.0, 0.0, 0.2, 0.1),
            splat_at(0.01, 0.0, 0.0, 1.0, 1.0),
            splat_at(10.0, 0.0, 0.0, 0.3, 0.1),
            splat_at(10.01, 0.0, 0.0, 1.0, 1.0),
        ];
        let indices =
            build_metal_lod_indices(&splats, MetalLodMode::Fixed, MetalLodOrder::Voxel, 2).unwrap();
        assert!(indices.contains(&1));
        assert!(indices.contains(&3));
    }
}
