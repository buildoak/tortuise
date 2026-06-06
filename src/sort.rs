use rayon::prelude::*;

use crate::splat::ProjectedSplat;

pub fn sort_by_depth(projected_splats: &mut [ProjectedSplat]) {
    projected_splats.par_sort_unstable_by(|a, b| {
        a.depth
            .total_cmp(&b.depth)
            .then_with(|| a.original_index.cmp(&b.original_index))
    });
}

#[cfg(test)]
mod tests {
    use super::sort_by_depth;
    use crate::splat::ProjectedSplat;

    fn projected(depth: f32, original_index: usize) -> ProjectedSplat {
        ProjectedSplat {
            screen_x: 0.0,
            screen_y: 0.0,
            depth,
            radius_x: 1.0,
            radius_y: 1.0,
            color: [0, 0, 0],
            opacity: 1.0,
            inv_cov_a: 1.0,
            inv_cov_b: 0.0,
            inv_cov_c: 1.0,
            original_index,
        }
    }

    #[test]
    fn equal_depth_uses_original_index_tiebreaker() {
        let mut splats = vec![projected(1.0, 9), projected(1.0, 2), projected(1.0, 5)];

        sort_by_depth(&mut splats);

        let order: Vec<usize> = splats.iter().map(|splat| splat.original_index).collect();
        assert_eq!(order, vec![2, 5, 9]);
    }

    #[test]
    fn nan_depths_have_total_order() {
        let mut splats = vec![
            projected(f32::NAN, 4),
            projected(0.0, 2),
            projected(f32::NEG_INFINITY, 1),
            projected(f32::INFINITY, 3),
        ];

        sort_by_depth(&mut splats);

        let order: Vec<usize> = splats.iter().map(|splat| splat.original_index).collect();
        assert_eq!(order, vec![1, 2, 3, 4]);
    }
}
