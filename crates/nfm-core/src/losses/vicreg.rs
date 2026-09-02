//! VICReg Variance-Covariance Anti-Collapse Hinge
//! Penalizes representation collapse by maintaining std(z) >= gamma across batch dimensions.

/// Computes VICReg variance hinge loss: mean(max(0, gamma - std(z_j)))
/// z: [batch_size, dim] row-major flattened
pub fn vicreg_variance_hinge(z: &[f32], batch_size: usize, dim: usize, gamma: f32, eps: f32) -> f32 {
    assert_eq!(z.len(), batch_size * dim);
    if batch_size <= 1 {
        return gamma;
    }

    let mut total_hinge = 0.0f32;

    for j in 0..dim {
        // Compute channel mean
        let mut mean = 0.0f32;
        for i in 0..batch_size {
            mean += z[i * dim + j];
        }
        mean /= batch_size as f32;

        // Compute channel variance
        let mut var = 0.0f32;
        for i in 0..batch_size {
            let diff = z[i * dim + j] - mean;
            var += diff * diff;
        }
        var /= (batch_size - 1) as f32;

        let std = (var + eps).sqrt();
        let hinge = (gamma - std).max(0.0);
        total_hinge += hinge;
    }

    total_hinge / (dim as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vicreg_variance_hinge_collapse_penalty() {
        let b = 4;
        let d = 2;
        // Collapsed representations: all zeros
        let collapsed = vec![0.0f32; b * d];
        let loss_collapsed = vicreg_variance_hinge(&collapsed, b, d, 1.0, 1e-4);
        assert!((loss_collapsed - 1.0).abs() < 1e-2);

        // Healthy representations with high variance
        let healthy = vec![
            -2.0, -2.0,
            -1.0, -1.0,
             1.0,  1.0,
             2.0,  2.0,
        ];
        let loss_healthy = vicreg_variance_hinge(&healthy, b, d, 1.0, 1e-4);
        assert_eq!(loss_healthy, 0.0);
    }
}
