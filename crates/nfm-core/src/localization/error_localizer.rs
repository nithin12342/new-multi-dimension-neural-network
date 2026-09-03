pub struct DualStageLocalizer {
    variance_floor: f32,
    grad_norm_cap: f32,
}

impl DualStageLocalizer {
    pub fn new(variance_floor: f32, grad_norm_cap: f32) -> Self {
        Self { variance_floor, grad_norm_cap }
    }

    /// Stage 1: Verifies modal encoder feature representations maintain active variance
    pub fn audit_intermediate_features(&self, features: &[f32], dim: usize) -> Result<(), &'static str> {
        if dim == 0 || features.is_empty() {
            return Err("Stage 1 Failure: Empty feature tensor or zero dimension");
        }
        for chunk in features.chunks(dim) {
            let n = chunk.len() as f32;
            let mean: f32 = chunk.iter().sum::<f32>() / n;
            let variance: f32 = chunk.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
            if variance < self.variance_floor {
                return Err("Stage 1 Failure: Intermediate representation collapse detected");
            }
        }
        Ok(())
    }

    /// Stage 2: Validates fused cross-modal gradients before parameter updates
    pub fn audit_fused_gradients(&self, grads: &[f32]) -> Result<(), &'static str> {
        let l2_norm_sq: f32 = grads.iter().map(|g| g * g).sum();
        if l2_norm_sq.sqrt() > self.grad_norm_cap {
            return Err("Stage 2 Failure: Fused gradient norm exceeds safety threshold");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage1_feature_variance_audit() {
        let localizer = DualStageLocalizer::new(1e-4, 100.0);
        
        // Healthy features
        let healthy = vec![1.0, -1.0, 2.0, -2.0];
        assert!(localizer.audit_intermediate_features(&healthy, 4).is_ok());

        // Collapsed features (variance = 0)
        let collapsed = vec![0.5, 0.5, 0.5, 0.5];
        assert!(localizer.audit_intermediate_features(&collapsed, 4).is_err());
    }

    #[test]
    fn test_stage2_gradient_norm_audit() {
        let localizer = DualStageLocalizer::new(1e-4, 10.0);
        
        // Safe gradients
        let safe_grads = vec![1.0, 2.0, 2.0]; // Norm = 3.0 < 10.0
        assert!(localizer.audit_fused_gradients(&safe_grads).is_ok());

        // Exploded gradients
        let exploded_grads = vec![10.0, 10.0, 10.0]; // Norm = sqrt(300) = 17.32 > 10.0
        assert!(localizer.audit_fused_gradients(&exploded_grads).is_err());
    }
}
