//! Boundary-Clipped Poincaré Ball Chart & Gyro-Projection
//! Enforces radius clipping ||x|| <= 1 - eps and bounded conformal scale lambda_x <= 1000.0.

pub struct PoincareManifold {
    pub c: f32,
    pub eps: f32,
}

impl PoincareManifold {
    pub fn new(c: f32) -> Self {
        Self { c, eps: 1e-4 }
    }

    #[inline(always)]
    pub fn project_boundary(&self, vectors: &mut [f32], dim: usize) {
        let max_norm = 1.0 - self.eps;
        for chunk in vectors.chunks_mut(dim) {
            let norm_sq: f32 = chunk.iter().map(|v| v * v).sum();
            let norm = norm_sq.sqrt();
            if norm >= max_norm {
                let scale = max_norm / (norm + 1e-7);
                for v in chunk.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }

    #[inline(always)]
    pub fn conformal_factor(&self, x: &[f32]) -> f32 {
        let norm_sq: f32 = x.iter().map(|v| v * v).sum();
        let denom = (1.0 - self.c * norm_sq).max(self.eps);
        (2.0 / denom).min(1000.0)
    }
}

pub struct PoincareBall {
    pub c: f32,
    pub eps: f32,
}

impl PoincareBall {
    pub fn new(c: f32) -> Self {
        Self { c, eps: 1e-4 }
    }

    /// Projects vectors onto the Poincaré ball with strict radius clipping: ||x|| <= 1 - eps
    #[inline(always)]
    pub fn project(&self, x: &mut [f32], dim: usize) {
        let max_norm = 1.0 - self.eps;
        for chunk in x.chunks_mut(dim) {
            let norm_sq: f32 = chunk.iter().map(|v| v * v).sum();
            let norm = norm_sq.sqrt();
            if norm >= max_norm {
                let scale = max_norm / (norm + 1e-7);
                for v in chunk.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }

    /// Computes conformal factor lambda_x = 2 / (1 - c * ||x||^2), capped at 1000.0
    #[inline(always)]
    pub fn lambda(&self, x: &[f32]) -> f32 {
        let norm_sq: f32 = x.iter().map(|v| v * v).sum();
        let denom = (1.0 - self.c * norm_sq).max(self.eps);
        (2.0 / denom).min(1000.0)
    }

    /// Computes Poincaré geodesic distance between two vectors x and y
    /// d(x, y) = acosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2) * (1 - ||y||^2)))
    pub fn distance(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let x_norm_sq: f32 = x.iter().map(|v| v * v).sum();
        let y_norm_sq: f32 = y.iter().map(|v| v * v).sum();

        let diff_norm_sq: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum();

        let denom = ((1.0 - self.c * x_norm_sq) * (1.0 - self.c * y_norm_sq)).max(self.eps);
        let alpha = 1.0 + 2.0 * self.c * diff_norm_sq / denom;
        let alpha_clamped = alpha.max(1.0 + self.eps);

        // acosh(z) = ln(z + sqrt(z^2 - 1))
        (alpha_clamped + (alpha_clamped * alpha_clamped - 1.0).sqrt()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_projection_and_lambda() {
        let ball = PoincareBall::new(1.0);
        let mut x = [0.8f32, 0.8f32]; // norm = sqrt(1.28) ~ 1.131 > 1.0
        ball.project(&mut x, 2);

        let norm_after = (x[0] * x[0] + x[1] * x[1]).sqrt();
        assert!(norm_after <= 1.0 - 1e-4 + 1e-6);

        let lambda_val = ball.lambda(&x);
        assert!(lambda_val <= 1000.0);
        assert!(lambda_val > 0.0);
    }
}
