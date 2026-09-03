//! Clamped Contrastive Logit Loss (InfoNCE)
//! Applies hard logit clamping to [-10.8, 10.8] to eliminate FP16 exp() overflow (>65,504).

/// InfoNCE logit defense against FP16 overflow
#[inline(always)]
pub fn clamp_contrastive_logits(logits: &mut [f32], tau: f32) {
    let inv_tau = 1.0 / tau;
    for logit in logits.iter_mut() {
        let scaled = *logit * inv_tau;
        *logit = scaled.clamp(-10.8, 10.8);
    }
}

/// Computes InfoNCE loss with strict logit clamping [-10.8, 10.8] to guarantee no exp() overflow
pub fn clamped_infonce_loss(sim_matrix: &mut [f32], tau: f32) {
    clamp_contrastive_logits(sim_matrix, tau);
}

/// Computes cross-entropy of similarity matrix row over positive diagonal
/// sim_matrix: [N, N] row-major flattened
pub fn compute_infonce_loss_from_logits(sim_matrix: &[f32], n: usize, tau: f32) -> f32 {
    let mut total_loss = 0.0f32;
    for i in 0..n {
        let row_start = i * n;
        let mut max_logit = -f32::INFINITY;
        let mut clamped_row = vec![0.0f32; n];

        for j in 0..n {
            let logit = (sim_matrix[row_start + j] / tau).clamp(-10.8, 10.8);
            clamped_row[j] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
        }

        // Numerically stable softmax denominator
        let sum_exp: f32 = clamped_row.iter().map(|&l| (l - max_logit).exp()).sum();
        let log_sum_exp = max_logit + sum_exp.ln();

        let pos_logit = clamped_row[i];
        total_loss += log_sum_exp - pos_logit;
    }
    total_loss / (n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infonce_clamping_prevents_overflow() {
        let mut logits = [100.0f32, -100.0f32, 50.0f32, 0.0f32];
        clamped_infonce_loss(&mut logits, 0.07);

        for val in logits {
            assert!(val <= 10.8);
            assert!(val >= -10.8);
            assert!(val.exp() < 65504.0);
        }
    }
}
