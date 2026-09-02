//! Inline Terminal Early Warning Monitor & Anomaly Guard
//! Inspects step telemetry in real-time, detecting loss surges, NaN explosions,
//! and hyperbolic boundary saturation.

#[derive(Debug, Clone, PartialEq)]
pub enum SafetyStatus {
    Ok,
    Warning(String),
    CriticalAbort(String),
}

pub struct TerminalSafetyMonitor {
    loss_spike_threshold: f32,
    consecutive_spikes: usize,
    max_consecutive_spikes: usize,
    boundary_threshold: f32,
}

impl TerminalSafetyMonitor {
    pub fn new(loss_threshold: f32, max_consecutive: usize) -> Self {
        Self {
            loss_spike_threshold: loss_threshold,
            consecutive_spikes: 0,
            max_consecutive_spikes: max_consecutive,
            boundary_threshold: 0.9999,
        }
    }

    pub fn inspect(
        &mut self,
        epoch: usize,
        step: usize,
        stream: &str,
        loss: f32,
        ppl: f32,
        radius: f32
    ) -> SafetyStatus {
        // 1. Loss spike / NaN overflow detection
        if loss.is_nan() || loss.is_infinite() || loss > self.loss_spike_threshold {
            self.consecutive_spikes += 1;
            let warn_msg = format!(
                "[WARNING] Epoch {epoch} | Step {step} | Stream: {stream} -> Loss surged to {loss:.4} (PPL: {ppl:.2})!"
            );

            if self.consecutive_spikes >= self.max_consecutive_spikes {
                let abort_msg = format!(
                    "[CRITICAL ABORT] Stream {stream} weights irrecoverably diverging after {} consecutive spikes.",
                    self.consecutive_spikes
                );
                return SafetyStatus::CriticalAbort(abort_msg);
            }
            return SafetyStatus::Warning(warn_msg);
        } else if self.consecutive_spikes > 0 {
            self.consecutive_spikes -= 1;
        }

        // 2. Poincaré boundary alert
        if radius >= self.boundary_threshold {
            let boundary_msg = format!(
                "[MANIFOLD ALERT] Epoch {epoch} | Step {step} -> Poincaré radius reached {radius:.5}. Conformal factor saturating."
            );
            return SafetyStatus::Warning(boundary_msg);
        }

        SafetyStatus::Ok
    }

    pub fn reset(&mut self) {
        self.consecutive_spikes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_safety_monitor_spikes() {
        let mut monitor = TerminalSafetyMonitor::new(10.0, 3);

        // Healthy
        assert_eq!(monitor.inspect(1, 1, "stream_1", 1.5, 12.0, 0.5), SafetyStatus::Ok);

        // Spike 1
        match monitor.inspect(1, 2, "stream_1", 25.0, 50.0, 0.5) {
            SafetyStatus::Warning(_) => {}
            _ => panic!("Expected Warning on spike 1"),
        }

        // Spike 2
        match monitor.inspect(1, 3, "stream_1", 28.0, 60.0, 0.5) {
            SafetyStatus::Warning(_) => {}
            _ => panic!("Expected Warning on spike 2"),
        }

        // Spike 3 -> CriticalAbort
        match monitor.inspect(1, 4, "stream_1", 30.0, 80.0, 0.5) {
            SafetyStatus::CriticalAbort(_) => {}
            _ => panic!("Expected CriticalAbort on spike 3"),
        }
    }
}
