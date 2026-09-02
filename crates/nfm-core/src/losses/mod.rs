pub mod infonce;
pub mod vicreg;

pub use infonce::{clamped_infonce_loss, compute_infonce_loss_from_logits};
pub use vicreg::vicreg_variance_hinge;
