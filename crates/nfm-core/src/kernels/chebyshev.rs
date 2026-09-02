//! Order-2 Chebyshev Functional Matrix Tile Contractions
//! Computes the polynomial recurrence:
//!   T_0(x) = 1
//!   T_1(x) = x
//!   T_2(x) = 2x^2 - 1
//! in CPU vector registers/SRAM with zero intermediate allocations.

/// Evaluates Order-2 Chebyshev polynomial expansions for a slice of values
/// into a contiguous destination slice of length `3 * src.len()`.
///
/// Layout of `dst`:
/// - `dst[0..len]`: T_0(x) = 1.0
/// - `dst[len..2*len]`: T_1(x) = clamp(x, -1.0, 1.0)
/// - `dst[2*len..3*len]`: T_2(x) = 2.0 * x^2 - 1.0
#[inline]
pub fn eval_chebyshev_order2_tile(src: &[f32], dst: &mut [f32]) {
    assert_eq!(
        src.len() * 3,
        dst.len(),
        "Destination slice must be exactly 3 times the source length"
    );
    let len = src.len();
    for i in 0..len {
        let x = src[i].clamp(-1.0, 1.0);
        dst[i] = 1.0;                         // T_0(x)
        dst[len + i] = x;                     // T_1(x)
        dst[2 * len + i] = 2.0 * x * x - 1.0; // T_2(x)
    }
}

/// Contracts a 16x16 tile using pre-computed Chebyshev basis polynomials and weights.
/// `tile_16x16`: 256 elements
/// `weights_order3`: 3 * 256 elements (weights for T0, T1, T2)
/// `out`: 256 elements
pub fn contract_tile_16x16(tile: &[f32], weights: &[f32], out: &mut [f32]) {
    assert_eq!(tile.len(), 256);
    assert_eq!(weights.len(), 3 * 256);
    assert_eq!(out.len(), 256);

    let mut basis = [0.0f32; 3 * 256];
    eval_chebyshev_order2_tile(tile, &mut basis);

    for i in 0..256 {
        let t0 = basis[i] * weights[i];
        let t1 = basis[256 + i] * weights[256 + i];
        let t2 = basis[512 + i] * weights[512 + i];
        out[i] = t0 + t1 + t2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_chebyshev_order2() {
        let src = [0.0f32, 0.5f32, 1.0f32, -1.0f32];
        let mut dst = [0.0f32; 12];
        eval_chebyshev_order2_tile(&src, &mut dst);

        // T_0(x) = 1.0
        for i in 0..4 {
            assert_eq!(dst[i], 1.0);
        }

        // T_1(x) = x
        assert_eq!(dst[4], 0.0);
        assert_eq!(dst[5], 0.5);
        assert_eq!(dst[6], 1.0);
        assert_eq!(dst[7], -1.0);

        // T_2(x) = 2x^2 - 1
        assert_eq!(dst[8], -1.0);       // 2*(0)^2 - 1 = -1
        assert_eq!(dst[9], -0.5);       // 2*(0.25) - 1 = -0.5
        assert_eq!(dst[10], 1.0);       // 2*(1)^2 - 1 = 1
        assert_eq!(dst[11], 1.0);       // 2*(-1)^2 - 1 = 1
    }
}
