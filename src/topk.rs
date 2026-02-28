//! Differentiable top-k selection.
//!
//! From: "Differentiable Top-k Operator with Optimal Transport" (NeurIPS 2020)

use crate::rank::sigmoid;

/// Differentiable Top-K selection.
///
/// Selects top-k elements in a differentiable manner using soft rank indicators.
///
/// # Returns
///
/// `(weighted_values, weighted_ranks)` where elements outside top-k are
/// attenuated toward zero.
pub fn differentiable_topk(
    values: &[f64],
    k: usize,
    regularization_strength: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = values.len();

    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }

    if k >= n {
        let ranks = crate::rank::soft_rank(values, regularization_strength);
        return (values.to_vec(), ranks);
    }

    let ranks = crate::rank::soft_rank(values, regularization_strength);

    let mut topk_values = Vec::with_capacity(n);
    let mut topk_ranks = Vec::with_capacity(n);

    for i in 0..n {
        let indicator = sigmoid((k as f64 - ranks[i]) * regularization_strength);
        topk_values.push(values[i] * indicator);
        topk_ranks.push(ranks[i] * indicator);
    }

    (topk_values, topk_ranks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differentiable_topk() {
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let (topk_vals, _topk_ranks) = differentiable_topk(&values, 3, 1.0);

        assert_eq!(topk_vals.len(), values.len());
        assert!(topk_vals[0] > topk_vals[1]); // 5.0 > 1.0
    }
}
