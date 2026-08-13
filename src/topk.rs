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
    // `soft_rank` uses 0 for the lowest value and n - 1 for the highest.
    // The top-k boundary therefore lies halfway between ranks n-k-1 and n-k.
    let cutoff = (n - k) as f64 - 0.5;

    let mut topk_values = Vec::with_capacity(n);
    let mut topk_ranks = Vec::with_capacity(n);

    for i in 0..n {
        let indicator = sigmoid((ranks[i] - cutoff) * regularization_strength);
        topk_values.push(values[i] * indicator);
        topk_ranks.push(ranks[i] * indicator);
    }

    (topk_values, topk_ranks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn sharp_limit_selects_highest_values() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (weighted, _) = differentiable_topk(&values, 2, 100.0);
        let weights: Vec<_> = weighted
            .iter()
            .zip(values)
            .map(|(weighted, value)| weighted / value)
            .collect();

        assert!(weights[..3].iter().all(|&weight| weight < 1e-10));
        assert!(weights[3..].iter().all(|&weight| weight > 0.999));
    }

    #[test]
    fn zero_k_returns_empty_outputs() {
        assert_eq!(
            differentiable_topk(&[1.0, 2.0, 3.0], 0, 10.0),
            (Vec::new(), Vec::new())
        );
    }

    #[test]
    fn k_at_least_len_preserves_values_and_ranks() {
        let values = [3.0, 1.0, 2.0];
        let expected_ranks = crate::rank::soft_rank(&values, 10.0);

        for k in [values.len(), values.len() + 1] {
            assert_eq!(
                differentiable_topk(&values, k, 10.0),
                (values.to_vec(), expected_ranks.clone())
            );
        }
    }

    proptest! {
        #[test]
        fn selection_weights_are_monotone_in_value(
            raw_values in proptest::collection::vec(1u16..1000, 2..20),
        ) {
            let values: Vec<f64> = raw_values.into_iter().map(f64::from).collect();
            let k = values.len().div_ceil(2);
            let (weighted, _) = differentiable_topk(&values, k, 10.0);
            let weights: Vec<_> = weighted
                .iter()
                .zip(&values)
                .map(|(weighted, value)| weighted / value)
                .collect();

            for i in 0..values.len() {
                for j in 0..values.len() {
                    if values[i] > values[j] {
                        prop_assert!(weights[i] >= weights[j]);
                    }
                }
            }
        }

        #[test]
        fn selection_is_permutation_equivariant(
            raw_values in proptest::collection::vec(1u16..1000, 2..20),
        ) {
            let values: Vec<f64> = raw_values.into_iter().map(f64::from).collect();
            let k = values.len().div_ceil(2);
            let expected = differentiable_topk(&values, k, 3.0);

            let reversed: Vec<_> = values.iter().copied().rev().collect();
            let mut actual = differentiable_topk(&reversed, k, 3.0);
            actual.0.reverse();
            actual.1.reverse();

            for (actual, expected) in actual.0.iter().zip(&expected.0) {
                prop_assert!((actual - expected).abs() <= 1e-12);
            }
            for (actual, expected) in actual.1.iter().zip(&expected.1) {
                prop_assert!((actual - expected).abs() <= 1e-12);
            }
        }
    }
}
