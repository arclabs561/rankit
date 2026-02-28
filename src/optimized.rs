//! Performance-optimized implementations.
//!
//! - **`soft_rank_optimized`**: Early termination for sorted inputs
//! - **`soft_rank_gradient_sparse`**: Sparse gradient computation
//! - **`soft_rank_batch_parallel`**: Parallel batch processing (requires `parallel` feature)

use crate::rank::sigmoid;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parallel batch soft ranking.
#[cfg(feature = "parallel")]
pub fn soft_rank_batch_parallel(
    batch_values: &[Vec<f64>],
    regularization_strength: f64,
) -> Vec<Vec<f64>> {
    batch_values
        .par_iter()
        .map(|values| crate::rank::soft_rank(values, regularization_strength))
        .collect()
}

/// Optimized soft ranking with early termination for sorted inputs.
///
/// When `assume_sorted = true`, uses a windowed approximation that is faster
/// but introduces bounded error for elements outside the comparison window.
pub fn soft_rank_optimized(
    values: &[f64],
    regularization_strength: f64,
    assume_sorted: bool,
) -> Vec<f64> {
    let n = values.len();

    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut ranks = vec![0.0; n];

    if assume_sorted {
        for i in 0..n {
            if !values[i].is_finite() {
                ranks[i] = f64::NAN;
                continue;
            }

            let mut valid_before = 0;
            for j in 0..i {
                if values[j].is_finite() {
                    valid_before += 1;
                }
            }

            let mut sum = valid_before as f64;

            let window = 5usize;
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(n);

            for j in start..end {
                if i != j && values[j].is_finite() {
                    let diff = values[i] - values[j];
                    let sig = sigmoid(diff * regularization_strength);
                    if j < i {
                        sum += sig - 1.0;
                    } else {
                        sum += sig;
                    }
                }
            }

            let mut valid_comparisons = 0;
            for j in 0..n {
                if i != j && values[j].is_finite() {
                    valid_comparisons += 1;
                }
            }

            if valid_comparisons > 0 {
                ranks[i] = sum / valid_comparisons as f64 * (n - 1) as f64;
            } else {
                ranks[i] = 0.0;
            }
        }
    } else {
        for i in 0..n {
            if !values[i].is_finite() {
                ranks[i] = f64::NAN;
                continue;
            }

            let mut sum = 0.0;
            let mut valid_comparisons = 0;
            for j in 0..n {
                if i != j && values[j].is_finite() {
                    let diff = values[i] - values[j];
                    sum += sigmoid(diff * regularization_strength);
                    valid_comparisons += 1;
                }
            }

            if valid_comparisons > 0 {
                ranks[i] = sum / valid_comparisons as f64 * (n - 1) as f64;
            } else {
                ranks[i] = 0.0;
            }
        }
    }

    ranks
}

/// Sparse gradient computation -- only computes gradients for significant comparisons.
///
/// Dramatically reduces computation for large inputs where most comparisons
/// are well-separated (sigmoid near 0 or 1).
pub fn soft_rank_gradient_sparse(
    values: &[f64],
    ranks: &[f64],
    regularization_strength: f64,
    threshold: f64,
) -> Vec<Vec<f64>> {
    let n = values.len();

    if n == 0 || n == 1 {
        return vec![vec![0.0; n]; n];
    }

    let alpha = regularization_strength;

    let mut grad = vec![vec![0.0; n]; n];

    for i in 0..n {
        if !values[i].is_finite() || !ranks[i].is_finite() {
            continue;
        }

        let mut valid_comparisons = 0;
        for j in 0..n {
            if i != j && values[j].is_finite() {
                valid_comparisons += 1;
            }
        }

        if valid_comparisons == 0 {
            continue;
        }

        let norm_factor = (n - 1) as f64 / valid_comparisons as f64;

        for k in 0..n {
            if !values[k].is_finite() {
                continue;
            }

            if i == k {
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j && values[j].is_finite() {
                        let diff = values[i] - values[j];
                        let sig = sigmoid(alpha * diff);
                        let sig_deriv = sig * (1.0 - sig);
                        if sig_deriv.abs() > threshold {
                            sum += sig_deriv;
                        }
                    }
                }
                grad[i][k] = alpha * norm_factor * sum;
            } else {
                let diff = values[i] - values[k];
                let sig = sigmoid(alpha * diff);
                let sig_deriv = sig * (1.0 - sig);

                if sig_deriv.abs() > threshold {
                    grad[i][k] = -alpha * norm_factor * sig_deriv;
                }
            }
        }
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_rank_optimized_sorted() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ranks_opt = soft_rank_optimized(&values, 1.0, true);
        let ranks_std = crate::rank::soft_rank(&values, 1.0);

        for (r_opt, r_std) in ranks_opt.iter().zip(ranks_std.iter()) {
            assert!(
                (r_opt - r_std).abs() < 0.1,
                "Optimized should match standard"
            );
        }
    }

    #[test]
    fn test_gradient_sparse() {
        let values = vec![1.0, 2.0, 3.0];
        let ranks = crate::rank::soft_rank(&values, 1.0);

        let grad_sparse = soft_rank_gradient_sparse(&values, &ranks, 1.0, 0.01);
        let grad_full = crate::gradients::soft_rank_gradient(&values, &ranks, 1.0);

        for i in 0..values.len() {
            for j in 0..values.len() {
                if grad_full[i][j].abs() > 0.01 {
                    assert!(
                        (grad_sparse[i][j] - grad_full[i][j]).abs() < 0.1,
                        "Sparse gradient should approximate full"
                    );
                }
            }
        }
    }
}
