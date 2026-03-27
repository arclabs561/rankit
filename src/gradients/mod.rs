//! Analytical gradient computation for differentiable ranking operations.
//!
//! - **Core gradients**: Soft ranking and Spearman loss gradients
//! - **LambdaRank**: LambdaRank gradient computation for Learning to Rank
//! - **Ranking SVM**: Ranking SVM gradient computation

mod error;
mod lambdarank;
/// Natural gradient via Fisher information for softmax-based ranking losses.
pub mod natural;
mod ranking_svm;

use crate::rank::sigmoid;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub use error::GradientError;
pub use lambdarank::{
    compute_lambdarank_gradients, ndcg_at_k, LambdaRankParams, LambdaRankTrainer,
};
pub use natural::{fisher_information_softmax, natural_gradient_softmax, with_natural_gradient};
pub use ranking_svm::{
    compute_ranking_svm_gradients, pairwise_hinge_loss, RankingSVMParams, RankingSVMTrainer,
};

/// Compute the gradient of soft_rank with respect to input values.
///
/// Returns gradient matrix [n, n] where `grad[i][j] = d(rank[i])/d(values[j])`.
///
/// When compiled with the `parallel` feature, uses rayon for the outer loop.
pub fn soft_rank_gradient(
    values: &[f64],
    ranks: &[f64],
    regularization_strength: f64,
) -> Vec<Vec<f64>> {
    let n = values.len();

    if n == 0 || n == 1 {
        return vec![vec![0.0; n]; n];
    }

    let alpha = regularization_strength;

    #[cfg(feature = "parallel")]
    {
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| compute_gradient_row(i, values, ranks, alpha, n))
            .collect();
        rows
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut grad = vec![vec![0.0; n]; n];
        for i in 0..n {
            grad[i] = compute_gradient_row(i, values, ranks, alpha, n);
        }
        grad
    }
}

#[inline]
fn compute_gradient_row(i: usize, values: &[f64], ranks: &[f64], alpha: f64, n: usize) -> Vec<f64> {
    let mut row = vec![0.0; n];

    if !values[i].is_finite() || !ranks[i].is_finite() {
        return row;
    }

    let valid_comparisons = values
        .iter()
        .enumerate()
        .filter(|&(j, v)| i != j && v.is_finite())
        .count();

    if valid_comparisons == 0 {
        return row;
    }

    let norm_factor = (n - 1) as f64 / valid_comparisons as f64;
    let alpha_norm = alpha * norm_factor;

    let sig_derivs: Vec<f64> = (0..n)
        .map(|j| {
            if i != j && values[j].is_finite() {
                let diff = values[i] - values[j];
                let sig = sigmoid(alpha * diff);
                sig * (1.0 - sig)
            } else {
                0.0
            }
        })
        .collect();

    let diagonal_sum: f64 = sig_derivs.iter().sum();
    row[i] = alpha_norm * diagonal_sum;

    for k in 0..n {
        if k != i && values[k].is_finite() {
            row[k] = -alpha_norm * sig_derivs[k];
        }
    }

    row
}

/// Compute gradient of Spearman loss with respect to predictions.
///
/// Spearman loss = 1 - Pearson_correlation(rank(pred), rank(target))
pub fn spearman_loss_gradient(
    predictions: &[f64],
    _targets: &[f64],
    pred_ranks: &[f64],
    target_ranks: &[f64],
    regularization_strength: f64,
) -> Vec<f64> {
    let n = predictions.len();

    if n < 2 {
        return vec![0.0; n];
    }

    let pred_mean = pred_ranks.iter().sum::<f64>() / n as f64;
    let target_mean = target_ranks.iter().sum::<f64>() / n as f64;

    let mut pred_var = 0.0;
    let mut target_var = 0.0;
    let mut covariance = 0.0;

    for i in 0..n {
        let pred_diff = pred_ranks[i] - pred_mean;
        let target_diff = target_ranks[i] - target_mean;
        pred_var += pred_diff * pred_diff;
        target_var += target_diff * target_diff;
        covariance += pred_diff * target_diff;
    }

    let denominator = (pred_var * target_var).sqrt();
    if denominator < 1e-8 {
        return vec![0.0; n];
    }

    let correlation = covariance / denominator;
    let pred_std = pred_var.sqrt();
    let target_std = target_var.sqrt();
    let inv_denom = 1.0 / denominator;

    let mut corr_grad_wrt_ranks = vec![0.0; n];
    for i in 0..n {
        let pred_diff = pred_ranks[i] - pred_mean;
        let target_diff = target_ranks[i] - target_mean;

        let term1 = target_diff * inv_denom;
        let term2 = correlation * pred_diff * target_std * inv_denom / pred_std;
        corr_grad_wrt_ranks[i] = term1 - term2;
    }

    let loss_grad_wrt_ranks: Vec<f64> = corr_grad_wrt_ranks.iter().map(|&g| -g).collect();

    let rank_grad = soft_rank_gradient(predictions, pred_ranks, regularization_strength);

    #[cfg(feature = "parallel")]
    {
        (0..n)
            .into_par_iter()
            .map(|j| {
                loss_grad_wrt_ranks
                    .iter()
                    .zip(rank_grad.iter())
                    .map(|(&lg, row)| lg * row[j])
                    .sum()
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut grad = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                grad[j] += loss_grad_wrt_ranks[i] * rank_grad[i][j];
            }
        }
        grad
    }
}

/// Sigmoid derivative: sigma'(x) = sigma(x) * (1 - sigma(x))
pub fn sigmoid_derivative(x: f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rank::soft_rank;

    #[test]
    fn test_soft_rank_gradient_basic() {
        let values = vec![1.0, 2.0, 3.0];
        let ranks = soft_rank(&values, 1.0);
        let grad = soft_rank_gradient(&values, &ranks, 1.0);

        assert_eq!(grad.len(), 3);
        assert_eq!(grad[0].len(), 3);

        assert!(grad[0][0] > 0.0);
        assert!(grad[1][1] > 0.0);
        assert!(grad[2][2] > 0.0);
    }

    #[test]
    fn test_spearman_loss_gradient_basic() {
        let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];

        let pred_ranks = soft_rank(&predictions, 1.0);
        let target_ranks = soft_rank(&targets, 1.0);

        let grad = spearman_loss_gradient(&predictions, &targets, &pred_ranks, &target_ranks, 1.0);

        assert_eq!(grad.len(), predictions.len());
        assert!(grad.iter().all(|&g| g.is_finite()));
    }
}
