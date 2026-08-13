//! Ranking SVM gradient computation.
//!
//! Based on:
//! - Herbrich et al. 1999, 2000: "Large Margin Rank Boundaries for Ordinal Regression"
//! - Joachims 2002: "Optimizing Search Engines using Clickthrough Data"
//! - Cao et al. 2006: "Adapting Ranking SVM to Document Retrieval"

use crate::gradients::error::GradientError;

/// Ranking SVM parameters.
#[derive(Debug, Clone, Copy)]
pub struct RankingSVMParams {
    /// Regularization parameter (C). Default: 1.0
    pub c: f32,
    /// Enable query normalization. Default: true
    pub query_normalization: bool,
    /// Enable cost sensitivity. Default: true
    pub cost_sensitivity: bool,
    /// Epsilon for numerical stability. Default: 1e-10
    pub epsilon: f32,
}

impl Default for RankingSVMParams {
    fn default() -> Self {
        Self {
            c: 1.0,
            query_normalization: true,
            cost_sensitivity: true,
            epsilon: 1e-10,
        }
    }
}

/// Pairwise hinge loss: max(0, 1 - (score_i - score_j)).
pub fn pairwise_hinge_loss(score_i: f32, score_j: f32, _params: RankingSVMParams) -> f32 {
    let score_diff = score_i - score_j;
    (1.0 - score_diff).max(0.0)
}

/// Compute Ranking SVM gradients for a ranked list.
///
/// # Errors
///
/// Returns `GradientError::EmptyInput` if inputs are empty.
/// Returns `GradientError::LengthMismatch` if scores and relevance differ in length.
pub fn compute_ranking_svm_gradients(
    scores: &[f32],
    relevance: &[f32],
    params: RankingSVMParams,
) -> Result<Vec<f32>, GradientError> {
    if scores.len() != relevance.len() {
        return Err(GradientError::LengthMismatch {
            scores_len: scores.len(),
            relevance_len: relevance.len(),
        });
    }

    if scores.is_empty() {
        return Err(GradientError::EmptyInput);
    }

    let n = scores.len();
    let mut gradients = vec![0.0; n];

    let mut valid_pairs = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if (relevance[i] - relevance[j]).abs() > params.epsilon {
                valid_pairs += 1;
            }
        }
    }

    let mu = if params.query_normalization && valid_pairs > 0 {
        1.0 / valid_pairs as f32
    } else {
        1.0
    };

    for i in 0..n {
        for j in (i + 1)..n {
            let rel_diff = relevance[i] - relevance[j];
            if rel_diff.abs() < params.epsilon {
                continue;
            }

            let (high_idx, low_idx) = if rel_diff > 0.0 { (i, j) } else { (j, i) };

            let score_diff = scores[high_idx] - scores[low_idx];

            if score_diff < 1.0 {
                let tau = if params.cost_sensitivity {
                    let min_rank = high_idx.min(low_idx);
                    1.0 / ((min_rank + 2) as f32).ln()
                } else {
                    1.0
                };

                let gradient_contribution = params.c * mu * tau;

                gradients[high_idx] -= gradient_contribution;
                gradients[low_idx] += gradient_contribution;
            }
        }
    }

    Ok(gradients)
}

/// Ranking SVM trainer.
pub struct RankingSVMTrainer {
    params: RankingSVMParams,
}

impl RankingSVMTrainer {
    /// Create a new Ranking SVM trainer.
    pub fn new(params: RankingSVMParams) -> Self {
        Self { params }
    }

    /// Compute gradients for a query-document list.
    pub fn compute_gradients(
        &self,
        scores: &[f32],
        relevance: &[f32],
    ) -> Result<Vec<f32>, GradientError> {
        compute_ranking_svm_gradients(scores, relevance, self.params)
    }
}

impl Default for RankingSVMTrainer {
    fn default() -> Self {
        Self::new(RankingSVMParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairwise_hinge_loss() {
        let params = RankingSVMParams::default();

        let loss1 = pairwise_hinge_loss(2.0, 0.5, params);
        assert_eq!(loss1, 0.0);

        let loss2 = pairwise_hinge_loss(1.0, 0.5, params);
        assert!((loss2 - 0.5).abs() < 1e-6);

        let loss3 = pairwise_hinge_loss(0.0, 1.0, params);
        assert!((loss3 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_ranking_svm_gradients() {
        let params = RankingSVMParams::default();
        let scores = vec![0.5, 0.8, 0.3];
        let relevance = vec![3.0, 1.0, 2.0];

        let gradients = compute_ranking_svm_gradients(&scores, &relevance, params).unwrap();

        assert_eq!(gradients.len(), 3);
        assert!(gradients[0] < 0.0);
        assert!(gradients[1] > 0.0);
    }

    fn ranking_hinge_objective(scores: &[f32], relevance: &[f32], c: f32) -> f32 {
        let mut loss = 0.0;
        for i in 0..scores.len() {
            for j in (i + 1)..scores.len() {
                if relevance[i] == relevance[j] {
                    continue;
                }
                let (high, low) = if relevance[i] > relevance[j] {
                    (i, j)
                } else {
                    (j, i)
                };
                loss += c * (1.0 - (scores[high] - scores[low])).max(0.0);
            }
        }
        loss
    }

    #[test]
    fn gradients_match_central_finite_differences() {
        let scores = [0.2, -0.1, 0.4];
        let relevance = [2.0, 0.0, 1.0];
        let params = RankingSVMParams {
            c: 1.7,
            query_normalization: false,
            cost_sensitivity: false,
            epsilon: 1e-10,
        };
        let actual = compute_ranking_svm_gradients(&scores, &relevance, params).unwrap();
        let step = 1e-3;

        for index in 0..scores.len() {
            let mut plus = scores;
            let mut minus = scores;
            plus[index] += step;
            minus[index] -= step;
            let expected = (ranking_hinge_objective(&plus, &relevance, params.c)
                - ranking_hinge_objective(&minus, &relevance, params.c))
                / (2.0 * step);
            assert!(
                (actual[index] - expected).abs() < 1e-3,
                "gradient[{index}] = {}, finite difference = {expected}",
                actual[index]
            );
        }
    }

    #[test]
    fn gradient_descent_step_increases_preferred_margin_and_reduces_loss() {
        let scores = [0.2, 0.0];
        let relevance = [1.0, 0.0];
        let params = RankingSVMParams {
            c: 1.0,
            query_normalization: false,
            cost_sensitivity: false,
            epsilon: 1e-10,
        };
        let gradient = compute_ranking_svm_gradients(&scores, &relevance, params).unwrap();
        let learning_rate = 0.1;
        let updated = [
            scores[0] - learning_rate * gradient[0],
            scores[1] - learning_rate * gradient[1],
        ];

        assert!(updated[0] - updated[1] > scores[0] - scores[1]);
        assert!(
            ranking_hinge_objective(&updated, &relevance, params.c)
                < ranking_hinge_objective(&scores, &relevance, params.c)
        );
    }

    #[test]
    fn test_error_handling() {
        let params = RankingSVMParams::default();

        let result = compute_ranking_svm_gradients(&[], &[], params);
        assert!(matches!(result, Err(GradientError::EmptyInput)));

        let result = compute_ranking_svm_gradients(&[1.0, 2.0], &[1.0], params);
        assert!(matches!(result, Err(GradientError::LengthMismatch { .. })));
    }
}
