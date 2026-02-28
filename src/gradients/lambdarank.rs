//! LambdaRank gradient computation.
//!
//! LambdaRank optimizes ranking metrics (like NDCG) directly by computing gradients
//! based on how swapping document pairs would change the metric.
//!
//! For a pair (i, j) where document i should rank higher than j:
//! ```text
//! lambda_ij = -sigma / (1 + exp(sigma * (s_i - s_j))) * |delta_NDCG| * tau * mu
//! ```

use crate::gradients::error::GradientError;

/// LambdaRank parameters.
#[derive(Debug, Clone, Copy)]
pub struct LambdaRankParams {
    /// Sigmoid parameter. Default: 1.0
    pub sigma: f32,
    /// Enable query normalization (Cao et al. 2006). Default: true
    pub query_normalization: bool,
    /// Enable cost sensitivity (position-based importance). Default: true
    pub cost_sensitivity: bool,
    /// Enable score normalization (LightGBM-style). Default: false
    pub score_normalization: bool,
    /// Enable exponential gain for NDCG (2^rel - 1). Default: true
    pub exponential_gain: bool,
}

impl Default for LambdaRankParams {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            query_normalization: true,
            cost_sensitivity: true,
            score_normalization: false,
            exponential_gain: true,
        }
    }
}

/// Compute NDCG at a given position.
///
/// # Errors
///
/// Returns `GradientError::EmptyInput` if relevance is empty.
/// Returns `GradientError::InvalidNDCG` if k > relevance length.
pub fn ndcg_at_k(
    relevance: &[f32],
    k: Option<usize>,
    exponential_gain: bool,
) -> Result<f32, GradientError> {
    if relevance.is_empty() {
        return Err(GradientError::EmptyInput);
    }

    let k = k.unwrap_or(relevance.len());

    if k == 0 {
        return Ok(0.0);
    }

    if k > relevance.len() {
        return Err(GradientError::InvalidNDCG {
            k,
            length: relevance.len(),
        });
    }

    let k = k.min(relevance.len());

    let mut dcg = 0.0;
    for i in 0..k {
        let gain = if exponential_gain {
            (2.0_f32).powf(relevance[i]) - 1.0
        } else {
            relevance[i]
        };
        let discount = 1.0 / ((i + 2) as f32).log2();
        dcg += gain * discount;
    }

    let mut ideal_relevance = relevance.to_vec();
    ideal_relevance.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut idcg = 0.0;
    for i in 0..k {
        let gain = if exponential_gain {
            (2.0_f32).powf(ideal_relevance[i]) - 1.0
        } else {
            ideal_relevance[i]
        };
        let discount = 1.0 / ((i + 2) as f32).log2();
        idcg += gain * discount;
    }

    if idcg == 0.0 {
        Ok(0.0)
    } else {
        Ok(dcg / idcg)
    }
}

/// Compute change in NDCG if two documents are swapped.
fn delta_ndcg(
    relevance: &[f32],
    pos_i: usize,
    pos_j: usize,
    k: Option<usize>,
    exponential_gain: bool,
    inv_idcg: Option<f32>,
) -> f32 {
    if pos_i >= relevance.len() || pos_j >= relevance.len() {
        return 0.0;
    }

    let k = k.unwrap_or(relevance.len());

    if pos_i >= k && pos_j >= k {
        return 0.0;
    }

    let gain_i = if exponential_gain {
        (2.0_f32).powf(relevance[pos_i]) - 1.0
    } else {
        relevance[pos_i]
    };
    let gain_j = if exponential_gain {
        (2.0_f32).powf(relevance[pos_j]) - 1.0
    } else {
        relevance[pos_j]
    };

    let discount_i = if pos_i < k {
        1.0 / ((pos_i + 2) as f32).log2()
    } else {
        0.0
    };
    let discount_j = if pos_j < k {
        1.0 / ((pos_j + 2) as f32).log2()
    } else {
        0.0
    };

    let gain_diff = gain_i - gain_j;
    let discount_diff = discount_i - discount_j;

    let inv_idcg_val = if let Some(idcg) = inv_idcg {
        idcg
    } else {
        let mut ideal_relevance = relevance.to_vec();
        ideal_relevance
            .sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut idcg = 0.0;
        for i in 0..k.min(ideal_relevance.len()) {
            let gain = if exponential_gain {
                (2.0_f32).powf(ideal_relevance[i]) - 1.0
            } else {
                ideal_relevance[i]
            };
            let discount = 1.0 / ((i + 2) as f32).log2();
            idcg += gain * discount;
        }
        if idcg > 0.0 {
            1.0 / idcg
        } else {
            0.0
        }
    };

    -(gain_diff * discount_diff * inv_idcg_val)
}

/// Compute LambdaRank gradients for a ranked list.
///
/// # Errors
///
/// Returns `GradientError::EmptyInput` if inputs are empty.
/// Returns `GradientError::LengthMismatch` if scores and relevance differ in length.
pub fn compute_lambdarank_gradients(
    scores: &[f32],
    relevance: &[f32],
    params: LambdaRankParams,
    k: Option<usize>,
) -> Result<Vec<f32>, GradientError> {
    if scores.is_empty() || relevance.is_empty() {
        return Err(GradientError::EmptyInput);
    }

    if scores.len() != relevance.len() {
        return Err(GradientError::LengthMismatch {
            scores_len: scores.len(),
            relevance_len: relevance.len(),
        });
    }

    let n = scores.len();
    let k_trunc = k.unwrap_or(n);

    let inv_idcg = {
        let mut ideal_relevance = relevance.to_vec();
        ideal_relevance
            .sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut idcg = 0.0;
        for i in 0..k_trunc.min(ideal_relevance.len()) {
            let gain = if params.exponential_gain {
                (2.0_f32).powf(ideal_relevance[i]) - 1.0
            } else {
                ideal_relevance[i]
            };
            let discount = 1.0 / ((i + 2) as f32).log2();
            idcg += gain * discount;
        }
        if idcg > 0.0 {
            1.0 / idcg
        } else {
            0.0
        }
    };

    let mut lambdas = vec![0.0; n];
    let mut sum_lambdas = 0.0;

    let (min_score, max_score) = if params.score_normalization && n > 0 {
        let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    } else {
        (0.0, 0.0)
    };
    let score_range = if params.score_normalization && max_score != min_score {
        max_score - min_score
    } else {
        1.0
    };

    let mut valid_pairs = 0;
    for i in 0..n.min(k_trunc) {
        for j in (i + 1)..n {
            if (relevance[i] - relevance[j]).abs() > 1e-10 {
                valid_pairs += 1;
            }
        }
    }

    let mu = if params.query_normalization && valid_pairs > 0 {
        1.0 / valid_pairs as f32
    } else {
        1.0
    };

    for i in 0..n.min(k_trunc) {
        for j in (i + 1)..n {
            let rel_diff = relevance[i] - relevance[j];
            if rel_diff.abs() < 1e-10 {
                continue;
            }

            let (high_idx, low_idx, high_rank, low_rank) = if rel_diff > 0.0 {
                (i, j, i, j)
            } else {
                (j, i, j, i)
            };

            let delta = delta_ndcg(
                relevance,
                high_rank,
                low_rank,
                k,
                params.exponential_gain,
                Some(inv_idcg),
            );

            let tau = if params.cost_sensitivity {
                let min_rank = high_rank.min(low_rank);
                1.0 / ((min_rank + 2) as f32).ln()
            } else {
                1.0
            };

            let score_diff = scores[high_idx] - scores[low_idx];

            let normalized_delta = if params.score_normalization {
                delta.abs() / (0.01 + score_diff.abs() / score_range.max(0.01))
            } else {
                delta.abs()
            };

            let lambda_ij = -params.sigma / (1.0 + (params.sigma * score_diff).exp())
                * normalized_delta
                * tau
                * mu;

            lambdas[high_idx] += lambda_ij;
            lambdas[low_idx] -= lambda_ij;

            sum_lambdas += 2.0 * lambda_ij.abs();
        }
    }

    if params.query_normalization && sum_lambdas > 0.0 {
        let norm_factor = (1.0 + sum_lambdas).log2() / sum_lambdas;
        for lambda in &mut lambdas {
            *lambda *= norm_factor;
        }
    }

    Ok(lambdas)
}

/// LambdaRank trainer.
pub struct LambdaRankTrainer {
    params: LambdaRankParams,
}

impl LambdaRankTrainer {
    /// Create a new LambdaRank trainer.
    pub fn new(params: LambdaRankParams) -> Self {
        Self { params }
    }

    /// Compute gradients for a query-document list.
    pub fn compute_gradients(
        &self,
        scores: &[f32],
        relevance: &[f32],
        k: Option<usize>,
    ) -> Result<Vec<f32>, GradientError> {
        compute_lambdarank_gradients(scores, relevance, self.params, k)
    }

    /// Compute gradients for a batch of queries with query normalization.
    pub fn compute_gradients_batch(
        &self,
        batch_scores: &[Vec<f32>],
        batch_relevance: &[Vec<f32>],
        k: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, GradientError> {
        if batch_scores.len() != batch_relevance.len() {
            return Err(GradientError::LengthMismatch {
                scores_len: batch_scores.len(),
                relevance_len: batch_relevance.len(),
            });
        }

        if batch_scores.is_empty() {
            return Err(GradientError::EmptyInput);
        }

        let mut pairs_per_query: Vec<usize> = Vec::with_capacity(batch_scores.len());
        for (scores, relevance) in batch_scores.iter().zip(batch_relevance.iter()) {
            if scores.len() != relevance.len() {
                return Err(GradientError::LengthMismatch {
                    scores_len: scores.len(),
                    relevance_len: relevance.len(),
                });
            }

            let mut pairs = 0;
            for i in 0..scores.len() {
                for j in (i + 1)..scores.len() {
                    if (relevance[i] - relevance[j]).abs() > 1e-10 {
                        pairs += 1;
                    }
                }
            }
            pairs_per_query.push(pairs);
        }

        let max_pairs = pairs_per_query.iter().max().copied().unwrap_or(1);

        let mut batch_lambdas = Vec::with_capacity(batch_scores.len());
        for (idx, (scores, relevance)) in
            batch_scores.iter().zip(batch_relevance.iter()).enumerate()
        {
            let mut lambdas = compute_lambdarank_gradients(scores, relevance, self.params, k)?;

            if self.params.query_normalization && max_pairs > 0 {
                let mu = pairs_per_query[idx] as f32 / max_pairs as f32;
                for lambda in &mut lambdas {
                    *lambda *= mu;
                }
            }

            batch_lambdas.push(lambdas);
        }

        Ok(batch_lambdas)
    }
}

impl Default for LambdaRankTrainer {
    fn default() -> Self {
        Self::new(LambdaRankParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndcg() {
        let relevance = vec![3.0, 2.0, 1.0];
        let ndcg = ndcg_at_k(&relevance, None, true).unwrap();
        assert!((ndcg - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lambda_rank() {
        let scores = vec![0.5, 0.8, 0.3];
        let relevance = vec![3.0, 1.0, 2.0];

        let trainer = LambdaRankTrainer::default();
        let lambdas = trainer
            .compute_gradients(&scores, &relevance, None)
            .unwrap();

        assert_eq!(lambdas.len(), 3);
        assert!(lambdas.iter().any(|&l| l != 0.0));
    }

    #[test]
    fn test_lambda_rank_with_optimizations() {
        let scores = vec![0.5, 0.8, 0.3];
        let relevance = vec![3.0, 1.0, 2.0];

        let params = LambdaRankParams {
            sigma: 1.0,
            query_normalization: true,
            cost_sensitivity: true,
            score_normalization: true,
            exponential_gain: true,
        };
        let trainer = LambdaRankTrainer::new(params);
        let lambdas = trainer
            .compute_gradients(&scores, &relevance, Some(10))
            .unwrap();

        assert_eq!(lambdas.len(), 3);
        assert!(lambdas.iter().any(|&l| l != 0.0));
    }
}
