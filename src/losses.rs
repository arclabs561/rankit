//! LTR loss functions and advanced ranking operations.
//!
//! - **RankNet**: Pairwise logistic loss (Burges et al., ICML 2005)
//! - **LambdaLoss**: NDCG-weighted pairwise loss (Burges, 2010)
//! - **ApproxNDCG**: Differentiable NDCG approximation (Qin & Liu, 2010)
//! - **ListNet**: Listwise cross-entropy loss (ICML 2007)
//! - **ListMLE**: Listwise maximum likelihood (ICML 2008)
//! - **SoftSort**: OT-based sorting relaxation (ICML 2020)

use crate::rank::sigmoid;

/// SoftSort using optimal transport (simplified).
///
/// From: "SoftSort: A Continuous Relaxation for the argsort Operator" (ICML 2020)
pub fn soft_rank_softsort(values: &[f64], regularization_strength: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut ranks = vec![0.0; n];
    let positions: Vec<f64> = (0..n).map(|i| i as f64).collect();

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
                let sig = sigmoid(diff * regularization_strength);

                let pos_diff = (positions[i] - positions[j]).abs();
                let weight = (-pos_diff / regularization_strength).exp();

                sum += sig * weight;
                valid_comparisons += 1;
            }
        }

        if valid_comparisons > 0 {
            ranks[i] = sum / valid_comparisons as f64 * (n - 1) as f64;
        } else {
            ranks[i] = 0.0;
        }
    }

    ranks
}

/// RankNet pairwise loss.
///
/// From: "Learning to Rank using Gradient Descent" (Burges et al., ICML 2005)
///
/// Loss = mean over pairs {i,j: y_i > y_j} of log(1 + exp(-(s_i - s_j)))
pub fn ranknet_loss(predictions: &[f64], relevance: &[f64]) -> f64 {
    let n = predictions.len();
    if n <= 1 || n != relevance.len() {
        return 0.0;
    }

    let mut loss = 0.0;
    let mut pair_count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if (relevance[i] - relevance[j]).abs() < 1e-10 {
                continue;
            }

            let (higher_idx, lower_idx) = if relevance[i] > relevance[j] {
                (i, j)
            } else {
                (j, i)
            };

            let diff = predictions[higher_idx] - predictions[lower_idx];
            loss += (1.0 + (-diff).exp()).ln();
            pair_count += 1;
        }
    }

    if pair_count > 0 {
        loss / pair_count as f64
    } else {
        0.0
    }
}

/// LambdaLoss: RankNet with NDCG-aware pair weighting.
///
/// From: "From RankNet to LambdaRank to LambdaMART" (Burges, 2010)
///
/// Loss = mean over pairs of |delta_NDCG| * log(1 + exp(-(s_high - s_low)))
pub fn lambda_loss(predictions: &[f64], relevance: &[f64], k: Option<usize>) -> f64 {
    let n = predictions.len();
    if n <= 1 || n != relevance.len() {
        return 0.0;
    }

    let k = k.unwrap_or(n);

    let idcg = compute_idcg(relevance, k);
    if idcg < 1e-10 {
        return 0.0;
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| predictions[b].partial_cmp(&predictions[a]).unwrap());

    let mut rank_of: Vec<usize> = vec![0; n];
    for (rank, &idx) in indices.iter().enumerate() {
        rank_of[idx] = rank;
    }

    let mut loss = 0.0;
    let mut pair_count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if (relevance[i] - relevance[j]).abs() < 1e-10 {
                continue;
            }

            let (higher_idx, lower_idx) = if relevance[i] > relevance[j] {
                (i, j)
            } else {
                (j, i)
            };

            let delta_ndcg = compute_delta_ndcg(
                relevance[higher_idx],
                relevance[lower_idx],
                rank_of[higher_idx],
                rank_of[lower_idx],
                idcg,
                k,
            );

            let diff = predictions[higher_idx] - predictions[lower_idx];
            loss += delta_ndcg.abs() * (1.0 + (-diff).exp()).ln();
            pair_count += 1;
        }
    }

    if pair_count > 0 {
        loss / pair_count as f64
    } else {
        0.0
    }
}

fn compute_idcg(relevance: &[f64], k: usize) -> f64 {
    let mut sorted_rel: Vec<f64> = relevance.to_vec();
    sorted_rel.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

    let mut idcg = 0.0;
    for (rank, &rel) in sorted_rel.iter().enumerate().take(k) {
        if rel > 0.0 {
            idcg += (2.0_f64.powf(rel) - 1.0) / (rank as f64 + 2.0).log2();
        }
    }
    idcg
}

fn compute_delta_ndcg(
    rel_i: f64,
    rel_j: f64,
    rank_i: usize,
    rank_j: usize,
    idcg: f64,
    k: usize,
) -> f64 {
    if rank_i >= k && rank_j >= k {
        return 0.0;
    }

    let discount_i = 1.0 / (rank_i as f64 + 2.0).log2();
    let discount_j = 1.0 / (rank_j as f64 + 2.0).log2();

    let gain_i = 2.0_f64.powf(rel_i) - 1.0;
    let gain_j = 2.0_f64.powf(rel_j) - 1.0;

    let current = gain_i * discount_i + gain_j * discount_j;
    let swapped = gain_i * discount_j + gain_j * discount_i;

    (swapped - current) / idcg
}

/// ApproxNDCG: differentiable approximation of NDCG.
///
/// From: Qin & Liu (2010). Returns approximate NDCG in [0, 1] (higher is better).
pub fn approx_ndcg(
    predictions: &[f64],
    relevance: &[f64],
    regularization_strength: f64,
    k: Option<usize>,
) -> f64 {
    let n = predictions.len();
    if n == 0 || n != relevance.len() {
        return 0.0;
    }

    let k = k.unwrap_or(n).min(n);

    let idcg = compute_idcg(relevance, k);
    if idcg < 1e-10 {
        return 1.0;
    }

    let soft_ranks = crate::rank::soft_rank(predictions, regularization_strength);

    let mut approx_dcg = 0.0;
    for i in 0..n {
        if relevance[i] <= 0.0 {
            continue;
        }

        let gain = 2.0_f64.powf(relevance[i]) - 1.0;
        let soft_rank_normalized = soft_ranks[i] * (n as f64 - 1.0);
        let soft_discount = 1.0 / (soft_rank_normalized + 2.0).log2();

        approx_dcg += gain * soft_discount;
    }

    (approx_dcg / idcg).min(1.0)
}

/// ApproxNDCG loss (1 - ApproxNDCG). Lower is better.
pub fn approx_ndcg_loss(
    predictions: &[f64],
    relevance: &[f64],
    regularization_strength: f64,
    k: Option<usize>,
) -> f64 {
    1.0 - approx_ndcg(predictions, relevance, regularization_strength, k)
}

/// ListNet-style listwise ranking loss.
///
/// From: "Learning to Rank: From Pairwise Approach to Listwise Approach" (ICML 2007)
pub fn listnet_loss(predictions: &[f64], targets: &[f64], regularization_strength: f64) -> f64 {
    let n = predictions.len();

    if n == 0 || n != targets.len() {
        return f64::INFINITY;
    }

    let pred_ranks = crate::rank::soft_rank(predictions, regularization_strength);
    let target_ranks = crate::rank::soft_rank(targets, regularization_strength);

    let pred_probs = softmax_from_ranks(&pred_ranks);
    let target_probs = softmax_from_ranks(&target_ranks);

    let mut loss = 0.0;
    for i in 0..n {
        if target_probs[i] > 1e-10 {
            loss -= target_probs[i] * pred_probs[i].ln();
        }
    }

    loss
}

fn softmax_from_ranks(ranks: &[f64]) -> Vec<f64> {
    let n = ranks.len();
    if n == 0 {
        return vec![];
    }

    let max_rank = ranks.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f64 = ranks.iter().map(|&r| (-(r - max_rank)).exp()).sum();

    ranks
        .iter()
        .map(|&r| (-(r - max_rank)).exp() / exp_sum)
        .collect()
}

/// ListMLE-style maximum likelihood estimation for ranking.
///
/// From: "Listwise Approach to Learning to Rank: Theory and Algorithm" (ICML 2008)
pub fn listmle_loss(predictions: &[f64], targets: &[f64], regularization_strength: f64) -> f64 {
    let n = predictions.len();

    if n == 0 || n != targets.len() {
        return f64::INFINITY;
    }

    let mut target_indices: Vec<usize> = (0..n).collect();
    target_indices.sort_unstable_by(|&a, &b| targets[b].partial_cmp(&targets[a]).unwrap());

    let pred_ranks = crate::rank::soft_rank(predictions, regularization_strength);

    let mut loss = 0.0;

    for i in 0..n {
        let idx = target_indices[i];
        let score = pred_ranks[idx];

        let mut denom = 0.0;
        for &jdx in target_indices.iter().skip(i) {
            denom += pred_ranks[jdx].exp();
        }

        if denom > 1e-10 {
            loss -= score - denom.ln();
        }
    }

    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ranknet_loss() {
        let predictions = vec![0.8, 0.3, 0.6];
        let relevance = vec![2.0, 0.0, 1.0];
        let loss = ranknet_loss(&predictions, &relevance);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_lambda_loss() {
        let predictions = vec![0.8, 0.3, 0.6];
        let relevance = vec![2.0, 0.0, 1.0];
        let loss = lambda_loss(&predictions, &relevance, None);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_listnet_loss() {
        let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];

        let loss = listnet_loss(&predictions, &targets, 1.0);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_listmle_loss() {
        let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];

        let loss = listmle_loss(&predictions, &targets, 1.0);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_approx_ndcg() {
        let predictions = vec![0.8, 0.3, 0.6];
        let relevance = vec![2.0, 0.0, 1.0];
        let ndcg = approx_ndcg(&predictions, &relevance, 1.0, None);
        assert!((0.0..=1.0).contains(&ndcg));
    }
}
