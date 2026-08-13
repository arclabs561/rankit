//! LTR loss functions and advanced ranking operations.
//!
//! - **RankNet**: Pairwise logistic loss (Burges et al., ICML 2005)
//! - **NDCG-weighted pairwise**: RankNet-style terms weighted by swapped-pair delta NDCG
//! - **ApproxNDCG**: Differentiable NDCG approximation (Qin & Liu, 2010)
//! - **ListNet**: Cross-entropy over top-one softmax distributions
//! - **ListMLE-style**: Permutation likelihood computed from soft ranks
//! - **SoftSort-style**: A simplified sorting relaxation

use crate::rank::sigmoid;

fn softplus(value: f64) -> f64 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

fn logaddexp(left: f64, right: f64) -> f64 {
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

/// SoftSort-inspired ranking heuristic.
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
            loss += softplus(-diff);
            pair_count += 1;
        }
    }

    if pair_count > 0 {
        loss / pair_count as f64
    } else {
        0.0
    }
}

/// RankNet-style loss with NDCG-aware pair weighting.
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
            loss += delta_ndcg.abs() * softplus(-diff);
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
/// When `k` is set, a smooth cutoff excludes documents below the top-k boundary.
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
        let position = (n as f64 - 1.0) - soft_ranks[i];
        let soft_discount = 1.0 / (position + 2.0).log2();
        let membership = if k < n {
            sigmoid(regularization_strength * (k as f64 - 0.5 - position))
        } else {
            1.0
        };

        approx_dcg += membership * gain * soft_discount;
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

/// Top-one ListNet cross-entropy.
///
/// From: "Learning to Rank: From Pairwise Approach to Listwise Approach" (ICML 2007)
/// `regularization_strength` is an inverse temperature: larger values produce
/// sharper top-one distributions.
pub fn listnet_loss(predictions: &[f64], targets: &[f64], regularization_strength: f64) -> f64 {
    let n = predictions.len();

    if n == 0 || n != targets.len() {
        return f64::INFINITY;
    }

    let pred_logits: Vec<_> = predictions
        .iter()
        .map(|score| score * regularization_strength)
        .collect();
    let target_logits: Vec<_> = targets
        .iter()
        .map(|score| score * regularization_strength)
        .collect();
    let target_probs = softmax(&target_logits);
    let pred_log_normalizer = logsumexp(&pred_logits);

    target_probs
        .iter()
        .zip(pred_logits)
        .map(|(target_probability, prediction)| {
            target_probability * (pred_log_normalizer - prediction)
        })
        .sum()
}

fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let log_normalizer = logsumexp(values);
    values
        .iter()
        .map(|value| (value - log_normalizer).exp())
        .collect()
}

fn logsumexp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum
        + values
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
}

/// ListMLE-inspired likelihood objective computed from soft ranks.
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

    let ordered_scores: Vec<_> = target_indices
        .iter()
        .map(|&index| pred_ranks[index])
        .collect();
    let mut suffix_logsumexp = *ordered_scores.last().unwrap();
    let mut loss = 0.0;
    for &score in ordered_scores.iter().rev().skip(1) {
        suffix_logsumexp = logaddexp(score, suffix_logsumexp);
        loss += suffix_logsumexp - score;
    }

    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discrete_ndcg_at_k(predictions: &[f64], relevance: &[f64], k: usize) -> f64 {
        let mut indices: Vec<_> = (0..predictions.len()).collect();
        indices.sort_unstable_by(|&a, &b| predictions[b].partial_cmp(&predictions[a]).unwrap());

        let dcg: f64 = indices
            .iter()
            .take(k)
            .enumerate()
            .map(|(rank, &index)| {
                (2.0_f64.powf(relevance[index]) - 1.0) / (rank as f64 + 2.0).log2()
            })
            .sum();
        let idcg = compute_idcg(relevance, k);
        if idcg > 0.0 {
            dcg / idcg
        } else {
            1.0
        }
    }

    fn permutations(values: &mut [f64], start: usize, output: &mut Vec<Vec<f64>>) {
        if start == values.len() {
            output.push(values.to_vec());
            return;
        }

        for index in start..values.len() {
            values.swap(start, index);
            permutations(values, start + 1, output);
            values.swap(start, index);
        }
    }

    #[test]
    fn test_ranknet_loss() {
        let predictions = vec![0.8, 0.3, 0.6];
        let relevance = vec![2.0, 0.0, 1.0];
        let loss = ranknet_loss(&predictions, &relevance);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn ranknet_loss_is_stable_for_extreme_margins() {
        let loss = ranknet_loss(&[-1000.0, 1000.0], &[1.0, 0.0]);
        assert!(loss.is_finite());
        assert!((loss - 2000.0).abs() < 1e-12);
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
    fn lambda_loss_is_stable_for_extreme_margins() {
        let loss = lambda_loss(&[-1000.0, 1000.0], &[1.0, 0.0], None);
        let expected = 2000.0 * (1.0 - 1.0 / 3.0_f64.log2());
        assert!(loss.is_finite());
        assert!((loss - expected).abs() < 1e-12);
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
    fn listnet_matches_top_one_cross_entropy() {
        let predictions = [2.0_f64.ln(), 3.0_f64.ln()];
        let targets = [4.0_f64.ln(), 0.0];
        let expected = -0.8 * 0.4_f64.ln() - 0.2 * 0.6_f64.ln();

        let actual = listnet_loss(&predictions, &targets, 1.0);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn listnet_is_stable_for_extreme_logits() {
        let aligned = listnet_loss(&[1000.0, -1000.0], &[1000.0, -1000.0], 1.0);
        let reversed = listnet_loss(&[-1000.0, 1000.0], &[1000.0, -1000.0], 1.0);

        assert!(aligned.is_finite());
        assert!(reversed.is_finite());
        assert!(aligned < 1e-12);
        assert!((reversed - 2000.0).abs() < 1e-12);
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
    fn listmle_is_stable_for_long_lists() {
        let predictions: Vec<_> = (0..800).map(|value| value as f64).collect();
        let targets = predictions.clone();

        let loss = listmle_loss(&predictions, &targets, 100.0);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_approx_ndcg() {
        let predictions = vec![0.8, 0.3, 0.6];
        let relevance = vec![2.0, 0.0, 1.0];
        let ndcg = approx_ndcg(&predictions, &relevance, 1.0, None);
        assert!((0.0..=1.0).contains(&ndcg));
    }

    #[test]
    fn approx_ndcg_at_k_converges_to_discrete_oracle() {
        let relevance = [3.0, 2.0, 1.0, 0.0];
        let mut scores = [1.0, 2.0, 3.0, 4.0];
        let mut score_permutations = Vec::new();
        permutations(&mut scores, 0, &mut score_permutations);

        for predictions in score_permutations {
            for k in 1..=predictions.len() {
                let expected = discrete_ndcg_at_k(&predictions, &relevance, k);
                let actual = approx_ndcg(&predictions, &relevance, 100.0, Some(k));
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "predictions={predictions:?}, k={k}, expected={expected}, actual={actual}"
                );
            }
        }
    }

    #[test]
    fn approx_ndcg_at_one_does_not_credit_lower_ranks() {
        let predictions = [1.0, 3.0, 2.0];
        let relevance = [3.0, 2.0, 1.0];

        let actual = approx_ndcg(&predictions, &relevance, 100.0, Some(1));
        let expected = 3.0 / 7.0;
        assert!((actual - expected).abs() < 1e-10);
    }
}
