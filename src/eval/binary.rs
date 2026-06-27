//! Binary relevance IR evaluation metrics.
//!
//! All metrics assume:
//! - `ranked`: List of document IDs in ranked order (best first)
//! - `relevant`: Set of relevant document IDs (ground truth)

use std::collections::HashSet;

/// Measures information flow quality in a retrieval pipeline.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DegradationMetrics {
    /// Noise estimate [0, 1]
    pub noise: f64,
    /// Loss estimate [0, 1]
    pub loss: f64,
    /// Waste estimate [0, 1]
    pub waste: f64,
}

impl DegradationMetrics {
    /// Compute metrics from Fisher Information, synthesis entropy, and coherence.
    pub fn compute(fisher_info: f64, synthesis_entropy: f64, avg_coherence: f64) -> Self {
        Self {
            noise: (1.0 - fisher_info).clamp(0.0, 1.0),
            loss: synthesis_entropy.clamp(0.0, 1.0),
            waste: (1.0 - avg_coherence).clamp(0.0, 1.0),
        }
    }
}

fn extract_ranks<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> Vec<usize> {
    ranked
        .iter()
        .enumerate()
        .filter(|(_, id)| relevant.contains(id))
        .map(|(i, _)| i + 1)
        .collect()
}

/// Precision at k: fraction of top-k that are relevant.
pub fn precision_at_k<I: Eq + std::hash::Hash>(
    ranked: &[I],
    relevant: &HashSet<I>,
    k: usize,
) -> f64 {
    let ranks = extract_ranks(ranked, relevant);
    rankops::metrics::precision_at_k(&ranks, k)
}

/// Recall at k: fraction of relevant docs in top-k.
pub fn recall_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    let ranks = extract_ranks(ranked, relevant);
    rankops::metrics::recall_at_k(&ranks, relevant.len(), k)
}

/// Mean Reciprocal Rank: 1 / rank of first relevant document.
pub fn mrr<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64 {
    // `extract_ranks` returns the ranks of every relevant document in ascending
    // order, so the first entry is the first relevant rank. Only that rank
    // defines the reciprocal rank for a single ranking. Passing all of them to
    // `rankops::metrics::mrr` would average 1/rank over every relevant document,
    // which is not MRR (it under-reports whenever there is more than one
    // relevant document).
    let ranks = extract_ranks(ranked, relevant);
    ranks
        .first()
        .map_or(0.0, |&first| rankops::metrics::mrr(&[first]))
}

/// Discounted Cumulative Gain at k.
pub fn dcg_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    let relevance: Vec<f64> = ranked
        .iter()
        .take(k)
        .map(|id| if relevant.contains(id) { 1.0 } else { 0.0 })
        .collect();
    rankops::metrics::dcg(&relevance)
}

/// Ideal DCG at k.
pub fn idcg_at_k(n_relevant: usize, k: usize) -> f64 {
    let ideal_relevance: Vec<f64> = (0..k)
        .map(|i| if i < n_relevant { 1.0 } else { 0.0 })
        .collect();
    rankops::metrics::dcg(&ideal_relevance)
}

/// Normalized DCG at k.
pub fn ndcg_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    let relevance: Vec<f64> = ranked
        .iter()
        .take(k)
        .map(|id| if relevant.contains(id) { 1.0 } else { 0.0 })
        .collect();
    let ideal_relevance: Vec<f64> = (0..k)
        .map(|i| if i < relevant.len() { 1.0 } else { 0.0 })
        .collect();
    rankops::metrics::ndcg(&relevance, &ideal_relevance)
}

/// Average Precision: average of precision at each relevant doc.
pub fn average_precision<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64 {
    let ranks = extract_ranks(ranked, relevant);
    rankops::metrics::average_precision(&ranks, relevant.len())
}

/// Expected Reciprocal Rank (ERR).
pub fn err_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    let ranks = extract_ranks(ranked, relevant);
    rankops::metrics::err_at_k(&ranks, k)
}

/// Rank-Biased Precision (RBP).
pub fn rbp_at_k<I: Eq + std::hash::Hash>(
    ranked: &[I],
    relevant: &HashSet<I>,
    k: usize,
    persistence: f64,
) -> f64 {
    let ranks = extract_ranks(ranked, relevant);
    rankops::metrics::rbp_at_k(&ranks, k, persistence)
}

/// F-measure at k: harmonic mean of precision and recall.
pub fn f_measure_at_k<I: Eq + std::hash::Hash>(
    ranked: &[I],
    relevant: &HashSet<I>,
    k: usize,
    beta: f64,
) -> f64 {
    let ranks = extract_ranks(ranked, relevant);
    rankops::metrics::f_measure_at_k(&ranks, relevant.len(), k, beta)
}

/// Success at k: whether at least one relevant document is in top-k.
pub fn success_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    // Binary: 1.0 if any relevant document falls in the top k, else 0.0.
    // hits_at_k(&ranks, k) would instead return (relevant in top-k) / (total
    // relevant) = recall@k, a fraction, which is not success@k.
    let ranks = extract_ranks(ranked, relevant);
    if ranks.iter().any(|&r| r > 0 && r <= k) {
        1.0
    } else {
        0.0
    }
}

/// R-Precision: Precision at R, where R is the number of relevant documents.
pub fn r_precision<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64 {
    let ranks = extract_ranks(ranked, relevant);
    rankops::metrics::r_precision(&ranks, relevant.len())
}

/// All metrics for a single ranking (binary relevance).
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    /// Precision at rank 1.
    pub precision_at_1: f64,
    /// Precision at rank 5.
    pub precision_at_5: f64,
    /// Precision at rank 10.
    pub precision_at_10: f64,
    /// Recall at rank 5.
    pub recall_at_5: f64,
    /// Recall at rank 10.
    pub recall_at_10: f64,
    /// Mean reciprocal rank.
    pub mrr: f64,
    /// Normalized discounted cumulative gain at rank 5.
    pub ndcg_at_5: f64,
    /// Normalized discounted cumulative gain at rank 10.
    pub ndcg_at_10: f64,
    /// Average precision over all relevant documents.
    pub average_precision: f64,
    /// Expected reciprocal rank at rank 10.
    pub err_at_10: f64,
    /// Rank-biased precision at rank 10.
    pub rbp_at_10: f64,
    /// F1 score at rank 10.
    pub f1_at_10: f64,
    /// Whether any relevant document appears in the top 10.
    pub success_at_10: f64,
    /// Precision at rank R, where R is the number of relevant documents.
    pub r_precision: f64,
}

#[cfg(feature = "serde")]
impl Metrics {
    /// Compute all metrics for a ranking.
    pub fn compute<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> Self {
        Self {
            precision_at_1: precision_at_k(ranked, relevant, 1),
            precision_at_5: precision_at_k(ranked, relevant, 5),
            precision_at_10: precision_at_k(ranked, relevant, 10),
            recall_at_5: recall_at_k(ranked, relevant, 5),
            recall_at_10: recall_at_k(ranked, relevant, 10),
            mrr: mrr(ranked, relevant),
            ndcg_at_5: ndcg_at_k(ranked, relevant, 5),
            ndcg_at_10: ndcg_at_k(ranked, relevant, 10),
            average_precision: average_precision(ranked, relevant),
            err_at_10: err_at_k(ranked, relevant, 10),
            rbp_at_10: rbp_at_k(ranked, relevant, 10, 0.95),
            f1_at_10: f_measure_at_k(ranked, relevant, 10, 1.0),
            success_at_10: success_at_k(ranked, relevant, 10),
            r_precision: r_precision(ranked, relevant),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_at_k() {
        let ranked = vec!["a", "b", "c", "d", "e"];
        let relevant: HashSet<_> = ["a", "c", "e"].into_iter().collect();

        assert!((precision_at_k(&ranked, &relevant, 1) - 1.0).abs() < 1e-9);
        assert!((precision_at_k(&ranked, &relevant, 2) - 0.5).abs() < 1e-9);
        assert!((precision_at_k(&ranked, &relevant, 5) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_mrr() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<_> = ["b"].into_iter().collect();

        assert!((mrr(&ranked, &relevant) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mrr_multiple_relevant_uses_first() {
        // Relevant docs at ranks 2 and 4. MRR is the reciprocal of the FIRST
        // relevant rank (1/2), not the mean of 1/2 and 1/4 (= 0.375).
        let ranked = vec!["a", "b", "c", "d"];
        let relevant: HashSet<_> = ["b", "d"].into_iter().collect();
        assert!((mrr(&ranked, &relevant) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg() {
        let ranked = vec!["a", "b", "c", "d"];
        let relevant: HashSet<_> = ["a", "c"].into_iter().collect();

        let dcg = 1.0 / 2.0_f64.log2() + 1.0 / 4.0_f64.log2();
        let idcg = 1.0 / 2.0_f64.log2() + 1.0 / 3.0_f64.log2();

        assert!((ndcg_at_k(&ranked, &relevant, 4) - dcg / idcg).abs() < 1e-9);
    }

    #[test]
    fn test_average_precision() {
        let ranked = vec!["a", "b", "c", "d"];
        let relevant: HashSet<_> = ["a", "c"].into_iter().collect();

        let ap = average_precision(&ranked, &relevant);
        assert!(ap > 0.8 && ap < 0.85);
    }

    #[test]
    fn test_success_at_k_is_binary() {
        // Relevant docs at ranks 2 and 7. success@3 must be 1.0 (rank 2 <= 3),
        // not the fraction of relevant docs in the top 3 (which would be 0.5).
        let ranked = vec!["a", "b", "c", "d", "e", "f", "g"];
        let relevant: HashSet<_> = ["b", "g"].into_iter().collect();
        assert!((success_at_k(&ranked, &relevant, 3) - 1.0).abs() < 1e-9);
        // And 0.0 when no relevant doc is in the top-k.
        assert!(success_at_k(&ranked, &relevant, 1).abs() < 1e-9);
    }
}
