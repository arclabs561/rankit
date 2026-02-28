//! Graded relevance IR evaluation metrics.
//!
//! These metrics use actual relevance scores (not just binary) in calculations.

use std::collections::HashMap;

/// Compute nDCG@k for graded relevance.
///
/// Uses actual relevance grades (0, 1, 2, ...) in the gain calculation.
///
/// Reference: Jarvelin & Kekalainen (2002)
pub fn compute_ndcg(ranked: &[(String, f32)], qrels: &HashMap<String, u32>, k: usize) -> f64 {
    let relevance: Vec<f64> = ranked
        .iter()
        .take(k)
        .map(|(doc_id, _)| qrels.get(doc_id).copied().unwrap_or(0) as f64)
        .collect();

    let mut ideal_gains: Vec<f64> = qrels
        .values()
        .copied()
        .filter(|&r| r > 0)
        .map(|r| r as f64)
        .collect();
    ideal_gains.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    let ideal_k: Vec<f64> = ideal_gains.into_iter().take(k).collect();

    fynch::metrics::ndcg(&relevance, &ideal_k)
}

/// Compute Mean Average Precision (MAP) for graded relevance.
///
/// Treats any relevance > 0 as relevant (binary conversion for MAP).
pub fn compute_map(ranked: &[(String, f32)], qrels: &HashMap<String, u32>) -> f64 {
    let n_relevant = qrels.values().filter(|&&rel| rel > 0).count();
    if n_relevant == 0 {
        return 0.0;
    }

    let ranks: Vec<usize> = ranked
        .iter()
        .enumerate()
        .filter(|(_, (doc_id, _))| qrels.get(doc_id).copied().unwrap_or(0) > 0)
        .map(|(i, _)| i + 1)
        .collect();

    fynch::metrics::average_precision(&ranks, n_relevant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_ndcg_graded() {
        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];
        let mut qrels = HashMap::new();
        let _ = qrels.insert("doc1".to_string(), 2);
        let _ = qrels.insert("doc2".to_string(), 1);
        let _ = qrels.insert("doc3".to_string(), 0);

        let ndcg = compute_ndcg(&ranked, &qrels, 3);
        assert!(ndcg > 0.0 && ndcg <= 1.0);
        assert!(ndcg > 0.5);
    }

    #[test]
    fn test_compute_map_graded() {
        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];
        let mut qrels = HashMap::new();
        let _ = qrels.insert("doc1".to_string(), 2);
        let _ = qrels.insert("doc2".to_string(), 1);
        let _ = qrels.insert("doc3".to_string(), 0);

        let map = compute_map(&ranked, &qrels);
        assert!((map - 1.0).abs() < 1e-9);
    }
}
