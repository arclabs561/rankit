//! Batch evaluation utilities for processing multiple queries.

use crate::eval::binary::*;
use crate::eval::trec::{Qrel, TrecRun};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use thiserror::Error;

/// Results for a single query evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResults {
    /// Query identifier.
    pub query_id: String,
    /// Metric name -> value.
    pub metrics: HashMap<String, f64>,
}

/// Batch evaluation results across multiple queries.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchResults {
    /// Per-query results.
    pub query_results: Vec<QueryResults>,
    /// Metric name -> mean value across queries.
    pub aggregated: HashMap<String, f64>,
}

/// Evaluation results for one TREC system.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecSystemResults {
    /// Run tag identifying the system.
    pub run_tag: String,
    /// Per-query and aggregate metric results for the system.
    pub results: BatchResults,
}

/// Errors raised while selecting or validating TREC systems.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TrecBatchError {
    /// The requested run tag does not occur in the input.
    #[error("TREC run tag '{0}' was not found")]
    RunTagNotFound(String),
    /// A system retrieved the same document more than once for a query.
    #[error("duplicate document '{doc_id}' for query '{query_id}' in TREC run '{run_tag}'")]
    DuplicateDocument {
        /// Run tag identifying the system.
        run_tag: String,
        /// Query containing the duplicate.
        query_id: String,
        /// Repeated document identifier.
        doc_id: String,
    },
}

/// Evaluate a batch of rankings using binary relevance metrics.
pub fn evaluate_batch_binary<I: Eq + std::hash::Hash + Clone>(
    rankings: &[Vec<I>],
    qrels: &[HashSet<I>],
    metrics: &[&str],
) -> BatchResults {
    assert_eq!(
        rankings.len(),
        qrels.len(),
        "rankings and qrels must have same length"
    );

    let mut query_results = Vec::new();
    let mut metric_sums: HashMap<String, f64> = HashMap::new();
    let mut metric_counts: HashMap<String, usize> = HashMap::new();

    for (i, (ranked, relevant)) in rankings.iter().zip(qrels.iter()).enumerate() {
        let mut query_metrics = HashMap::new();

        for metric_name in metrics {
            let value = match *metric_name {
                "ndcg@10" => ndcg_at_k(ranked, relevant, 10),
                "ndcg@5" => ndcg_at_k(ranked, relevant, 5),
                "precision@10" => precision_at_k(ranked, relevant, 10),
                "precision@5" => precision_at_k(ranked, relevant, 5),
                "precision@1" => precision_at_k(ranked, relevant, 1),
                "recall@10" => recall_at_k(ranked, relevant, 10),
                "recall@5" => recall_at_k(ranked, relevant, 5),
                "mrr" => mrr(ranked, relevant),
                "ap" | "map" => average_precision(ranked, relevant),
                "err@10" => err_at_k(ranked, relevant, 10),
                "rbp@10" => rbp_at_k(ranked, relevant, 10, 0.95),
                "f1@10" => f_measure_at_k(ranked, relevant, 10, 1.0),
                "success@10" => success_at_k(ranked, relevant, 10),
                "r_precision" => r_precision(ranked, relevant),
                _ => {
                    eprintln!("Unknown metric: {}", metric_name);
                    continue;
                }
            };

            query_metrics.insert(metric_name.to_string(), value);
            *metric_sums.entry(metric_name.to_string()).or_insert(0.0) += value;
            *metric_counts.entry(metric_name.to_string()).or_insert(0) += 1;
        }

        query_results.push(QueryResults {
            query_id: format!("query_{}", i),
            metrics: query_metrics,
        });
    }

    let aggregated: HashMap<String, f64> = metric_sums
        .into_iter()
        .map(|(name, sum)| {
            let count = metric_counts.get(&name).copied().unwrap_or(1);
            (name, sum / count as f64)
        })
        .collect();

    BatchResults {
        query_results,
        aggregated,
    }
}

/// Evaluate a single-system TREC run and qrels in batch.
///
/// This compatibility wrapper requires all entries to have the same run tag.
/// Use [`evaluate_trec_system`] to select a system from a multi-system input or
/// [`evaluate_all_trec_systems`] to evaluate every system.
///
/// # Panics
///
/// Panics if the input contains multiple run tags or duplicate documents for a
/// query. Multi-system input cannot be represented by [`BatchResults`].
pub fn evaluate_trec_batch(runs: &[TrecRun], qrels: &[Qrel], metrics: &[&str]) -> BatchResults {
    let run_tags: BTreeSet<&str> = runs.iter().map(|run| run.run_tag.as_str()).collect();
    assert!(
        run_tags.len() <= 1,
        "evaluate_trec_batch requires one run tag; use evaluate_trec_system or evaluate_all_trec_systems for multi-system input"
    );

    match run_tags.first() {
        Some(run_tag) => evaluate_trec_system(runs, qrels, metrics, run_tag)
            .expect("single-system TREC input must be valid"),
        None => BatchResults {
            query_results: Vec::new(),
            aggregated: HashMap::new(),
        },
    }
}

/// Evaluate the system identified by `run_tag`.
///
/// Queries are emitted in ascending query-ID order. As in `trec_eval`'s
/// default mode, only queries occurring in both the selected run and qrels are
/// evaluated. Score ties are broken by ascending document ID, so input order
/// does not affect the result.
pub fn evaluate_trec_system(
    runs: &[TrecRun],
    qrels: &[Qrel],
    metrics: &[&str],
    run_tag: &str,
) -> Result<BatchResults, TrecBatchError> {
    use crate::eval::trec::group_qrels_by_query;

    if !runs.iter().any(|run| run.run_tag == run_tag) {
        return Err(TrecBatchError::RunTagNotFound(run_tag.to_string()));
    }

    let qrels_by_query = group_qrels_by_query(qrels);
    let mut runs_by_query: BTreeMap<&str, Vec<&TrecRun>> = BTreeMap::new();

    for run in runs.iter().filter(|run| run.run_tag == run_tag) {
        let query_runs = runs_by_query.entry(run.query_id.as_str()).or_default();
        if query_runs.iter().any(|seen| seen.doc_id == run.doc_id) {
            return Err(TrecBatchError::DuplicateDocument {
                run_tag: run_tag.to_string(),
                query_id: run.query_id.clone(),
                doc_id: run.doc_id.clone(),
            });
        }
        query_runs.push(run);
    }

    let mut query_results = Vec::new();
    let mut metric_sums: HashMap<String, f64> = HashMap::new();
    let mut metric_counts: HashMap<String, usize> = HashMap::new();

    for (query_id, query_runs) in &mut runs_by_query {
        let Some(query_qrels) = qrels_by_query.get(*query_id) else {
            continue;
        };

        query_runs.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        let ranked_ids: Vec<&String> = query_runs.iter().map(|run| &run.doc_id).collect();

        let relevant: HashSet<_> = query_qrels
            .iter()
            .filter(|(_, &rel)| rel > 0)
            .map(|(id, _)| id)
            .collect();

        let mut query_metrics = HashMap::new();

        for metric_name in metrics {
            let value = match *metric_name {
                "ndcg@10" => ndcg_at_k(&ranked_ids, &relevant, 10),
                "ndcg@5" => ndcg_at_k(&ranked_ids, &relevant, 5),
                "precision@10" => precision_at_k(&ranked_ids, &relevant, 10),
                "precision@5" => precision_at_k(&ranked_ids, &relevant, 5),
                "precision@1" => precision_at_k(&ranked_ids, &relevant, 1),
                "recall@10" => recall_at_k(&ranked_ids, &relevant, 10),
                "recall@5" => recall_at_k(&ranked_ids, &relevant, 5),
                "mrr" => mrr(&ranked_ids, &relevant),
                "ap" | "map" => average_precision(&ranked_ids, &relevant),
                "err@10" => err_at_k(&ranked_ids, &relevant, 10),
                "rbp@10" => rbp_at_k(&ranked_ids, &relevant, 10, 0.95),
                "f1@10" => f_measure_at_k(&ranked_ids, &relevant, 10, 1.0),
                "success@10" => success_at_k(&ranked_ids, &relevant, 10),
                "r_precision" => r_precision(&ranked_ids, &relevant),
                _ => {
                    eprintln!("Unknown metric: {}", metric_name);
                    continue;
                }
            };

            query_metrics.insert(metric_name.to_string(), value);
            *metric_sums.entry(metric_name.to_string()).or_insert(0.0) += value;
            *metric_counts.entry(metric_name.to_string()).or_insert(0) += 1;
        }

        query_results.push(QueryResults {
            query_id: (*query_id).to_string(),
            metrics: query_metrics,
        });
    }

    let aggregated: HashMap<String, f64> = metric_sums
        .into_iter()
        .map(|(name, sum)| {
            let count = metric_counts.get(&name).copied().unwrap_or(1);
            (name, sum / count as f64)
        })
        .collect();

    Ok(BatchResults {
        query_results,
        aggregated,
    })
}

/// Evaluate every system in a possibly multi-system TREC input.
///
/// Each distinct run tag is evaluated exactly once and results are returned in
/// ascending run-tag order, independently of input order.
pub fn evaluate_all_trec_systems(
    runs: &[TrecRun],
    qrels: &[Qrel],
    metrics: &[&str],
) -> Result<Vec<TrecSystemResults>, TrecBatchError> {
    let run_tags: BTreeSet<&str> = runs.iter().map(|run| run.run_tag.as_str()).collect();

    run_tags
        .into_iter()
        .map(|run_tag| {
            evaluate_trec_system(runs, qrels, metrics, run_tag).map(|results| TrecSystemResults {
                run_tag: run_tag.to_string(),
                results,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trec_run(query_id: &str, doc_id: &str, score: f32, run_tag: &str) -> TrecRun {
        TrecRun {
            query_id: query_id.to_string(),
            doc_id: doc_id.to_string(),
            rank: 1,
            score,
            run_tag: run_tag.to_string(),
        }
    }

    fn qrel(query_id: &str, doc_id: &str, relevance: u32) -> Qrel {
        Qrel {
            query_id: query_id.to_string(),
            doc_id: doc_id.to_string(),
            relevance,
        }
    }

    #[test]
    fn test_evaluate_batch_binary() {
        let rankings = vec![vec!["doc1", "doc2", "doc3"], vec!["doc4", "doc5", "doc6"]];
        let qrels = vec![
            ["doc1", "doc3"].into_iter().collect::<HashSet<_>>(),
            ["doc4"].into_iter().collect::<HashSet<_>>(),
        ];

        let results = evaluate_batch_binary(&rankings, &qrels, &["ndcg@10", "precision@5"]);

        assert_eq!(results.query_results.len(), 2);
        assert!(results.aggregated.contains_key("ndcg@10"));
        assert!(results.aggregated.contains_key("precision@5"));
    }

    #[test]
    fn evaluates_all_systems_once_in_deterministic_order() {
        let runs = vec![
            trec_run("q2", "bad", 1.0, "zeta"),
            trec_run("q1", "good", 1.0, "alpha"),
            trec_run("q1", "bad", 0.0, "alpha"),
            trec_run("q1", "bad", 1.0, "zeta"),
            trec_run("q1", "good", 0.0, "zeta"),
            trec_run("q2", "good", 1.0, "alpha"),
        ];
        let qrels = vec![qrel("q2", "good", 1), qrel("q1", "good", 1)];

        let forward = evaluate_all_trec_systems(&runs, &qrels, &["precision@1"]).unwrap();
        let reverse = evaluate_all_trec_systems(
            &runs.iter().cloned().rev().collect::<Vec<_>>(),
            &qrels.iter().cloned().rev().collect::<Vec<_>>(),
            &["precision@1"],
        )
        .unwrap();

        assert_eq!(forward, reverse);
        assert_eq!(
            forward
                .iter()
                .map(|system| system.run_tag.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "zeta"]
        );
        assert_eq!(forward[0].results.aggregated["precision@1"], 1.0);
        assert_eq!(forward[1].results.aggregated["precision@1"], 0.0);
        assert_eq!(
            forward[0]
                .results
                .query_results
                .iter()
                .map(|query| query.query_id.as_str())
                .collect::<Vec<_>>(),
            ["q1", "q2"]
        );
    }

    #[test]
    fn explicit_system_selection_does_not_mix_run_tags() {
        let runs = vec![
            trec_run("q1", "good", 1.0, "good-system"),
            trec_run("q1", "bad", 1.0, "bad-system"),
        ];
        let qrels = vec![qrel("q1", "good", 1)];

        let selected =
            evaluate_trec_system(&runs, &qrels, &["precision@1"], "good-system").unwrap();

        assert_eq!(selected.aggregated["precision@1"], 1.0);
        assert_eq!(
            evaluate_trec_system(&runs, &qrels, &["precision@1"], "missing"),
            Err(TrecBatchError::RunTagNotFound("missing".to_string()))
        );
    }

    #[test]
    fn ignores_run_queries_without_qrels_like_default_trec_eval() {
        let runs = vec![
            trec_run("judged", "good", 1.0, "system"),
            trec_run("unjudged", "bad", 1.0, "system"),
        ];
        let qrels = vec![qrel("judged", "good", 1), qrel("not-retrieved", "good", 1)];

        let results = evaluate_trec_system(&runs, &qrels, &["precision@1"], "system").unwrap();

        assert_eq!(results.query_results.len(), 1);
        assert_eq!(results.query_results[0].query_id, "judged");
        assert_eq!(results.aggregated["precision@1"], 1.0);
    }

    #[test]
    fn rejects_duplicate_documents_within_a_system_query() {
        let runs = vec![
            trec_run("q1", "same", 1.0, "system"),
            trec_run("q1", "same", 0.5, "system"),
        ];

        assert_eq!(
            evaluate_trec_system(&runs, &[qrel("q1", "same", 1)], &["mrr"], "system"),
            Err(TrecBatchError::DuplicateDocument {
                run_tag: "system".to_string(),
                query_id: "q1".to_string(),
                doc_id: "same".to_string(),
            })
        );
    }

    #[test]
    #[should_panic(expected = "evaluate_trec_batch requires one run tag")]
    fn compatibility_wrapper_rejects_multi_system_input() {
        let runs = vec![
            trec_run("q1", "a", 1.0, "alpha"),
            trec_run("q1", "b", 1.0, "beta"),
        ];

        let _ = evaluate_trec_batch(&runs, &[qrel("q1", "a", 1)], &["mrr"]);
    }
}
