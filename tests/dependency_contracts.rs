#![cfg(feature = "eval")]

use std::collections::HashSet;

use rankit::binary;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn public_spearman_reexport_and_batch_match_fynch() {
    let predictions = vec![vec![0.1, 0.9, 0.3], vec![3.0, 1.0, 2.0]];
    let targets = vec![vec![0.0, 1.0, 0.2], vec![2.0, 0.0, 1.0]];
    let strength = 0.75;

    let expected: Vec<_> = predictions
        .iter()
        .zip(&targets)
        .map(|(prediction, target)| fynch::loss::spearman_loss(prediction, target, strength))
        .collect();

    assert_close(
        rankit::spearman_loss(&predictions[0], &targets[0], strength),
        expected[0],
    );
    assert_eq!(
        rankit::spearman_loss_batch(&predictions, &targets, strength),
        expected
    );
}

#[test]
fn binary_metric_wrappers_preserve_rankops_rank_contracts() {
    let ranked = ["irrelevant", "first", "other", "second"];
    let relevant: HashSet<_> = ["first", "second"].into_iter().collect();
    let relevant_ranks = [2, 4];
    let relevance = [0.0, 1.0, 0.0, 1.0];
    let ideal_relevance = [1.0, 1.0, 0.0, 0.0];

    assert_close(
        binary::precision_at_k(&ranked, &relevant, 4),
        rankops::metrics::precision_at_k(&relevant_ranks, 4),
    );
    assert_close(
        binary::recall_at_k(&ranked, &relevant, 4),
        rankops::metrics::recall_at_k(&relevant_ranks, relevant.len(), 4),
    );
    assert_close(
        binary::dcg_at_k(&ranked, &relevant, 4),
        rankops::metrics::dcg(&relevance),
    );
    assert_close(
        binary::idcg_at_k(relevant.len(), 4),
        rankops::metrics::dcg(&ideal_relevance),
    );
    assert_close(
        binary::ndcg_at_k(&ranked, &relevant, 4),
        rankops::metrics::ndcg(&relevance, &ideal_relevance),
    );
    assert_close(
        binary::average_precision(&ranked, &relevant),
        rankops::metrics::average_precision(&relevant_ranks, relevant.len()),
    );
    assert_close(
        binary::err_at_k(&ranked, &relevant, 4),
        rankops::metrics::err_at_k(&relevant_ranks, 4),
    );
    assert_close(
        binary::rbp_at_k(&ranked, &relevant, 4, 0.8),
        rankops::metrics::rbp_at_k(&relevant_ranks, 4, 0.8),
    );
    assert_close(
        binary::f_measure_at_k(&ranked, &relevant, 4, 1.0),
        rankops::metrics::f_measure_at_k(&relevant_ranks, relevant.len(), 4, 1.0),
    );
    assert_close(
        binary::r_precision(&ranked, &relevant),
        rankops::metrics::r_precision(&relevant_ranks, relevant.len()),
    );
}
