//! Rosetta correctness fixtures: rankit IR evaluation metrics asserted against
//! scikit-learn and numpy.
//!
//! Reference values in `fixtures/rosetta/rankit_ir.json` come from
//! `gen_rankit.py` (their provenance). rankit's nDCG uses LINEAR gain with a
//! log2(rank+1) discount, so the matching oracle is sklearn.metrics.ndcg_score
//! (linear gain), NOT pytrec_eval which uses 2^rel-1. Both rankit's binary and
//! graded nDCG paths use linear gain, so sklearn covers both. MAP / MRR /
//! precision@k / recall@k are unambiguous IR formulas with no scikit-learn
//! function, so their reference is the canonical formula computed in numpy
//! (cross-implementation check).
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_rankit.py`.

use rankit::{binary, graded};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

const FIXTURE: &str = include_str!("fixtures/rosetta/rankit_ir.json");

#[derive(Deserialize)]
struct Fixture {
    n: usize,
    relevant: Vec<usize>,
    grades: Vec<u32>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    ndcg_bin_5: f64,
    ndcg_bin_10: f64,
    ndcg_graded_5: f64,
    ndcg_graded_10: f64,
    map: f64,
    mrr: f64,
    precision_at_5: f64,
    recall_at_5: f64,
}

fn close(got: f64, want: f64, label: &str) {
    let tol = 1e-9 * (1.0 + want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label}: rankit={got} ref={want} diff={diff} tol={tol}"
    );
}

#[test]
fn rosetta_ir_metrics_match_sklearn_numpy() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let e = &fx.expected;

    // Documents are evaluated in index order; relevant is the binary ground truth.
    let ranked: Vec<usize> = (0..fx.n).collect();
    let relevant: HashSet<usize> = fx.relevant.iter().copied().collect();

    // nDCG (binary) vs sklearn.ndcg_score with binary y_true.
    close(
        binary::ndcg_at_k(&ranked, &relevant, 5),
        e.ndcg_bin_5,
        "ndcg_bin_5",
    );
    close(
        binary::ndcg_at_k(&ranked, &relevant, 10),
        e.ndcg_bin_10,
        "ndcg_bin_10",
    );

    // nDCG (graded) vs sklearn.ndcg_score with graded y_true. compute_ndcg orders
    // by the ranked list (the f32 score is unused), so pass docs in index order.
    let ranked_pairs: Vec<(String, f32)> = (0..fx.n).map(|i| (i.to_string(), 0.0)).collect();
    let qrels: HashMap<String, u32> = fx
        .grades
        .iter()
        .enumerate()
        .map(|(i, &g)| (i.to_string(), g))
        .collect();
    close(
        graded::compute_ndcg(&ranked_pairs, &qrels, 5),
        e.ndcg_graded_5,
        "ndcg_graded_5",
    );
    close(
        graded::compute_ndcg(&ranked_pairs, &qrels, 10),
        e.ndcg_graded_10,
        "ndcg_graded_10",
    );

    // IR formulas vs numpy (canonical definitions).
    close(binary::average_precision(&ranked, &relevant), e.map, "map");
    close(binary::mrr(&ranked, &relevant), e.mrr, "mrr");
    close(
        binary::precision_at_k(&ranked, &relevant, 5),
        e.precision_at_5,
        "precision_at_5",
    );
    close(
        binary::recall_at_k(&ranked, &relevant, 5),
        e.recall_at_5,
        "recall_at_5",
    );
}
