//! Significance testing for learning-to-rank: `rankit` metrics + `statskit` stats.
//!
//! A single mean NDCG does not tell you whether one ranker actually beats
//! another — LTR results are reported with a paired significance test over
//! per-query metrics. This composes `rankit::eval` (true NDCG@k) with
//! `statskit`'s Wilcoxon signed-rank test and BCa bootstrap CI: the honest
//! comparison layer that ranker training and evaluation should use instead of
//! eyeballing an average.
//!
//! Run: `cargo run --example ltr_significance --features eval`

use rankit::eval::binary::ndcg_at_k;
use statskit::{bootstrap_bca, wilcoxon, BootstrapConfig};
use std::collections::HashSet;

/// Deterministic xorshift, so the example is reproducible.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn main() {
    let n_queries = 24;
    let n_docs = 60usize;
    let k = 10;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;

    // Per-query NDCG@k for two rankers over the same queries (a paired sample).
    let mut strong = Vec::with_capacity(n_queries);
    let mut weak = Vec::with_capacity(n_queries);

    for q in 0..n_queries {
        // Five relevant documents per query.
        let relevant: HashSet<usize> = (0..5).map(|i| (q * 11 + i * 7) % n_docs).collect();

        // Strong ranker: relevant documents at the top, the rest after.
        let mut strong_rank: Vec<usize> = relevant.iter().copied().collect();
        strong_rank.extend((0..n_docs).filter(|d| !relevant.contains(d)));

        // Weak ranker: a deterministic shuffle, so relevant docs are scattered.
        let mut weak_rank: Vec<usize> = (0..n_docs).collect();
        for i in (1..n_docs).rev() {
            let j = (next(&mut rng) as usize) % (i + 1);
            weak_rank.swap(i, j);
        }

        strong.push(ndcg_at_k(&strong_rank, &relevant, k));
        weak.push(ndcg_at_k(&weak_rank, &relevant, k));
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!("LTR significance: rankit NDCG@{k} + statskit");
    println!("  strong ranker mean NDCG@{k}: {:.4}", mean(&strong));
    println!("  weak   ranker mean NDCG@{k}: {:.4}", mean(&weak));

    // Paired Wilcoxon signed-rank test over the per-query NDCG.
    let w = wilcoxon(&strong, &weak, 0.05);
    println!(
        "  Wilcoxon: statistic={:.1}, p={:.2e}, significant={}",
        w.statistic, w.p_value, w.significant
    );

    // BCa bootstrap confidence interval on the mean NDCG difference.
    let boot = bootstrap_bca(
        &strong,
        &weak,
        |a, b| mean(a) - mean(b),
        BootstrapConfig::default(),
    );
    println!(
        "  mean NDCG@{k} difference: {:.4} (95% CI [{:.4}, {:.4}])",
        boot.point_estimate, boot.lower, boot.upper
    );

    // The strong ranker beats the weak one significantly, and the bootstrap CI
    // on the difference excludes zero. A regression that broke the metric or the
    // test would fail one of these.
    assert!(
        w.significant,
        "strong vs weak NDCG should be significant by Wilcoxon"
    );
    assert!(
        boot.lower > 0.0,
        "the mean-NDCG-difference CI should exclude zero"
    );
    println!("  [PASS] strong ranker significantly beats weak (p<0.05, CI excludes 0)");
}
