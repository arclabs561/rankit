//! Linear learning-to-rank on the LightGBM ranking dataset (LETOR-style: 300
//! sparse features, graded relevance 0-4, query-grouped). Trains on real data,
//! reports held-out NDCG@10 and MAP via rankit's own graded eval.
//!
//! ```sh
//! ./scripts/fetch_lightgbm_rank.sh
//! cargo run --release --example ltr_lightgbm_train_eval
//! ```
//!
//! Score is `w . x`. `compute_lambdarank_gradients` returns dC/ds_i, so by the
//! chain rule dC/dw = sum_i (dC/ds_i) x_i and the weight update is
//! `w -= lr * dC/dw`.

use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

use rankit::eval::graded::{compute_map, compute_ndcg};
use rankit::gradients::{compute_lambdarank_gradients, LambdaRankParams};

/// One query: a list of documents (dense feature vectors) and their relevance.
struct Query {
    docs: Vec<Vec<f32>>,
    rels: Vec<f32>,
}

/// Parse a LibSVM sparse file + its `.query` group-size sibling into queries.
/// Each data row is `label idx:val idx:val ...` (1-based feature indices).
fn load(data_path: &Path, query_path: &Path, n_features: usize) -> std::io::Result<Vec<Query>> {
    let data = std::fs::read_to_string(data_path)?;
    let groups = std::fs::read_to_string(query_path)?;

    let mut rows: Vec<(f32, Vec<f32>)> = Vec::new();
    for line in data.lines().filter(|l| !l.trim().is_empty()) {
        let mut toks = line.split_whitespace();
        let label: f32 = toks.next().unwrap().parse().unwrap();
        let mut feat = vec![0.0f32; n_features];
        for tok in toks {
            let (idx, val) = tok.split_once(':').unwrap();
            let idx: usize = idx.parse().unwrap();
            if idx >= 1 && idx <= n_features {
                feat[idx - 1] = val.parse().unwrap();
            }
        }
        rows.push((label, feat));
    }

    let sizes: Vec<usize> = groups
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect();

    let mut queries = Vec::with_capacity(sizes.len());
    let mut cursor = 0;
    for size in sizes {
        let slice = &rows[cursor..cursor + size];
        queries.push(Query {
            docs: slice.iter().map(|(_, f)| f.clone()).collect(),
            rels: slice.iter().map(|(r, _)| *r).collect(),
        });
        cursor += size;
    }
    Ok(queries)
}

/// Detect the max 1-based feature index across both LibSVM files.
fn max_feature_index(paths: &[&Path]) -> std::io::Result<usize> {
    let mut max_idx = 0;
    for p in paths {
        for line in std::fs::read_to_string(p)?.lines() {
            for tok in line.split_whitespace().skip(1) {
                if let Some((idx, _)) = tok.split_once(':') {
                    max_idx = max_idx.max(idx.parse::<usize>().unwrap_or(0));
                }
            }
        }
    }
    Ok(max_idx)
}

fn score(w: &[f32], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(a, b)| a * b).sum()
}

/// Mean NDCG@k and MAP over a split, ranking each query's docs by `w . x`.
fn evaluate(queries: &[Query], w: &[f32], k: usize) -> (f64, f64) {
    let (mut ndcg_sum, mut map_sum) = (0.0, 0.0);
    for q in queries {
        let mut ranked: Vec<(String, f32)> = q
            .docs
            .iter()
            .enumerate()
            .map(|(i, x)| (i.to_string(), score(w, x)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let qrels: HashMap<String, u32> = q
            .rels
            .iter()
            .enumerate()
            .map(|(i, &r)| (i.to_string(), r as u32))
            .collect();

        ndcg_sum += compute_ndcg(&ranked, &qrels, k);
        map_sum += compute_map(&ranked, &qrels);
    }
    let n = queries.len() as f64;
    (ndcg_sum / n, map_sum / n)
}

/// Deterministic in-place Fisher-Yates shuffle (xorshift, no rng dependency).
fn shuffle(order: &mut [usize], state: &mut u64) {
    for i in (1..order.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let j = (*state % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

fn main() -> ExitCode {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/lightgbm_rank");
    let train_data = dir.join("rank.train");
    let train_q = dir.join("rank.train.query");
    let test_data = dir.join("rank.test");
    let test_q = dir.join("rank.test.query");

    if !train_data.exists() {
        // Data-gated: no-op cleanly (exit 0) when the dataset is absent, so CI
        // that compiles/runs examples does not fail on the missing fixture.
        eprintln!(
            "dataset not found at {}\nrun: ./scripts/fetch_lightgbm_rank.sh",
            dir.display()
        );
        return ExitCode::SUCCESS;
    }

    let n_features = max_feature_index(&[&train_data, &test_data]).unwrap();
    let train = load(&train_data, &train_q, n_features).unwrap();
    let test = load(&test_data, &test_q, n_features).unwrap();
    println!(
        "features: {n_features}  train queries: {}  test queries: {}",
        train.len(),
        test.len()
    );

    const K: usize = 10;
    let params = LambdaRankParams::default();

    // Random-init baseline (small seeded weights) and natural-order baseline.
    let mut w = vec![0.0f32; n_features];
    let mut seed = 0x9E3779B97F4A7C15u64;
    for wi in w.iter_mut() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *wi = ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.02;
    }
    let (ndcg0, map0) = evaluate(&test, &w, K);
    println!("\nbaseline (random init)   NDCG@{K} = {ndcg0:.4}   MAP = {map0:.4}");

    // Train: per-query SGD on LambdaRank gradients.
    let epochs = 60;
    let lr = 1.0f32;
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = 0xDEADBEEFCAFEu64;
    let mut grad = vec![0.0f32; n_features];

    for epoch in 0..epochs {
        shuffle(&mut order, &mut rng);
        for &qi in &order {
            let q = &train[qi];
            let scores: Vec<f32> = q.docs.iter().map(|x| score(&w, x)).collect();
            let lambdas = match compute_lambdarank_gradients(&scores, &q.rels, params, Some(K)) {
                Ok(l) => l,
                Err(_) => continue, // single-doc or all-equal-relevance query
            };
            grad.iter_mut().for_each(|g| *g = 0.0);
            for (i, x) in q.docs.iter().enumerate() {
                let li = lambdas[i];
                for (g, &xf) in grad.iter_mut().zip(x) {
                    *g += li * xf;
                }
            }
            for (wf, &g) in w.iter_mut().zip(&grad) {
                *wf -= lr * g;
            }
        }
        if epoch % 10 == 9 || epoch == 0 {
            let (tr_ndcg, _) = evaluate(&train, &w, K);
            println!("epoch {:>2}  train NDCG@{K} = {tr_ndcg:.4}", epoch + 1);
        }
    }

    let (ndcg1, map1) = evaluate(&test, &w, K);
    println!("\ntrained                  NDCG@{K} = {ndcg1:.4}   MAP = {map1:.4}");
    println!(
        "improvement              NDCG@{K} {:+.4}   MAP {:+.4}",
        ndcg1 - ndcg0,
        map1 - map0
    );

    ExitCode::SUCCESS
}
