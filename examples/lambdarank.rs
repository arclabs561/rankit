//! LambdaRank gradient computation on sample data.
//!
//! Shows how LambdaRank produces per-document gradient signals that push
//! higher-relevance documents toward higher scores. The gradients are
//! weighted by the NDCG change from swapping each document pair.

use rankit::{
    compute_lambdarank_gradients, ndcg_at_k,
    LambdaRankParams, LambdaRankTrainer,
};

fn main() {
    // Model scores (current predictions) and ground-truth relevance labels.
    // The model currently ranks doc_1 (rel=1) above doc_0 (rel=3) -- a mis-ranking.
    let scores = vec![0.5_f32, 0.8, 0.3, 0.1, 0.6];
    let relevance = vec![3.0_f32, 1.0, 2.0, 0.0, 1.0];

    // Current NDCG before any update.
    // ndcg_at_k expects relevance in the model's predicted order.
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let ordered_rel: Vec<f32> = indices.iter().map(|&i| relevance[i]).collect();
    let ndcg = ndcg_at_k(&ordered_rel, None, true).unwrap();
    println!("Current ranking by score: {:?}", indices);
    println!("Relevance in that order:  {:?}", ordered_rel);
    println!("NDCG (full list):         {ndcg:.4}\n");

    // --- Compute gradients with default params ---
    let params = LambdaRankParams::default();
    let lambdas = compute_lambdarank_gradients(&scores, &relevance, params, None).unwrap();

    println!("LambdaRank gradients (default params):");
    for (i, (&s, &l)) in scores.iter().zip(lambdas.iter()).enumerate() {
        let rel = relevance[i];
        let direction = if l > 0.0 { "push DOWN" } else if l < 0.0 { "push UP" } else { "no change" };
        println!("  doc_{i}: score={s:.1}, rel={rel:.0}, lambda={l:+.6} ({direction})");
    }
    // Negative lambda -> gradient pushes score up; positive -> pushes score down.

    // --- Trainer API with batch support ---
    let trainer = LambdaRankTrainer::new(LambdaRankParams {
        sigma: 1.0,
        query_normalization: true,
        cost_sensitivity: true,
        score_normalization: false,
        exponential_gain: true,
    });

    // Two queries, each with their own document set.
    let batch_scores = vec![
        vec![0.5, 0.8, 0.3],
        vec![0.9, 0.2, 0.7, 0.4],
    ];
    let batch_relevance = vec![
        vec![3.0, 1.0, 2.0],
        vec![0.0, 3.0, 1.0, 2.0],
    ];

    let batch_lambdas = trainer
        .compute_gradients_batch(&batch_scores, &batch_relevance, None)
        .unwrap();

    println!("\nBatch gradients ({} queries):", batch_lambdas.len());
    for (q, lambdas) in batch_lambdas.iter().enumerate() {
        let formatted: Vec<String> = lambdas.iter().map(|l| format!("{l:+.6}")).collect();
        println!("  query {q}: [{}]", formatted.join(", "));
    }
}
