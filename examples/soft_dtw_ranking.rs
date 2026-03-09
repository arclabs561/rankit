//! Soft-DTW as a ranking sequence loss.
//!
//! Demonstrates using structops' differentiable DTW to compare ranked sequences.
//! Unlike pointwise losses (RankNet, LambdaLoss), Soft-DTW tolerates local
//! order variations and different-length sequences -- useful when ground truth
//! is an ordering rather than absolute relevance labels.
//!
//! Run: cargo run --example soft_dtw_ranking

fn main() {
    use rankit::soft_rank;
    use structops::soft_dtw::{soft_dtw, soft_dtw_divergence};

    println!("=== Soft-DTW for Ranking Comparison ===\n");

    // Two ranking models produce score vectors for the same 5 documents
    let relevance = [3.0, 2.0, 1.0, 0.0, 0.0]; // ground truth

    let model_a_scores = [0.9, 0.7, 0.4, 0.1, 0.05]; // good ranker
    let model_b_scores = [0.5, 0.8, 0.3, 0.9, 0.1]; // poor ranker (swaps top docs)

    // Convert scores to soft ranks
    let temp = 0.5;
    let ranks_true = soft_rank(relevance.as_ref(), temp);
    let ranks_a = soft_rank(model_a_scores.as_ref(), temp);
    let ranks_b = soft_rank(model_b_scores.as_ref(), temp);

    println!("Ground truth ranks: {:.2?}", ranks_true);
    println!("Model A ranks:     {:.2?}", ranks_a);
    println!("Model B ranks:     {:.2?}", ranks_b);

    // Soft-DTW divergence: measures sequence alignment distance
    let gamma = 1.0;
    let dtw_a = soft_dtw_divergence(&ranks_true, &ranks_a, gamma).unwrap();
    let dtw_b = soft_dtw_divergence(&ranks_true, &ranks_b, gamma).unwrap();

    println!("\nSoft-DTW divergence (lower = better):");
    println!("  Model A vs truth: {dtw_a:.4}");
    println!("  Model B vs truth: {dtw_b:.4}");
    assert!(
        dtw_a < dtw_b,
        "Good ranker should have lower DTW divergence"
    );

    // Compare with raw Soft-DTW (not divergence) at different temperatures
    println!("\n--- Temperature sensitivity ---\n");
    println!("  gamma | DTW(A,truth) | DTW(B,truth) | ratio B/A");
    println!("  ------|-------------|-------------|----------");
    for &g in &[0.1, 0.5, 1.0, 2.0, 5.0] {
        let da = soft_dtw(&ranks_true, &ranks_a, g).unwrap();
        let db = soft_dtw(&ranks_true, &ranks_b, g).unwrap();
        println!("  {g:5.1} | {da:11.4} | {db:11.4} | {:.2}", db / da);
    }

    // Different-length comparison: comparing top-3 vs full ranking
    println!("\n--- Different-length sequences ---\n");
    let top3_true = &ranks_true[..3];
    let full_a = &ranks_a;
    let dtw_partial = soft_dtw(top3_true, full_a, 1.0).unwrap();
    println!("Soft-DTW(top-3 truth, full model A): {dtw_partial:.4}");
    println!("(DTW naturally handles length mismatch via warping)");
}
