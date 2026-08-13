//! Differentiable soft ranking on a simple score vector.
//!
//! Demonstrates how `soft_rank` produces continuous rank approximations that
//! converge to discrete ranks as regularization strength increases.

use rankit::{pairwise_logistic_rank, soft_rank, soft_rank_sigmoid, soft_rank_smooth_i};

fn main() {
    let scores = vec![5.0, 1.0, 2.0, 4.0, 3.0];
    // Discrete ranks (0-indexed, 0=lowest): [4, 0, 1, 3, 2]

    // --- Effect of regularization strength (inverse temperature) ---
    println!("Scores: {:?}\n", scores);
    for &alpha in &[0.1, 1.0, 10.0, 100.0] {
        let ranks = soft_rank(&scores, alpha);
        let formatted: Vec<String> = ranks.iter().map(|r| format!("{r:.3}")).collect();
        println!("alpha={alpha:>5.1}  ranks=[{}]", formatted.join(", "));
    }
    // Higher alpha -> ranks snap closer to [4.0, 0.0, 1.0, 3.0, 2.0]

    // --- Compare ranking methods ---
    let alpha = 5.0;
    println!("\nMethod comparison (alpha={alpha}):");
    #[allow(clippy::type_complexity)]
    let methods: [(&str, fn(&[f64], f64) -> Vec<f64>); 3] = [
        ("Sigmoid", soft_rank_sigmoid),
        ("Pairwise logistic", pairwise_logistic_rank),
        ("Sigmoid alias", soft_rank_smooth_i),
    ];
    for (name, method) in &methods {
        let ranks = method(&scores, alpha);
        let formatted: Vec<String> = ranks.iter().map(|r| format!("{r:.3}")).collect();
        println!(
            "  {name:<12} [{formatted}]",
            formatted = formatted.join(", ")
        );
    }

    // --- Ordering preservation check ---
    let ranks = soft_rank(&scores, 10.0);
    assert!(ranks[1] < ranks[2], "score 1.0 should rank below 2.0");
    assert!(ranks[2] < ranks[4], "score 2.0 should rank below 3.0");
    assert!(ranks[4] < ranks[3], "score 3.0 should rank below 4.0");
    assert!(ranks[3] < ranks[0], "score 4.0 should rank below 5.0");
    println!("\nOrdering preserved: all pairwise rank comparisons match score order.");
}
