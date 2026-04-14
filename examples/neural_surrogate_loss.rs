//! NeuralLoss-style surrogate: learn a loss function from metric evaluations.
//!
//! Instead of hand-crafting a differentiable approximation of NDCG (like
//! ApproxNDCG or LambdaLoss), learn a surrogate loss that correlates with
//! the true metric. This example uses a simple linear surrogate.
//!
//! Pipeline:
//! 1. Generate random permutations of a relevance list
//! 2. Compute true NDCG for each permutation
//! 3. Fit a linear surrogate: loss(scores, relevance) = sum_ij w_ij * f(s_i, r_j)
//! 4. Compare surrogate ordering with true NDCG ordering
//!
//! Reference: Rolinek, Musil, Paulus, Vlastelica, Hobein-Friberg, Pogancic
//! (2020), "Optimizing Rank-Based Metrics with Blackbox Differentiation"

use rankit::losses::{approx_ndcg, lambda_loss};

fn main() {
    let relevance = [3.0, 2.0, 1.0, 0.0, 0.0];
    let n = relevance.len();

    // Generate permutations and compute true NDCG.
    let permutations = generate_permutations(n, 100, 42);
    let mut samples: Vec<(Vec<f64>, f64)> = Vec::new();

    for perm in &permutations {
        // Create scores that induce this permutation (higher score = ranked higher).
        let scores: Vec<f64> = perm.iter().map(|&i| (n - i) as f64).collect();
        let ndcg = compute_ndcg(&scores, &relevance);
        samples.push((scores, ndcg));
    }

    // Fit a simple linear surrogate: predicted_loss = sum_i w_i * s_i * r_i
    // (This is intentionally simple -- a real NeuralLoss would use an MLP.)
    let weights = fit_linear_surrogate(&samples, &relevance, 0.01, 200);

    println!("NeuralLoss-style surrogate learning");
    println!("  {} training permutations, {} items", samples.len(), n);
    println!(
        "  Learned weights: {:?}",
        weights
            .iter()
            .map(|w| format!("{w:.3}"))
            .collect::<Vec<_>>()
    );
    println!();

    // Evaluate: does the surrogate rank permutations similarly to true NDCG?
    let test_perms = generate_permutations(n, 50, 99);
    let mut true_ndcgs = Vec::new();
    let mut surrogate_scores = Vec::new();

    for perm in &test_perms {
        let scores: Vec<f64> = perm.iter().map(|&i| (n - i) as f64).collect();
        let ndcg = compute_ndcg(&scores, &relevance);
        let surrogate = apply_surrogate(&scores, &relevance, &weights);
        true_ndcgs.push(ndcg);
        surrogate_scores.push(surrogate);
    }

    let correlation = spearman_correlation(&true_ndcgs, &surrogate_scores);
    println!("Spearman correlation (surrogate vs true NDCG): {correlation:.4}");

    // Compare with existing losses.
    let mut approx_scores = Vec::new();
    let mut lambda_scores = Vec::new();
    for perm in &test_perms {
        let scores: Vec<f64> = perm.iter().map(|&i| (n - i) as f64).collect();
        approx_scores.push(-approx_ndcg(&scores, &relevance, 1.0, None));
        lambda_scores.push(-lambda_loss(&scores, &relevance, None));
    }

    let corr_approx = spearman_correlation(&true_ndcgs, &approx_scores);
    let corr_lambda = spearman_correlation(&true_ndcgs, &lambda_scores);

    println!("Spearman correlation (ApproxNDCG vs true NDCG): {corr_approx:.4}");
    println!("Spearman correlation (LambdaLoss vs true NDCG): {corr_lambda:.4}");
    println!();

    if correlation > 0.5 {
        println!("Surrogate learned a useful approximation of NDCG ranking.");
    } else {
        println!("Surrogate correlation is weak (expected for a linear model).");
    }
}

fn compute_ndcg(scores: &[f64], relevance: &[f64]) -> f64 {
    let _n = scores.len();
    let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut dcg = 0.0;
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        dcg += (2.0_f64.powf(relevance[idx]) - 1.0) / (rank as f64 + 2.0).log2();
    }

    // Ideal DCG.
    let mut sorted_rel: Vec<f64> = relevance.to_vec();
    sorted_rel.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut idcg = 0.0;
    for (rank, &r) in sorted_rel.iter().enumerate() {
        idcg += (2.0_f64.powf(r) - 1.0) / (rank as f64 + 2.0).log2();
    }

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn generate_permutations(n: usize, count: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = seed;
    let mut result = Vec::new();
    for _ in 0..count {
        let mut perm: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle with simple xorshift.
        for i in (1..n).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            perm.swap(i, j);
        }
        result.push(perm);
    }
    result
}

fn fit_linear_surrogate(
    samples: &[(Vec<f64>, f64)],
    relevance: &[f64],
    lr: f64,
    steps: usize,
) -> Vec<f64> {
    let n = relevance.len();
    let mut weights = vec![1.0 / n as f64; n];

    for _ in 0..steps {
        let mut grad = vec![0.0; n];
        let mut total_loss = 0.0;

        for (scores, target_ndcg) in samples {
            let predicted = apply_surrogate(scores, relevance, &weights);
            let residual = predicted - target_ndcg;
            total_loss += residual * residual;

            for i in 0..n {
                grad[i] += 2.0 * residual * scores[i] * relevance[i];
            }
        }

        let batch = samples.len() as f64;
        // Gradient clipping.
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let clip = 10.0;
        let scale = if grad_norm > clip {
            clip / grad_norm
        } else {
            1.0
        };
        for i in 0..n {
            weights[i] -= lr * scale * grad[i] / batch;
        }

        let _ = total_loss; // used for monitoring if needed
    }

    weights
}

fn apply_surrogate(scores: &[f64], relevance: &[f64], weights: &[f64]) -> f64 {
    scores
        .iter()
        .zip(relevance)
        .zip(weights)
        .map(|((&s, &r), &w)| w * s * r)
        .sum()
}

fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let rank_x = rank(x);
    let rank_y = rank(y);

    let mean_x: f64 = rank_x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = rank_y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = rank_x[i] - mean_x;
        let dy = rank_y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        ranks[idx] = rank as f64;
    }
    ranks
}
