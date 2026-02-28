//! Gumbel-Softmax sampling and relaxed top-k.
//!
//! Requires the `gumbel` feature (enables `rand` dependency).

use rand::Rng;

/// Generate Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0,1).
pub fn gumbel_noise(rng: &mut impl Rng) -> f64 {
    let u: f64 = rng.random_range(0.0..1.0);
    let u = u.clamp(1e-10, 1.0 - 1e-10);
    -(-u.ln()).ln()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let n = logits.len();
    if n == 0 {
        return vec![];
    }

    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f64 = exps.iter().sum();

    if sum > 1e-10 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    }
}

/// Gumbel-Softmax: differentiable sampling from categorical distribution.
///
/// From: "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., ICLR 2017)
///
/// # Arguments
///
/// * `logits` - Unnormalized log probabilities
/// * `temperature` - Lower = sharper, higher = smoother
/// * `scale` - Scaling factor for logits
/// * `rng` - Random number generator
pub fn gumbel_softmax(
    logits: &[f64],
    temperature: f64,
    scale: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let n = logits.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    let mut gumbel_logits = Vec::with_capacity(n);
    for &logit in logits {
        let g = gumbel_noise(rng);
        gumbel_logits.push((g + scale * logit) / temperature);
    }

    softmax(&gumbel_logits)
}

/// Relaxed Top-k using Gumbel-Softmax.
///
/// From: "Gumbel Reranking" (Huang et al., ACL 2025)
///
/// Creates a soft mask where top-k elements have high values (~1.0)
/// and others have low values (~0.0).
pub fn relaxed_topk_gumbel(
    scores: &[f64],
    k: usize,
    temperature: f64,
    scale: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let n = scores.len();
    if n == 0 || k == 0 {
        return vec![];
    }
    if k >= n {
        return vec![1.0; n];
    }

    let mut max_mask = vec![0.0; n];

    for _ in 0..k {
        let mask = gumbel_softmax(scores, temperature, scale, rng);
        for i in 0..n {
            if mask[i] > max_mask[i] {
                max_mask[i] = mask[i];
            }
        }
    }

    max_mask
}

/// Generate Gumbel-based attention mask for RAG reranking.
///
/// Convenience wrapper around `relaxed_topk_gumbel`.
pub fn gumbel_attention_mask(
    reranker_scores: &[f64],
    k: usize,
    temperature: f64,
    scale: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    relaxed_topk_gumbel(reranker_scores, k, temperature, scale, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_gumbel_softmax() {
        let mut rng = StdRng::seed_from_u64(42);
        let logits = vec![0.5, 1.0, 0.3];
        let probs = gumbel_softmax(&logits, 0.5, 1.0, &mut rng);

        assert_eq!(probs.len(), logits.len());
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_relaxed_topk_gumbel() {
        let mut rng = StdRng::seed_from_u64(42);
        let scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];
        let mask = relaxed_topk_gumbel(&scores, 3, 0.5, 1.0, &mut rng);

        assert_eq!(mask.len(), scores.len());
        assert!(mask.iter().all(|&m| m >= 0.0 && m <= 1.0));
    }

    #[test]
    fn test_edge_cases() {
        let mut rng = StdRng::seed_from_u64(42);

        let empty: Vec<f64> = vec![];
        assert_eq!(gumbel_softmax(&empty, 0.5, 1.0, &mut rng).len(), 0);

        let single = vec![1.0];
        assert_eq!(gumbel_softmax(&single, 0.5, 1.0, &mut rng), vec![1.0]);

        let scores = vec![0.5, 0.3];
        assert_eq!(
            relaxed_topk_gumbel(&scores, 5, 0.5, 1.0, &mut rng),
            vec![1.0, 1.0]
        );
    }
}
