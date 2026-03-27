//! Gumbel-Softmax sampling and relaxed top-k.
//!
//! Delegates to [`kuji`] for correct implementations of the Gumbel-Softmax
//! trick and the iterated masked-softmax relaxed top-k (Kool et al., 2019).
//!
//! Requires the `gumbel` feature (enables `rand` + `kuji` dependencies).

use rand::Rng;

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
    kuji::gumbel_softmax(logits, temperature, scale, rng)
}

/// Relaxed Top-k using iterated Gumbel-Softmax (Kool et al., 2019).
///
/// Creates a soft k-hot vector where top-k elements have high values (~1.0)
/// and others have low values (~0.0). Uses iterated masked-softmax to enforce
/// without-replacement structure: each selection suppresses previously selected
/// items via `log(1 - onehot)`.
///
/// The output sums to approximately k (unlike element-wise max of independent
/// draws, which does not enforce this property).
pub fn relaxed_topk_gumbel(
    scores: &[f64],
    k: usize,
    temperature: f64,
    scale: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    kuji::relaxed_topk_gumbel(scores, k, temperature, scale, rng)
}

/// Generate Gumbel-based attention mask for RAG reranking.
///
/// Convenience wrapper around [`relaxed_topk_gumbel`].
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
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn test_relaxed_topk_sums_to_k() {
        let mut rng = StdRng::seed_from_u64(42);
        let scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];
        let k = 3;
        let mask = relaxed_topk_gumbel(&scores, k, 0.1, 1.0, &mut rng);

        assert_eq!(mask.len(), scores.len());
        // Values are non-negative (accumulated softmax outputs, may slightly exceed 1.0)
        assert!(mask.iter().all(|&m| m >= -1e-10));

        // Key property: the mask should sum to approximately k
        let sum: f64 = mask.iter().sum();
        assert!(
            (sum - k as f64).abs() < 1.5,
            "relaxed top-k mask should sum to ~{k}, got {sum}"
        );
    }

    #[test]
    fn test_edge_cases() {
        let mut rng = StdRng::seed_from_u64(42);

        let empty: Vec<f64> = vec![];
        assert!(relaxed_topk_gumbel(&empty, 3, 0.5, 1.0, &mut rng).is_empty());

        let scores = vec![0.5, 0.3];
        assert_eq!(
            relaxed_topk_gumbel(&scores, 5, 0.5, 1.0, &mut rng),
            vec![1.0, 1.0]
        );
    }
}
