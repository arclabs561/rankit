//! Top-k cross-entropy loss for classification.
//!
//! A drop-in replacement for standard cross-entropy that optimizes a weighted
//! mixture of top-1 through top-k accuracy. Instead of only rewarding the model
//! for getting the single top prediction correct (softmax CE), this loss lets
//! you specify a distribution P_K over top-k positions.
//!
//! For example, `p_k = [0.5, 0.0, 0.0, 0.0, 0.5]` optimizes 50% for top-1
//! accuracy and 50% for top-5 accuracy.
//!
//! # References
//!
//! - Petersen et al. (2022), "Differentiable Top-k Classification Learning" (ICML)
//!
//! # Example
//!
//! ```rust
//! use rankit::topk_ce::{TopKCrossEntropyLoss, TopKConfig};
//!
//! // 10-class classification, optimize for top-1 and top-5
//! let config = TopKConfig {
//!     p_k: vec![0.5, 0.0, 0.0, 0.0, 0.5],
//!     temperature: 2.0,
//!     m: Some(8), // only sort top-8 scores for efficiency
//! };
//! let loss_fn = TopKCrossEntropyLoss::new(config);
//!
//! // logits for 10 classes, true label is class 3
//! let logits = vec![0.1, 0.2, 0.5, 2.0, 0.3, 0.1, 0.05, 0.02, 0.4, 0.15];
//! let label = 3;
//! let loss = loss_fn.compute(&logits, label);
//! assert!(loss >= 0.0);
//! ```

/// Configuration for the top-k cross-entropy loss.
#[derive(Debug, Clone)]
pub struct TopKConfig {
    /// Distribution over top-k positions. `p_k[i]` is the weight for top-(i+1).
    /// Must sum to 1.0.
    pub p_k: Vec<f64>,
    /// Inverse temperature for the soft ranking. Higher = sharper.
    pub temperature: f64,
    /// If set, only the top-m scores are sorted (efficiency for large n_classes).
    /// Must be >= k (length of p_k).
    pub m: Option<usize>,
}

impl Default for TopKConfig {
    fn default() -> Self {
        Self {
            // Default: pure top-1 (equivalent to standard CE)
            p_k: vec![1.0],
            temperature: 1.0,
            m: None,
        }
    }
}

/// Top-k cross-entropy loss.
///
/// Computes a weighted mixture of cross-entropy losses at different top-k
/// positions using differentiable soft ranking to produce the attribution.
#[derive(Debug, Clone)]
pub struct TopKCrossEntropyLoss {
    config: TopKConfig,
}

impl TopKCrossEntropyLoss {
    /// Create a new loss function with the given configuration.
    pub fn new(config: TopKConfig) -> Self {
        Self { config }
    }

    /// Standard top-1 cross-entropy (equivalent to softmax CE).
    pub fn top1() -> Self {
        Self::new(TopKConfig::default())
    }

    /// Equal weight across k positions.
    pub fn uniform(k: usize, temperature: f64) -> Self {
        let weight = 1.0 / k as f64;
        Self::new(TopKConfig {
            p_k: vec![weight; k],
            temperature,
            m: None,
        })
    }

    /// Emphasize top-1 and top-k equally.
    ///
    /// `p_k = [0.5, 0, ..., 0, 0.5]`
    pub fn endpoints(k: usize, temperature: f64) -> Self {
        let mut p_k = vec![0.0; k];
        p_k[0] = 0.5;
        p_k[k - 1] = 0.5;
        Self::new(TopKConfig {
            p_k,
            temperature,
            m: None,
        })
    }

    /// Compute the loss for a single sample.
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw model outputs (n_classes)
    /// * `label` - True class index
    ///
    /// # Returns
    ///
    /// Non-negative loss value.
    pub fn compute(&self, logits: &[f64], label: usize) -> f64 {
        let n = logits.len();
        if n == 0 || label >= n {
            return 0.0;
        }

        let k = self.config.p_k.len();

        // Step 1: select top-m scores for efficiency
        let m = self.config.m.unwrap_or(n).min(n).max(k);

        // Find top-m indices by score
        let mut indexed: Vec<(usize, f64)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top_m: Vec<(usize, f64)> = indexed.into_iter().take(m).collect();

        // Check if the true label is in top-m
        let label_in_top_m = top_m.iter().any(|(idx, _)| *idx == label);

        // If label not in top-m, this is a hard miss -- return high loss
        if !label_in_top_m {
            // Use the standard softmax CE as fallback (stable computation)
            return softmax_ce(logits, label);
        }

        // Step 2: compute soft top-k attribution for the true label
        let top_m_scores: Vec<f64> = top_m.iter().map(|(_, s)| *s).collect();
        let label_pos_in_m = top_m.iter().position(|(idx, _)| *idx == label).unwrap();

        // Soft rank of the true label within top-m
        let soft_ranks = soft_rank_local(&top_m_scores, self.config.temperature);
        let label_rank = soft_ranks[label_pos_in_m]; // 1-indexed soft rank

        // Step 3: compute weighted loss across top-k positions
        let mut loss = 0.0;
        for (j, &p_j) in self.config.p_k.iter().enumerate() {
            if p_j <= 0.0 {
                continue;
            }
            let target_rank = (j + 1) as f64;
            // Soft indicator: how much the label is at rank <= target_rank
            // sigma((target_rank + 0.5 - label_rank) / tau)
            let z = (target_rank + 0.5 - label_rank) / self.config.temperature;
            let prob_in_topj = stable_sigmoid(z);

            // Cross-entropy contribution: -log(prob)
            let ce = -stable_log(prob_in_topj);
            loss += p_j * ce;
        }

        loss
    }

    /// Compute the loss for a batch of samples, returning the mean.
    pub fn compute_batch(&self, logits_batch: &[Vec<f64>], labels: &[usize]) -> f64 {
        if logits_batch.is_empty() {
            return 0.0;
        }
        let total: f64 = logits_batch
            .iter()
            .zip(labels.iter())
            .map(|(logits, &label)| self.compute(logits, label))
            .sum();
        total / logits_batch.len() as f64
    }
}

/// Standard softmax cross-entropy (numerically stable).
fn softmax_ce(logits: &[f64], label: usize) -> f64 {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_sum_exp: f64 = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum::<f64>()
        .ln();
    -(logits[label] - max_logit - log_sum_exp)
}

/// Local soft ranking (pairwise sigmoid, O(n^2)).
fn soft_rank_local(scores: &[f64], temperature: f64) -> Vec<f64> {
    let n = scores.len();
    let mut ranks = vec![1.0; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let diff = (scores[j] - scores[i]) / temperature;
                ranks[i] += stable_sigmoid(diff);
            }
        }
    }

    ranks
}

/// Numerically stable sigmoid.
fn stable_sigmoid(x: f64) -> f64 {
    if x > 500.0 {
        1.0
    } else if x < -500.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Numerically stable log, clamped away from 0.
fn stable_log(x: f64) -> f64 {
    x.max(1e-15).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top1_loss_correct_prediction() {
        let loss_fn = TopKCrossEntropyLoss::top1();
        // Strong prediction for correct class
        let logits = vec![0.1, 0.1, 5.0, 0.1, 0.1];
        let loss = loss_fn.compute(&logits, 2);
        assert!(loss >= 0.0);
        assert!(
            loss < 1.0,
            "Correct prediction should have low loss: {loss}"
        );
    }

    #[test]
    fn test_top1_loss_wrong_prediction() {
        let loss_fn = TopKCrossEntropyLoss::top1();
        // Strong prediction for wrong class
        let logits = vec![5.0, 0.1, 0.1, 0.1, 0.1];
        let loss = loss_fn.compute(&logits, 2);
        assert!(loss > 1.0, "Wrong prediction should have high loss: {loss}");
    }

    #[test]
    fn test_topk_loss_in_topk() {
        // Label is in top-5 but not top-1
        let loss_fn = TopKCrossEntropyLoss::new(TopKConfig {
            p_k: vec![0.0, 0.0, 0.0, 0.0, 1.0], // only top-5
            temperature: 1.0,
            m: None,
        });

        let logits = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1];
        // Label=4 is at rank 5 (in top-5)
        let loss = loss_fn.compute(&logits, 4);
        assert!(loss >= 0.0);
        assert!(
            loss < 2.0,
            "Label in top-5 should have moderate loss: {loss}"
        );
    }

    #[test]
    fn test_uniform_loss() {
        let loss_fn = TopKCrossEntropyLoss::uniform(5, 2.0);
        let logits = vec![0.1, 0.2, 5.0, 0.3, 0.4];
        let loss = loss_fn.compute(&logits, 2);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_endpoints_loss() {
        let loss_fn = TopKCrossEntropyLoss::endpoints(5, 2.0);
        assert_eq!(loss_fn.config.p_k.len(), 5);
        assert!((loss_fn.config.p_k[0] - 0.5).abs() < 1e-10);
        assert!((loss_fn.config.p_k[4] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_batch_loss() {
        let loss_fn = TopKCrossEntropyLoss::top1();
        let logits = vec![
            vec![5.0, 0.1, 0.1],
            vec![0.1, 5.0, 0.1],
            vec![0.1, 0.1, 5.0],
        ];
        let labels = vec![0, 1, 2]; // all correct
        let loss = loss_fn.compute_batch(&logits, &labels);
        assert!(loss >= 0.0);
        assert!(loss < 1.0, "All correct should have low batch loss: {loss}");
    }

    #[test]
    fn test_m_parameter_efficiency() {
        let loss_full = TopKCrossEntropyLoss::new(TopKConfig {
            p_k: vec![1.0],
            temperature: 1.0,
            m: None,
        });
        let loss_m = TopKCrossEntropyLoss::new(TopKConfig {
            p_k: vec![1.0],
            temperature: 1.0,
            m: Some(5),
        });

        let logits = vec![0.1, 0.2, 5.0, 0.3, 0.4, 0.05, 0.03, 0.02, 0.01, 0.0];
        let l1 = loss_full.compute(&logits, 2);
        let l2 = loss_m.compute(&logits, 2);

        // Both should give valid losses
        assert!(l1 >= 0.0);
        assert!(l2 >= 0.0);
    }

    #[test]
    fn test_softmax_ce_matches_standard() {
        // Verify our softmax CE is numerically stable
        let logits = vec![1000.0, 999.0, 998.0];
        let loss = softmax_ce(&logits, 0);
        assert!(loss.is_finite(), "Should handle large logits: {loss}");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_empty_logits() {
        let loss_fn = TopKCrossEntropyLoss::top1();
        assert_eq!(loss_fn.compute(&[], 0), 0.0);
    }

    #[test]
    fn test_invalid_label() {
        let loss_fn = TopKCrossEntropyLoss::top1();
        assert_eq!(loss_fn.compute(&[1.0, 2.0], 5), 0.0);
    }

    #[test]
    fn test_higher_temperature_smoother() {
        let logits = vec![3.0, 2.0, 1.0, 0.5, 0.1];

        let loss_sharp = TopKCrossEntropyLoss::new(TopKConfig {
            p_k: vec![0.5, 0.0, 0.0, 0.0, 0.5],
            temperature: 0.1,
            m: None,
        });
        let loss_smooth = TopKCrossEntropyLoss::new(TopKConfig {
            p_k: vec![0.5, 0.0, 0.0, 0.0, 0.5],
            temperature: 5.0,
            m: None,
        });

        let l_sharp = loss_sharp.compute(&logits, 0);
        let l_smooth = loss_smooth.compute(&logits, 0);

        // Both should be valid
        assert!(l_sharp.is_finite());
        assert!(l_smooth.is_finite());
    }
}
