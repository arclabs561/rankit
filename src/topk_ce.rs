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
//! let loss_fn = TopKCrossEntropyLoss::try_new(config)?;
//!
//! // logits for 10 classes, true label is class 3
//! let logits = vec![0.1, 0.2, 0.5, 2.0, 0.3, 0.1, 0.05, 0.02, 0.4, 0.15];
//! let label = 3;
//! let loss = loss_fn.compute(&logits, label);
//! assert!(loss >= 0.0);
//! # Ok::<(), rankit::topk_ce::TopKConfigError>(())
//! ```

use thiserror::Error;

use crate::{neural_sort, soft_sort, SortingError};

const PROBABILITY_SUM_TOLERANCE: f64 = 1e-12;

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

impl TopKConfig {
    /// Validate the probability distribution, temperature, and truncation.
    pub fn validate(&self) -> Result<(), TopKConfigError> {
        if self.p_k.is_empty() {
            return Err(TopKConfigError::EmptyDistribution);
        }

        for (index, &probability) in self.p_k.iter().enumerate() {
            if !probability.is_finite() {
                return Err(TopKConfigError::NonFiniteProbability { index, probability });
            }
            if probability < 0.0 {
                return Err(TopKConfigError::NegativeProbability { index, probability });
            }
        }

        let sum: f64 = self.p_k.iter().sum();
        if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err(TopKConfigError::NonNormalizedDistribution { sum });
        }

        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(TopKConfigError::InvalidTemperature(self.temperature));
        }

        if let Some(m) = self.m {
            if m < self.p_k.len() {
                return Err(TopKConfigError::InvalidTruncation {
                    m,
                    k: self.p_k.len(),
                });
            }
        }

        Ok(())
    }
}

/// Invalid top-k cross-entropy configuration.
#[derive(Debug, Clone, Error, PartialEq)]
#[non_exhaustive]
pub enum TopKConfigError {
    /// The distribution contains no top-k positions.
    #[error("p_k must not be empty")]
    EmptyDistribution,
    /// A distribution entry is NaN or infinite.
    #[error("p_k[{index}] must be finite, got {probability}")]
    NonFiniteProbability {
        /// Index of the invalid entry.
        index: usize,
        /// Invalid probability.
        probability: f64,
    },
    /// A distribution entry is negative.
    #[error("p_k[{index}] must be non-negative, got {probability}")]
    NegativeProbability {
        /// Index of the invalid entry.
        index: usize,
        /// Invalid probability.
        probability: f64,
    },
    /// The distribution does not sum to one.
    #[error("p_k must sum to 1 (within {PROBABILITY_SUM_TOLERANCE}), got {sum}")]
    NonNormalizedDistribution {
        /// Observed sum.
        sum: f64,
    },
    /// The smoothing temperature is not finite and strictly positive.
    #[error("temperature must be finite and greater than zero, got {0}")]
    InvalidTemperature(f64),
    /// The top-m truncation is smaller than the largest requested k.
    #[error("m must be at least the number of p_k entries ({k}), got {m}")]
    InvalidTruncation {
        /// Requested truncation.
        m: usize,
        /// Number of top-k positions.
        k: usize,
    },
}

/// Invalid input to a top-k cross-entropy computation.
#[derive(Debug, Clone, Error, PartialEq)]
#[non_exhaustive]
pub enum TopKComputeError {
    /// The loss was created with an invalid configuration.
    #[error(transparent)]
    InvalidConfig(#[from] TopKConfigError),
    /// A sample has no class logits.
    #[error("logits must not be empty")]
    EmptyLogits,
    /// The label does not identify one of the sample's classes.
    #[error("label {label} is out of bounds for {classes} classes")]
    LabelOutOfBounds {
        /// Invalid label.
        label: usize,
        /// Number of available classes.
        classes: usize,
    },
    /// The number of samples and labels differs.
    #[error("logits batch length ({samples}) does not match labels length ({labels})")]
    BatchLengthMismatch {
        /// Number of samples.
        samples: usize,
        /// Number of labels.
        labels: usize,
    },
}

/// Full-matrix relaxation used by [`DifferentiableTopKCrossEntropyLoss`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopKRelaxation {
    /// NeuralSort's row-stochastic permutation relaxation.
    NeuralSort,
    /// SoftSort with absolute-distance logits.
    SoftSort,
}

/// Treatment of the top-1 component in the differentiable loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Top1Mode {
    /// Derive every top-k inclusion probability from the relaxed sort matrix.
    Relaxed,
    /// Use ordinary softmax probabilities for the top-1 component.
    SoftmaxMixture,
}

/// Configuration for the scalar reference implementation of differentiable
/// top-k cross-entropy.
#[derive(Debug, Clone)]
pub struct DifferentiableTopKConfig {
    /// Distribution over top-k objectives. Trailing zero entries are ignored.
    pub p_k: Vec<f64>,
    /// Inverse temperature (steepness) of the sorting relaxation.
    pub inverse_temperature: f64,
    /// Optional number of logits retained before applying the full matrix sort.
    pub m: Option<usize>,
    /// Differentiable sorting relaxation.
    pub relaxation: TopKRelaxation,
    /// Treatment of the top-1 term.
    pub top1_mode: Top1Mode,
}

impl DifferentiableTopKConfig {
    /// Validate the distribution, inverse temperature, and truncation.
    pub fn validate(&self) -> Result<(), DifferentiableTopKConfigError> {
        if self.p_k.is_empty() {
            return Err(DifferentiableTopKConfigError::EmptyDistribution);
        }
        for (index, &probability) in self.p_k.iter().enumerate() {
            if !probability.is_finite() {
                return Err(DifferentiableTopKConfigError::NonFiniteProbability {
                    index,
                    probability,
                });
            }
            if probability < 0.0 {
                return Err(DifferentiableTopKConfigError::NegativeProbability {
                    index,
                    probability,
                });
            }
        }
        let sum: f64 = self.p_k.iter().sum();
        if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err(DifferentiableTopKConfigError::NonNormalizedDistribution { sum });
        }
        if !self.inverse_temperature.is_finite()
            || self.inverse_temperature <= 0.0
            || !(1.0 / self.inverse_temperature).is_finite()
        {
            return Err(DifferentiableTopKConfigError::InvalidInverseTemperature(
                self.inverse_temperature,
            ));
        }

        let effective_k = self
            .p_k
            .iter()
            .rposition(|&probability| probability > 0.0)
            .map_or(0, |index| index + 1);
        if let Some(m) = self.m {
            if m < effective_k {
                return Err(DifferentiableTopKConfigError::InvalidTruncation { m, effective_k });
            }
        }
        Ok(())
    }
}

/// Invalid differentiable top-k cross-entropy configuration.
#[derive(Debug, Clone, Error, PartialEq)]
#[non_exhaustive]
pub enum DifferentiableTopKConfigError {
    /// The distribution contains no top-k positions.
    #[error("p_k must not be empty")]
    EmptyDistribution,
    /// A distribution entry is NaN or infinite.
    #[error("p_k[{index}] must be finite, got {probability}")]
    NonFiniteProbability {
        /// Index of the invalid entry.
        index: usize,
        /// Invalid probability.
        probability: f64,
    },
    /// A distribution entry is negative.
    #[error("p_k[{index}] must be non-negative, got {probability}")]
    NegativeProbability {
        /// Index of the invalid entry.
        index: usize,
        /// Invalid probability.
        probability: f64,
    },
    /// The distribution does not sum to one.
    #[error("p_k must sum to 1 (within {PROBABILITY_SUM_TOLERANCE}), got {sum}")]
    NonNormalizedDistribution {
        /// Observed sum.
        sum: f64,
    },
    /// The inverse temperature is not finite and strictly positive.
    #[error("inverse_temperature must have a finite positive reciprocal, got {0}")]
    InvalidInverseTemperature(f64),
    /// The truncation cannot represent the largest top-k objective with weight.
    #[error("m must be at least the effective k ({effective_k}), got {m}")]
    InvalidTruncation {
        /// Requested truncation.
        m: usize,
        /// Last top-k position with nonzero weight.
        effective_k: usize,
    },
}

/// Invalid input to a differentiable top-k cross-entropy computation.
#[derive(Debug, Clone, Error, PartialEq)]
#[non_exhaustive]
pub enum DifferentiableTopKComputeError {
    /// A sample has no class logits.
    #[error("logits must not be empty")]
    EmptyLogits,
    /// The label does not identify one of the sample's classes.
    #[error("label {label} is out of bounds for {classes} classes")]
    LabelOutOfBounds {
        /// Invalid label.
        label: usize,
        /// Number of available classes.
        classes: usize,
    },
    /// A logit is NaN or infinite.
    #[error("logit at index {index} must be finite, got {value}")]
    NonFiniteLogit {
        /// Index of the invalid logit.
        index: usize,
        /// Invalid logit.
        value: f64,
    },
    /// The configured objective requests a k larger than the class count.
    #[error("effective k ({effective_k}) exceeds the number of classes ({classes})")]
    TooFewClasses {
        /// Last top-k position with nonzero weight.
        effective_k: usize,
        /// Number of available classes.
        classes: usize,
    },
    /// The selected sorting relaxation rejected its input.
    #[error(transparent)]
    Sorting(#[from] SortingError),
}

/// Scalar, reference-faithful differentiable top-k cross-entropy.
///
/// This implementation composes Rankit's full-matrix [`neural_sort`] and
/// [`soft_sort`] operators following Petersen et al.'s loss construction. It
/// returns only an `f64` loss and does not provide automatic differentiation.
/// [`TopKCrossEntropyLoss`] remains available as the legacy lightweight
/// heuristic.
#[derive(Debug, Clone)]
pub struct DifferentiableTopKCrossEntropyLoss {
    config: DifferentiableTopKConfig,
    effective_k: usize,
}

impl DifferentiableTopKCrossEntropyLoss {
    /// Create a loss after validating its configuration.
    pub fn try_new(
        config: DifferentiableTopKConfig,
    ) -> Result<Self, DifferentiableTopKConfigError> {
        config.validate()?;
        let effective_k = config
            .p_k
            .iter()
            .rposition(|&probability| probability > 0.0)
            .map_or(0, |index| index + 1);
        Ok(Self {
            config,
            effective_k,
        })
    }

    /// Compute the loss for one sample.
    pub fn compute(
        &self,
        logits: &[f64],
        label: usize,
    ) -> Result<f64, DifferentiableTopKComputeError> {
        if logits.is_empty() {
            return Err(DifferentiableTopKComputeError::EmptyLogits);
        }
        if label >= logits.len() {
            return Err(DifferentiableTopKComputeError::LabelOutOfBounds {
                label,
                classes: logits.len(),
            });
        }
        if let Some((index, &value)) = logits
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(DifferentiableTopKComputeError::NonFiniteLogit { index, value });
        }
        if self.effective_k > logits.len() {
            return Err(DifferentiableTopKComputeError::TooFewClasses {
                effective_k: self.effective_k,
                classes: logits.len(),
            });
        }

        let (selected, selected_label) = retain_true_label(logits, label, self.config.m);
        let temperature = 1.0 / self.config.inverse_temperature;
        let permutation = match self.config.relaxation {
            TopKRelaxation::NeuralSort => neural_sort(&selected, temperature)?,
            TopKRelaxation::SoftSort => soft_sort(&selected, temperature)?,
        };

        let probability = differentiable_topk_probability(
            &selected,
            selected_label,
            &permutation,
            &self.config.p_k[..self.effective_k],
            self.config.top1_mode,
        );

        // Match the official implementation's bounded affine stabilization.
        Ok(-(probability * (1.0 - 2e-7) + 1e-7).ln())
    }
}

fn differentiable_topk_probability(
    logits: &[f64],
    label: usize,
    permutation: &[Vec<f64>],
    p_k: &[f64],
    top1_mode: Top1Mode,
) -> f64 {
    let mut probability = 0.0;
    for (index, &weight) in p_k.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        let k = index + 1;
        let inclusion = if k == 1 && top1_mode == Top1Mode::SoftmaxMixture {
            softmax_probability(logits, label)
        } else {
            permutation[..k].iter().map(|row| row[label]).sum()
        };
        probability += weight * inclusion;
    }
    probability
}

fn retain_true_label(logits: &[f64], label: usize, m: Option<usize>) -> (Vec<f64>, usize) {
    let Some(m) = m.filter(|&m| m < logits.len()) else {
        return (logits.to_vec(), label);
    };
    let mut false_logits: Vec<(usize, f64)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| *index != label)
        .collect();
    false_logits.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut selected = Vec::with_capacity(m);
    selected.push(logits[label]);
    selected.extend(false_logits.into_iter().take(m - 1).map(|(_, value)| value));
    (selected, 0)
}

fn softmax_probability(logits: &[f64], label: usize) -> f64 {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let denominator: f64 = logits.iter().map(|&value| (value - max).exp()).sum();
    (logits[label] - max).exp() / denominator
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
    /// Create a loss without validating its configuration.
    ///
    /// This compatibility constructor preserves the original API. Prefer
    /// [`Self::try_new`] when configuration can come from user input.
    pub fn new(config: TopKConfig) -> Self {
        Self { config }
    }

    /// Create a loss after validating its configuration.
    pub fn try_new(config: TopKConfig) -> Result<Self, TopKConfigError> {
        config.validate()?;
        Ok(Self { config })
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

    /// Create a validated loss with equal weight across `k` positions.
    pub fn try_uniform(k: usize, temperature: f64) -> Result<Self, TopKConfigError> {
        let weight = 1.0 / k as f64;
        Self::try_new(TopKConfig {
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

    /// Create a validated loss emphasizing top-1 and top-k equally.
    pub fn try_endpoints(k: usize, temperature: f64) -> Result<Self, TopKConfigError> {
        if k == 0 {
            return Err(TopKConfigError::EmptyDistribution);
        }
        let mut p_k = vec![0.0; k];
        p_k[0] = 0.5;
        p_k[k - 1] += 0.5;
        Self::try_new(TopKConfig {
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
    ///
    /// For compatibility, empty logits or an out-of-range label return `0.0`,
    /// and this method does not validate the configuration. Use
    /// [`Self::try_compute`] when those cases should be errors.
    pub fn compute(&self, logits: &[f64], label: usize) -> f64 {
        let n = logits.len();
        if n == 0 || label >= n {
            return 0.0;
        }

        let k = self.config.p_k.len();

        // The paper's default top-1 mode uses ordinary softmax cross-entropy.
        // Besides matching that definition, this keeps `top1()` independent of
        // the soft-ranking temperature and truncation settings.
        if self.config.p_k.as_slice() == [1.0] {
            return softmax_ce(logits, label);
        }

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

    /// Compute the loss for one sample after validating configuration and input.
    pub fn try_compute(&self, logits: &[f64], label: usize) -> Result<f64, TopKComputeError> {
        self.config.validate()?;
        if logits.is_empty() {
            return Err(TopKComputeError::EmptyLogits);
        }
        if label >= logits.len() {
            return Err(TopKComputeError::LabelOutOfBounds {
                label,
                classes: logits.len(),
            });
        }
        Ok(self.compute(logits, label))
    }

    /// Compute the loss for a batch of samples, returning the mean.
    ///
    /// For compatibility, this method processes the shared prefix when batch
    /// and label lengths differ. Use [`Self::try_compute_batch`] to reject a
    /// mismatch.
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

    /// Compute the mean batch loss after validating configuration and inputs.
    pub fn try_compute_batch(
        &self,
        logits_batch: &[Vec<f64>],
        labels: &[usize],
    ) -> Result<f64, TopKComputeError> {
        self.config.validate()?;
        if logits_batch.len() != labels.len() {
            return Err(TopKComputeError::BatchLengthMismatch {
                samples: logits_batch.len(),
                labels: labels.len(),
            });
        }
        if logits_batch.is_empty() {
            return Ok(0.0);
        }

        let total = logits_batch
            .iter()
            .zip(labels)
            .try_fold(0.0, |total, (logits, &label)| {
                self.try_compute(logits, label).map(|loss| total + loss)
            })?;
        Ok(total / logits_batch.len() as f64)
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
    fn test_top1_matches_softmax_cross_entropy() {
        let loss_fn = TopKCrossEntropyLoss::top1();
        let logits = vec![1.25, -0.5, 0.75, 2.0];

        for label in 0..logits.len() {
            let actual = loss_fn.compute(&logits, label);
            let expected = softmax_ce(&logits, label);
            assert!(
                (actual - expected).abs() < 1e-12,
                "top1 loss differs from softmax CE for label {label}: {actual} != {expected}"
            );
        }
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

    #[test]
    fn validated_constructor_accepts_default_config() {
        assert!(TopKCrossEntropyLoss::try_new(TopKConfig::default()).is_ok());
        assert_eq!(
            TopKCrossEntropyLoss::try_uniform(0, 1.0).unwrap_err(),
            TopKConfigError::EmptyDistribution
        );
        assert_eq!(
            TopKCrossEntropyLoss::try_endpoints(0, 1.0).unwrap_err(),
            TopKConfigError::EmptyDistribution
        );
        assert!(TopKCrossEntropyLoss::try_endpoints(1, 1.0).is_ok());
    }

    #[test]
    fn config_rejects_empty_distribution() {
        let error = TopKConfig {
            p_k: vec![],
            temperature: 1.0,
            m: None,
        }
        .validate()
        .unwrap_err();
        assert_eq!(error, TopKConfigError::EmptyDistribution);
    }

    #[test]
    fn config_rejects_non_finite_and_negative_probabilities() {
        let non_finite = TopKConfig {
            p_k: vec![f64::NAN],
            temperature: 1.0,
            m: None,
        }
        .validate()
        .unwrap_err();
        assert!(matches!(
            non_finite,
            TopKConfigError::NonFiniteProbability { index: 0, probability }
                if probability.is_nan()
        ));

        let negative = TopKConfig {
            p_k: vec![1.1, -0.1],
            temperature: 1.0,
            m: None,
        }
        .validate()
        .unwrap_err();
        assert_eq!(
            negative,
            TopKConfigError::NegativeProbability {
                index: 1,
                probability: -0.1,
            }
        );
    }

    #[test]
    fn config_rejects_non_normalized_distribution() {
        let error = TopKConfig {
            p_k: vec![0.25, 0.25],
            temperature: 1.0,
            m: None,
        }
        .validate()
        .unwrap_err();
        assert_eq!(
            error,
            TopKConfigError::NonNormalizedDistribution { sum: 0.5 }
        );
    }

    #[test]
    fn config_rejects_invalid_temperature_and_m() {
        for temperature in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let error = TopKConfig {
                p_k: vec![1.0],
                temperature,
                m: None,
            }
            .validate()
            .unwrap_err();
            assert!(matches!(error, TopKConfigError::InvalidTemperature(value)
                if value == temperature || (value.is_nan() && temperature.is_nan())));
        }

        let error = TopKConfig {
            p_k: vec![0.5, 0.5],
            temperature: 1.0,
            m: Some(1),
        }
        .validate()
        .unwrap_err();
        assert_eq!(error, TopKConfigError::InvalidTruncation { m: 1, k: 2 });
    }

    #[test]
    fn checked_compute_rejects_empty_logits_and_invalid_label() {
        let loss = TopKCrossEntropyLoss::top1();
        assert_eq!(loss.try_compute(&[], 0), Err(TopKComputeError::EmptyLogits));
        assert_eq!(
            loss.try_compute(&[1.0, 2.0], 2),
            Err(TopKComputeError::LabelOutOfBounds {
                label: 2,
                classes: 2,
            })
        );
    }

    #[test]
    fn checked_compute_rejects_compatibility_constructed_invalid_config() {
        let loss = TopKCrossEntropyLoss::new(TopKConfig {
            p_k: vec![0.25],
            temperature: 1.0,
            m: None,
        });
        assert_eq!(
            loss.try_compute(&[1.0], 0),
            Err(TopKComputeError::InvalidConfig(
                TopKConfigError::NonNormalizedDistribution { sum: 0.25 }
            ))
        );
    }

    #[test]
    fn checked_batch_rejects_length_and_sample_mismatches() {
        let loss = TopKCrossEntropyLoss::top1();
        assert_eq!(
            loss.try_compute_batch(&[vec![1.0, 2.0]], &[]),
            Err(TopKComputeError::BatchLengthMismatch {
                samples: 1,
                labels: 0,
            })
        );
        assert_eq!(
            loss.try_compute_batch(&[vec![1.0]], &[1]),
            Err(TopKComputeError::LabelOutOfBounds {
                label: 1,
                classes: 1,
            })
        );
    }

    #[test]
    fn checked_and_compatibility_computations_agree_for_valid_inputs() {
        let loss = TopKCrossEntropyLoss::try_new(TopKConfig {
            p_k: vec![0.5, 0.5],
            temperature: 1.0,
            m: Some(2),
        })
        .unwrap();
        let logits = vec![vec![2.0, 1.0], vec![0.5, 1.5]];
        let labels = vec![0, 1];

        assert_eq!(
            loss.try_compute(&logits[0], labels[0]).unwrap(),
            loss.compute(&logits[0], labels[0])
        );
        assert_eq!(
            loss.try_compute_batch(&logits, &labels).unwrap(),
            loss.compute_batch(&logits, &labels)
        );
    }

    fn reference_loss(
        p_k: Vec<f64>,
        inverse_temperature: f64,
        m: Option<usize>,
        relaxation: TopKRelaxation,
        top1_mode: Top1Mode,
    ) -> DifferentiableTopKCrossEntropyLoss {
        DifferentiableTopKCrossEntropyLoss::try_new(DifferentiableTopKConfig {
            p_k,
            inverse_temperature,
            m,
            relaxation,
            top1_mode,
        })
        .unwrap()
    }

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() <= 1e-12, "{left} != {right}");
    }

    #[test]
    fn reference_loss_matches_direct_full_matrix_composition() {
        let logits = [1.5, -0.25, 0.75, 2.0];
        let p_k = [0.25, 0.0, 0.75];
        for relaxation in [TopKRelaxation::NeuralSort, TopKRelaxation::SoftSort] {
            let matrix = match relaxation {
                TopKRelaxation::NeuralSort => neural_sort(&logits, 0.5).unwrap(),
                TopKRelaxation::SoftSort => soft_sort(&logits, 0.5).unwrap(),
            };
            let probability =
                p_k[0] * matrix[0][2] + p_k[2] * matrix[..3].iter().map(|row| row[2]).sum::<f64>();
            let expected = -(probability * (1.0 - 2e-7) + 1e-7).ln();
            let actual = reference_loss(p_k.to_vec(), 2.0, None, relaxation, Top1Mode::Relaxed)
                .compute(&logits, 2)
                .unwrap();
            assert_close(actual, expected);
        }
    }

    #[test]
    fn explicit_matrix_uses_descending_rows_as_topk_positions() {
        let matrix = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.6, 0.3],
            vec![0.1, 0.3, 0.6],
        ];
        let probability = differentiable_topk_probability(
            &[2.0, 1.0, 0.0],
            1,
            &matrix,
            &[0.25, 0.75],
            Top1Mode::Relaxed,
        );
        assert_close(probability, 0.25 * 0.1 + 0.75 * 0.7);
    }

    #[test]
    fn softmax_mixture_top1_matches_official_stabilized_formula() {
        let logits = [2.0, 0.0, -1.0];
        let probability = softmax_probability(&logits, 0);
        let expected = -(probability * (1.0 - 2e-7) + 1e-7).ln();
        let actual = reference_loss(
            vec![1.0],
            3.0,
            None,
            TopKRelaxation::NeuralSort,
            Top1Mode::SoftmaxMixture,
        )
        .compute(&logits, 0)
        .unwrap();
        assert_close(actual, expected);
    }

    #[test]
    fn truncation_retains_true_label_and_highest_false_logits() {
        let logits = [9.0, 8.0, -20.0, 7.0];
        let truncated = reference_loss(
            vec![0.0, 1.0],
            2.0,
            Some(2),
            TopKRelaxation::SoftSort,
            Top1Mode::Relaxed,
        )
        .compute(&logits, 2)
        .unwrap();
        let explicit = reference_loss(
            vec![0.0, 1.0],
            2.0,
            None,
            TopKRelaxation::SoftSort,
            Top1Mode::Relaxed,
        )
        .compute(&[-20.0, 9.0], 0)
        .unwrap();
        assert_close(truncated, explicit);
    }

    #[test]
    fn trailing_zero_weights_do_not_increase_effective_k() {
        let with_zeros = reference_loss(
            vec![0.5, 0.5, 0.0, 0.0],
            1.5,
            Some(2),
            TopKRelaxation::NeuralSort,
            Top1Mode::Relaxed,
        );
        let trimmed = reference_loss(
            vec![0.5, 0.5],
            1.5,
            Some(2),
            TopKRelaxation::NeuralSort,
            Top1Mode::Relaxed,
        );
        let logits = [0.25, 2.0, -1.0, 0.75];
        assert_close(
            with_zeros.compute(&logits, 0).unwrap(),
            trimmed.compute(&logits, 0).unwrap(),
        );
    }

    #[test]
    fn reference_loss_is_permutation_and_translation_invariant() {
        for relaxation in [TopKRelaxation::NeuralSort, TopKRelaxation::SoftSort] {
            let loss = reference_loss(
                vec![0.2, 0.3, 0.5],
                0.8,
                None,
                relaxation,
                Top1Mode::SoftmaxMixture,
            );
            let original = loss.compute(&[1.0, -2.0, 0.5, 3.0], 2).unwrap();
            let permuted = loss.compute(&[3.0, 0.5, 1.0, -2.0], 1).unwrap();
            let translated = loss.compute(&[18.0, 15.0, 17.5, 20.0], 2).unwrap();
            assert_close(original, permuted);
            assert_close(original, translated);
        }
    }

    #[test]
    fn ties_and_extreme_finite_logits_produce_finite_losses() {
        for relaxation in [TopKRelaxation::NeuralSort, TopKRelaxation::SoftSort] {
            let loss = reference_loss(vec![0.5, 0.5], 4.0, None, relaxation, Top1Mode::Relaxed);
            assert_close(
                loss.compute(&[1.0, 1.0, 0.0], 0).unwrap(),
                loss.compute(&[1.0, 1.0, 0.0], 1).unwrap(),
            );
            assert!(loss.compute(&[1e150, 0.0, -1e150], 1).unwrap().is_finite());
        }
    }

    #[test]
    fn reference_loss_rejects_invalid_configuration_and_inputs() {
        let error = DifferentiableTopKCrossEntropyLoss::try_new(DifferentiableTopKConfig {
            p_k: vec![0.0, 1.0],
            inverse_temperature: 1.0,
            m: Some(1),
            relaxation: TopKRelaxation::SoftSort,
            top1_mode: Top1Mode::Relaxed,
        })
        .unwrap_err();
        assert_eq!(
            error,
            DifferentiableTopKConfigError::InvalidTruncation {
                m: 1,
                effective_k: 2
            }
        );

        let loss = reference_loss(
            vec![1.0],
            1.0,
            None,
            TopKRelaxation::SoftSort,
            Top1Mode::Relaxed,
        );
        assert!(matches!(
            loss.compute(&[0.0, f64::NAN], 0),
            Err(DifferentiableTopKComputeError::NonFiniteLogit { index: 1, value })
                if value.is_nan()
        ));
    }
}
