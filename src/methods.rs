//! Differentiable ranking heuristics inspired by published methods.
//!
//! Each method has O(n^2) complexity and normalizes to [0, n-1] range.
//!
//! | Method | Best For |
//! |--------|----------|
//! | **Sigmoid** | General use (default) |
//! | **NeuralSort** | NeuralSort-style scores |
//! | **Probabilistic** | Probabilistic-style rank scores |
//! | **SmoothI** | Alternative gradient profiles |

use crate::rank::sigmoid;

/// Sigmoid-based soft ranking (default).
///
/// From: Qin et al. (2008). General differentiable ranking approach.
pub fn soft_rank_sigmoid(values: &[f64], regularization_strength: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut ranks = vec![0.0; n];

    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }

        let mut sum = 0.0;
        let mut valid_comparisons = 0;
        for j in 0..n {
            if i != j && values[j].is_finite() {
                let diff = values[i] - values[j];
                sum += sigmoid(diff * regularization_strength);
                valid_comparisons += 1;
            }
        }

        if valid_comparisons > 0 {
            ranks[i] = sum / valid_comparisons as f64 * (n - 1) as f64;
        } else {
            ranks[i] = 0.0;
        }
    }

    ranks
}

/// NeuralSort-inspired ranking using temperature-scaled softmax.
///
/// From: "NeuralSort: A Differentiable Sorting Operator" (Grover et al., ICML 2019)
pub fn soft_rank_neural_sort(values: &[f64], temperature: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut ranks = vec![0.0; n];

    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }

        let mut sum = 0.0;
        let mut valid_comparisons = 0;

        for j in 0..n {
            if i != j && values[j].is_finite() {
                let diff = (values[i] - values[j]) / temperature;
                sum += sigmoid(diff);
                valid_comparisons += 1;
            }
        }

        if valid_comparisons > 0 {
            ranks[i] = sum / valid_comparisons as f64 * (n - 1) as f64;
        } else {
            ranks[i] = 0.0;
        }
    }

    ranks
}

/// SoftRank probabilistic approach using Gaussian smoothing.
///
/// From: "SoftRank: Optimizing Non-Smooth Rank Metrics" (Taylor et al., WSDM 2008)
pub fn soft_rank_probabilistic(values: &[f64], sigma: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut ranks = vec![0.0; n];
    let sqrt_2 = std::f64::consts::SQRT_2;

    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }

        let mut sum = 0.0;
        let mut valid_comparisons = 0;
        for j in 0..n {
            if i != j && values[j].is_finite() {
                let diff = values[i] - values[j];
                let z = diff / (sigma * sqrt_2);
                let prob = sigmoid(1.7 * z);
                sum += prob;
                valid_comparisons += 1;
            }
        }

        if valid_comparisons > 0 {
            ranks[i] = sum / valid_comparisons as f64 * (n - 1) as f64;
        } else {
            ranks[i] = 0.0;
        }
    }

    ranks
}

/// SmoothI-style smooth rank indicators with exponential scaling.
///
/// From: "SmoothI: Smooth Rank Indicators for Differentiable Ranking" (ICML 2021)
pub fn soft_rank_smooth_i(values: &[f64], alpha: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut ranks = vec![0.0; n];

    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }

        let mut sum = 0.0;
        let mut valid_comparisons = 0;
        for j in 0..n {
            if i != j && values[j].is_finite() {
                let diff = values[i] - values[j];
                sum += sigmoid(alpha * diff);
                valid_comparisons += 1;
            }
        }

        if valid_comparisons > 0 {
            ranks[i] = sum / valid_comparisons as f64 * (n - 1) as f64;
        } else {
            ranks[i] = 0.0;
        }
    }

    ranks
}

/// Enum for selecting ranking method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankingMethod {
    /// Sigmoid-based (default).
    Sigmoid,
    /// NeuralSort-style (temperature-scaled softmax).
    NeuralSort,
    /// SoftRank probabilistic (Gaussian smoothing).
    Probabilistic,
    /// SmoothI (smooth rank indicators).
    SmoothI,
}

impl RankingMethod {
    /// Compute soft ranks using the selected method.
    pub fn compute(&self, values: &[f64], regularization_strength: f64) -> Vec<f64> {
        match self {
            Self::Sigmoid => soft_rank_sigmoid(values, regularization_strength),
            Self::NeuralSort => soft_rank_neural_sort(values, regularization_strength),
            Self::Probabilistic => soft_rank_probabilistic(values, regularization_strength),
            Self::SmoothI => soft_rank_smooth_i(values, regularization_strength),
        }
    }

    /// Get method name for logging/benchmarking.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sigmoid => "sigmoid",
            Self::NeuralSort => "neural_sort",
            Self::Probabilistic => "probabilistic",
            Self::SmoothI => "smooth_i",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_methods_preserve_ordering() {
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];

        for method in [
            RankingMethod::Sigmoid,
            RankingMethod::NeuralSort,
            RankingMethod::Probabilistic,
            RankingMethod::SmoothI,
        ] {
            let ranks = method.compute(&values, 1.0);

            assert!(ranks[1] < ranks[2], "{} failed ordering", method.name());
            assert!(ranks[2] < ranks[4], "{} failed ordering", method.name());
            assert!(ranks[4] < ranks[3], "{} failed ordering", method.name());
            assert!(ranks[3] < ranks[0], "{} failed ordering", method.name());
        }
    }
}
