//! Differentiable ranking heuristics inspired by published methods.
//!
//! Each method has O(n^2) complexity and normalizes to [0, n-1] range.
//!
//! The sigmoid implementation is the default.
//!
//! The paper-named functions and [`RankingMethod`] variants are retained for
//! compatibility. They are pairwise heuristics, not implementations of the
//! matrix-valued operators in [`crate::sorting`].

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

/// Pairwise logistic soft ranks parameterized by smoothing temperature.
///
/// Lower positive temperatures produce sharper pairwise comparisons. This is
/// a scalar-rank heuristic; use [`crate::neural_sort`] for NeuralSort.
pub fn pairwise_logistic_rank(values: &[f64], temperature: f64) -> Vec<f64> {
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

/// Compatibility name for [`pairwise_logistic_rank`].
///
/// Despite its historical name, this does not implement NeuralSort.
pub fn soft_rank_neural_sort(values: &[f64], temperature: f64) -> Vec<f64> {
    pairwise_logistic_rank(values, temperature)
}

/// Pairwise logistic approximation to Gaussian-smoothed ranks.
///
/// This uses a scaled logistic CDF in place of the normal CDF. It is not the
/// exact Gaussian SoftRank operator of Taylor et al.
pub fn logistic_gaussian_rank_approximation(values: &[f64], sigma: f64) -> Vec<f64> {
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

/// Compatibility name for [`logistic_gaussian_rank_approximation`].
pub fn soft_rank_probabilistic(values: &[f64], sigma: f64) -> Vec<f64> {
    logistic_gaussian_rank_approximation(values, sigma)
}

/// Compatibility name for [`soft_rank_sigmoid`].
///
/// Despite its historical name, this does not implement SmoothI.
pub fn soft_rank_smooth_i(values: &[f64], alpha: f64) -> Vec<f64> {
    soft_rank_sigmoid(values, alpha)
}

/// Enum for selecting ranking method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankingMethod {
    /// Sigmoid-based (default).
    Sigmoid,
    /// Compatibility variant for [`pairwise_logistic_rank`], not NeuralSort.
    NeuralSort,
    /// Compatibility variant for [`logistic_gaussian_rank_approximation`].
    Probabilistic,
    /// Compatibility variant for [`soft_rank_sigmoid`], not SmoothI.
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
    fn compatibility_names_have_exact_algebraic_identities() {
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        assert_eq!(
            soft_rank_neural_sort(&values, 0.7),
            pairwise_logistic_rank(&values, 0.7)
        );
        assert_eq!(
            soft_rank_probabilistic(&values, 0.7),
            logistic_gaussian_rank_approximation(&values, 0.7)
        );
        assert_eq!(
            soft_rank_smooth_i(&values, 0.7),
            soft_rank_sigmoid(&values, 0.7)
        );
        assert_eq!(
            RankingMethod::NeuralSort.compute(&values, 0.7),
            pairwise_logistic_rank(&values, 0.7)
        );
    }
}
