//! Differentiable ranking operations using smooth relaxation.
//!
//! This module implements the sigmoid-based approach to differentiable ranking.
//! O(n^2) complexity, suitable for n < 1000.
//!
//! # Algorithm
//!
//! The rank of element `i` is computed as:
//! ```text
//! rank[i] = (sum_{j != i} sigmoid(alpha * (values[i] - values[j]))) / valid_comparisons * (n-1)
//! ```
//!
//! where `alpha = regularization_strength` controls the sigmoid sharpness.
//!
//! Ranks are normalized to **[0, n-1]** range (0 = lowest, n-1 = highest).

/// Compute soft ranks for a vector of values.
///
/// Uses a smooth relaxation of the discrete ranking operation, enabling
/// gradient flow through the ranking.
///
/// # Arguments
///
/// * `values` - Input values to rank
/// * `regularization_strength` - Temperature parameter controlling sharpness
///   (higher = sharper, more discrete-like behavior)
///
/// # Returns
///
/// Vector of soft ranks in [0, n-1] range.
///
/// # Example
///
/// ```rust
/// use rankit::soft_rank;
///
/// let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
/// let ranks = soft_rank(&values, 10.0);
/// // ranks approach [4.0, 0.0, 1.0, 3.0, 2.0]
/// assert!(ranks[1] < ranks[2]);
/// assert!(ranks[2] < ranks[4]);
/// assert!(ranks[4] < ranks[3]);
/// assert!(ranks[3] < ranks[0]);
/// ```
///
/// # Edge Cases
///
/// - **Empty input**: Returns empty vector
/// - **Single element**: Returns `[0.0]`
/// - **NaN/Inf values**: Returns NaN for that element's rank
/// - **All values identical**: All elements receive rank `(n-1)/2`
pub fn soft_rank(values: &[f64], regularization_strength: f64) -> Vec<f64> {
    let n = values.len();

    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let finite_count: usize = values.iter().filter(|&&v| v.is_finite()).count();

    if finite_count == 0 {
        return vec![f64::NAN; n];
    }

    let mut ranks = vec![0.0; n];

    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }

        let mut sum = 0.0;
        let mut valid_comparisons = 0;

        for j in 0..i {
            if values[j].is_finite() {
                let diff = values[i] - values[j];
                sum += sigmoid(diff * regularization_strength);
                valid_comparisons += 1;
            }
        }
        for j in (i + 1)..n {
            if values[j].is_finite() {
                let diff = values[i] - values[j];
                sum += sigmoid(diff * regularization_strength);
                valid_comparisons += 1;
            }
        }

        if valid_comparisons > 0 {
            ranks[i] = (sum / valid_comparisons as f64) * (n - 1) as f64;
        } else {
            ranks[i] = 0.0;
        }
    }

    ranks
}

/// Numerically stable sigmoid: sigma(x) = 1 / (1 + exp(-x))
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x > 500.0 {
        return 1.0;
    }
    if x < -500.0 {
        return 0.0;
    }

    if x > 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_rank_basic() {
        let values = vec![1.0, 2.0, 3.0];
        let ranks = soft_rank(&values, 10.0);

        assert!(ranks[0] < ranks[1]);
        assert!(ranks[1] < ranks[2]);
    }

    #[test]
    fn test_soft_rank_preserves_ordering() {
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let ranks = soft_rank(&values, 1.0);

        assert!(ranks[1] < ranks[2]);
        assert!(ranks[2] < ranks[4]);
        assert!(ranks[4] < ranks[3]);
        assert!(ranks[3] < ranks[0]);
    }
}
