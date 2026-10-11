//! Differentiable ranking operations using smooth relaxation.
//!
//! This module implements the sigmoid-based approach to differentiable ranking.
//! O(n^2) complexity, suitable for n < 1000.
//!
//! # Heuristic
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
/// * `regularization_strength` - Inverse temperature controlling sharpness
///   (higher = sharper, more discrete-like behavior). This multiplies the
///   score differences, so it runs opposite to the regularization `ε` of
///   Blondel et al. 2020 (and `fynch`), where larger `ε` is smoother.
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
/// - **Infinite strength**: comparisons are hard; tied values each count as
///   half a win, so ties share their average rank
pub fn soft_rank(values: &[f64], regularization_strength: f64) -> Vec<f64> {
    pairwise_soft_rank(values, 1.0 / regularization_strength)
}

/// Pairwise logistic soft ranks in rankit's convention: ascending, in
/// `[0, n-1]`, with `temperature = 1 / regularization_strength`.
///
/// The pairwise sum is `fynch::soft_rank` on the finite entries. fynch ranks
/// descending in `[1, m]`, so `m - fynch_rank` is the number of soft wins. Non-finite
/// entries get NaN and are left out of the comparisons; the finite ranks are
/// then rescaled from `[0, m-1]` to `[0, n-1]`.
pub(crate) fn pairwise_soft_rank(values: &[f64], temperature: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    // A negative temperature reverses every comparison, which equals ranking
    // the negated values at the positive temperature. fynch rejects zero
    // (infinite strength); the smallest normal temperature makes every
    // comparison between non-subnormal gaps hard.
    let (sign, temperature) = if temperature.is_sign_negative() {
        (-1.0, -temperature)
    } else {
        (1.0, temperature)
    };
    let temperature = if temperature == 0.0 {
        f64::MIN_POSITIVE
    } else {
        temperature
    };

    let finite: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| sign * v)
        .collect();
    let m = finite.len();
    let finite_ranks: Vec<f64> = match m {
        0 => return vec![f64::NAN; n],
        1 => vec![0.0],
        _ => match fynch::soft_rank(&finite, temperature) {
            Ok(descending) => {
                let scale = (n - 1) as f64 / (m - 1) as f64;
                descending
                    .into_iter()
                    .map(|rank| (m as f64 - rank) * scale)
                    .collect()
            }
            // Unreachable: the input is non-empty and the temperature is
            // positive or NaN, the only cases fynch accepts.
            Err(_) => vec![f64::NAN; m],
        },
    };

    let mut finite_ranks = finite_ranks.into_iter();
    values
        .iter()
        .map(|v| {
            if v.is_finite() {
                finite_ranks.next().unwrap_or(f64::NAN)
            } else {
                f64::NAN
            }
        })
        .collect()
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
