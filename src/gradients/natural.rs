//! Natural gradient computation for softmax-based ranking losses.
//!
//! The natural gradient preconditions the vanilla gradient by the inverse Fisher
//! information matrix of the score distribution. For softmax-parameterized
//! distributions, the Fisher information has a closed-form inverse, avoiding
//! explicit matrix inversion.
//!
//! $$F = \mathrm{diag}(p) - p p^\top$$
//!
//! The inverse (on the tangent space orthogonal to the all-ones vector) gives:
//!
//! $$\tilde{g}_i = g_i / p_i - \sum_j g_j$$
//!
//! Ref: Martens 2020, "New Insights and Perspectives on the Natural Gradient
//! Method" (JMLR 21:1-76).

/// Compute the natural gradient for softmax-based ranking losses.
///
/// Given raw gradients `grad` and softmax probabilities `softmax_probs`,
/// returns the natural gradient using the closed-form inverse of the
/// softmax Fisher information matrix.
///
/// The formula is: `natural_grad_i = grad_i / p_i - sum_j(grad_j)`
///
/// # Panics
///
/// Panics if `grad` and `softmax_probs` have different lengths.
pub fn natural_gradient_softmax(grad: &[f64], softmax_probs: &[f64]) -> Vec<f64> {
    assert_eq!(
        grad.len(),
        softmax_probs.len(),
        "grad and softmax_probs must have the same length"
    );

    let n = grad.len();
    if n == 0 {
        return vec![];
    }

    let grad_sum: f64 = grad.iter().sum();

    grad.iter()
        .zip(softmax_probs.iter())
        .map(|(&g, &p)| {
            if p < 1e-30 {
                // Avoid division by near-zero probability.
                // Items with vanishing probability get zero natural gradient.
                0.0
            } else {
                g / p - grad_sum
            }
        })
        .collect()
}

/// Compute the Fisher information matrix for a softmax distribution.
///
/// Returns the `n x n` matrix `F = diag(p) - p * p^T`, flattened in
/// row-major order.
///
/// Properties:
/// - Symmetric and positive semi-definite.
/// - Exactly one zero eigenvalue (eigenvector = all-ones), since softmax
///   probabilities sum to 1.
/// - Rank = `n - 1`.
pub fn fisher_information_softmax(softmax_probs: &[f64]) -> Vec<f64> {
    let n = softmax_probs.len();
    let mut fisher = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            let val = if i == j {
                softmax_probs[i] * (1.0 - softmax_probs[i])
            } else {
                -softmax_probs[i] * softmax_probs[j]
            };
            fisher[i * n + j] = val;
        }
    }

    fisher
}

/// Apply natural gradient preconditioning to any ranking loss.
///
/// 1. Computes softmax probabilities from `scores`.
/// 2. Calls `loss_grad_fn` to get the vanilla gradient.
/// 3. Preconditions the gradient with the inverse Fisher information.
///
/// This is the main entry point for natural gradient ranking optimization.
pub fn with_natural_gradient<F>(loss_grad_fn: F, scores: &[f64]) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = scores.len();
    if n == 0 {
        return vec![];
    }

    let probs = softmax(scores);
    let grad = loss_grad_fn(scores);
    natural_gradient_softmax(&grad, &probs)
}

/// Stable softmax computation (subtract max for numerical stability).
fn softmax(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return vec![];
    }

    let max_s = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f64 = exps.iter().sum();

    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform distribution: natural gradient == vanilla gradient (up to
    /// a constant offset from the sum term). For uniform p_i = 1/n,
    /// natural_grad_i = n * g_i - sum(g). When g is zero-mean, this is
    /// just n * g_i.
    #[test]
    fn uniform_probs_scales_gradient() {
        let n = 4;
        let probs = vec![0.25; n];
        // Zero-mean gradient so the sum term vanishes.
        let grad = vec![0.3, -0.1, 0.2, -0.4];

        let nat_grad = natural_gradient_softmax(&grad, &probs);

        // Each entry should be g_i / (1/n) - sum(g) = n*g_i - 0 = n*g_i
        let grad_sum: f64 = grad.iter().sum();
        for i in 0..n {
            let expected = grad[i] / probs[i] - grad_sum;
            assert!(
                (nat_grad[i] - expected).abs() < 1e-10,
                "i={i}: got {}, expected {expected}",
                nat_grad[i]
            );
        }
    }

    /// Fisher information matrix is symmetric.
    #[test]
    fn fisher_is_symmetric() {
        let probs = vec![0.1, 0.3, 0.4, 0.2];
        let n = probs.len();
        let fisher = fisher_information_softmax(&probs);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (fisher[i * n + j] - fisher[j * n + i]).abs() < 1e-15,
                    "Fisher not symmetric at [{i}][{j}]"
                );
            }
        }
    }

    /// Fisher information matrix is PSD: all eigenvalues >= 0.
    /// For a 3x3 case, verify via characteristic polynomial or direct
    /// computation of v^T F v for random v.
    #[test]
    fn fisher_is_psd() {
        let probs = vec![0.2, 0.5, 0.3];
        let n = probs.len();
        let fisher = fisher_information_softmax(&probs);

        // Check v^T F v >= 0 for several test vectors.
        let test_vecs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, 0.0],
            vec![1.0, -0.5, -0.5],
            vec![-2.0, 1.0, 1.0],
        ];

        for v in &test_vecs {
            let mut vtfv = 0.0;
            for i in 0..n {
                for j in 0..n {
                    vtfv += v[i] * fisher[i * n + j] * v[j];
                }
            }
            assert!(vtfv >= -1e-15, "v^T F v = {vtfv} < 0 for v = {v:?}");
        }
    }

    /// Fisher has one zero eigenvalue (the all-ones eigenvector).
    /// F * [1,1,...,1] should be zero.
    #[test]
    fn fisher_null_space_is_ones() {
        let probs = vec![0.15, 0.35, 0.25, 0.25];
        let n = probs.len();
        let fisher = fisher_information_softmax(&probs);
        let ones = vec![1.0; n];

        for i in 0..n {
            let mut row_dot = 0.0;
            for j in 0..n {
                row_dot += fisher[i * n + j] * ones[j];
            }
            assert!(
                row_dot.abs() < 1e-15,
                "F * ones is not zero at row {i}: got {row_dot}"
            );
        }
    }

    /// Natural gradient scales inversely with probability: rare items
    /// (low p_i) get larger update magnitudes than frequent items (high p_i),
    /// for the same raw gradient.
    #[test]
    fn rare_items_get_larger_updates() {
        let probs = vec![0.05, 0.45, 0.50];
        // Same raw gradient for all items.
        let grad = vec![1.0, 1.0, 1.0];

        let nat_grad = natural_gradient_softmax(&grad, &probs);

        // Item 0 (rare, p=0.05) should have larger magnitude than item 2 (common, p=0.50).
        assert!(
            nat_grad[0].abs() > nat_grad[2].abs(),
            "Rare item grad {} should exceed common item grad {}",
            nat_grad[0].abs(),
            nat_grad[2].abs()
        );
    }

    /// 3-item ranking: verify natural gradient on a concrete example.
    #[test]
    fn three_item_concrete() {
        let scores = vec![2.0, 1.0, 0.0];
        let probs = softmax(&scores);

        // Synthetic gradient (e.g., from a ranking loss).
        let grad = vec![-0.5, 0.2, 0.3];
        let nat_grad = natural_gradient_softmax(&grad, &probs);

        // Verify manually: sum(grad) = 0.0, so natural_grad_i = grad_i / p_i.
        let grad_sum: f64 = grad.iter().sum();
        for i in 0..3 {
            let expected = grad[i] / probs[i] - grad_sum;
            assert!(
                (nat_grad[i] - expected).abs() < 1e-10,
                "i={i}: got {}, expected {expected}",
                nat_grad[i]
            );
        }
    }

    /// with_natural_gradient integrates softmax + gradient + preconditioning.
    #[test]
    fn with_natural_gradient_integration() {
        let scores = vec![1.0, 2.0, 3.0, 0.5];

        let result = with_natural_gradient(
            |s| {
                // Simple loss gradient: d/ds_i (sum s_i^2) = 2*s_i
                s.iter().map(|&x| 2.0 * x).collect()
            },
            &scores,
        );

        assert_eq!(result.len(), scores.len());
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    /// Empty input returns empty output.
    #[test]
    fn empty_input() {
        assert!(natural_gradient_softmax(&[], &[]).is_empty());
        assert!(fisher_information_softmax(&[]).is_empty());
        assert!(with_natural_gradient(|_| vec![], &[]).is_empty());
    }

    /// Single element: Fisher is [p*(1-p)], natural gradient is g/p - g.
    #[test]
    fn single_element() {
        let probs = vec![1.0]; // softmax of a single element is always 1.0
        let grad = vec![0.5];
        let nat_grad = natural_gradient_softmax(&grad, &probs);
        // g/p - sum(g) = 0.5/1.0 - 0.5 = 0.0
        assert!((nat_grad[0]).abs() < 1e-15);

        let fisher = fisher_information_softmax(&probs);
        // F = [1*(1-1)] = [0]
        assert!((fisher[0]).abs() < 1e-15);
    }
}
