//! Finite-difference gradient correctness tests.

use crate::gradients::soft_rank_gradient;
#[cfg(feature = "losses")]
use crate::losses::{approx_ndcg_loss, ranknet_loss};
use crate::rank::soft_rank;

const EPSILON: f64 = 1e-4;
const JACOBIAN_TOL: f64 = 1e-2;

/// Numerically compute the Jacobian of soft_rank via central finite differences.
fn numerical_jacobian(values: &[f64], reg: f64) -> Vec<Vec<f64>> {
    let n = values.len();
    let mut jac = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut perturbed = values.to_vec();

        perturbed[j] = values[j] + EPSILON;
        let ranks_plus = soft_rank(&perturbed, reg);

        perturbed[j] = values[j] - EPSILON;
        let ranks_minus = soft_rank(&perturbed, reg);

        for i in 0..n {
            jac[i][j] = (ranks_plus[i] - ranks_minus[i]) / (2.0 * EPSILON);
        }
    }
    jac
}

/// Numerically compute the gradient of a scalar loss w.r.t. inputs.
#[cfg(feature = "losses")]
fn numerical_gradient(values: &[f64], loss_fn: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let n = values.len();
    let mut grad = vec![0.0; n];
    for j in 0..n {
        let mut perturbed = values.to_vec();

        perturbed[j] = values[j] + EPSILON;
        let loss_plus = loss_fn(&perturbed);

        perturbed[j] = values[j] - EPSILON;
        let loss_minus = loss_fn(&perturbed);

        grad[j] = (loss_plus - loss_minus) / (2.0 * EPSILON);
    }
    grad
}

/// Fixed "random-like" test vectors (deterministic, no RNG dependency).
fn test_vectors() -> Vec<Vec<f64>> {
    vec![
        vec![0.3, -1.2, 0.7, 2.1, -0.5],
        vec![1.0, 2.0, 3.0],
        vec![-0.1, 0.4, -0.8, 1.5, 0.2, -1.3, 0.9],
        vec![10.0, -10.0, 0.0, 5.0],
    ]
}

// ---------------------------------------------------------------------------
// Test 1: Finite-difference test for soft_rank_gradient
// ---------------------------------------------------------------------------

#[test]
fn soft_rank_gradient_matches_finite_differences() {
    for values in &test_vectors() {
        for &reg in &[0.5, 1.0, 5.0] {
            let ranks = soft_rank(values, reg);
            let analytical = soft_rank_gradient(values, &ranks, reg);
            let numerical = numerical_jacobian(values, reg);

            let n = values.len();
            for i in 0..n {
                for j in 0..n {
                    let diff = (analytical[i][j] - numerical[i][j]).abs();
                    assert!(
                        diff < JACOBIAN_TOL,
                        "Jacobian mismatch at [{i}][{j}]: analytical={}, numerical={}, diff={diff}, \
                         values={values:?}, reg={reg}",
                        analytical[i][j],
                        numerical[i][j],
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Gradient direction test for RankNet loss
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "losses")]
fn ranknet_gradient_direction_reduces_loss() {
    let relevances: Vec<Vec<f64>> = vec![
        vec![2.0, 0.0, 1.0, 3.0, 1.0],
        vec![3.0, 1.0, 2.0],
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0],
    ];

    for (values, relevance) in test_vectors()
        .into_iter()
        .zip(relevances.iter())
        .filter(|(v, r)| v.len() == r.len())
    {
        let loss_before = ranknet_loss(&values, relevance);
        if loss_before < 1e-10 {
            continue; // already at minimum
        }

        let grad = numerical_gradient(&values, |v| ranknet_loss(v, relevance));

        // Take a small step in the negative gradient direction.
        let step_size = 0.01;
        let stepped: Vec<f64> = values
            .iter()
            .zip(grad.iter())
            .map(|(&v, &g)| v - step_size * g)
            .collect();

        let loss_after = ranknet_loss(&stepped, relevance);
        assert!(
            loss_after < loss_before + 1e-8,
            "RankNet loss did not decrease: before={loss_before}, after={loss_after}, \
             values={values:?}, relevance={relevance:?}",
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: Perfect ranking produces near-zero ApproxNDCG loss and gradients
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "losses")]
fn perfect_ranking_near_zero_approx_ndcg() {
    // Predictions perfectly ordered by relevance (highest prediction = highest relevance).
    let relevance = vec![3.0, 2.0, 1.0, 0.0];
    let predictions = vec![10.0, 7.0, 4.0, 1.0];

    for &reg in &[1.0, 5.0, 10.0] {
        let loss = approx_ndcg_loss(&predictions, &relevance, reg, None);
        assert!(
            loss < 0.15,
            "ApproxNDCG loss should be near zero for perfect ranking, got {loss} (reg={reg})",
        );
    }
}

#[test]
#[cfg(feature = "losses")]
fn perfect_ranking_near_zero_gradients() {
    // When the ranking is already perfect, the gradient of the loss should be
    // small -- there is little incentive to move any score.
    let relevance = vec![3.0, 2.0, 1.0, 0.0];
    let predictions = vec![10.0, 7.0, 4.0, 1.0];
    let reg = 5.0;

    let grad = numerical_gradient(&predictions, |p| approx_ndcg_loss(p, &relevance, reg, None));

    let max_grad = grad.iter().map(|g| g.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_grad < 0.1,
        "Gradient magnitude should be near zero for perfect ranking, got max |grad|={max_grad}, \
         grad={grad:?}",
    );
}
