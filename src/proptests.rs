//! Property tests for core ranking operations.

use proptest::prelude::*;

proptest! {
    #[test]
    fn soft_rank_preserves_ordering(
        a in -100.0..100.0_f64,
        b in -100.0..100.0_f64,
        reg in 0.1..10.0_f64,
    ) {
        let values = vec![a, b];
        let ranks = crate::rank::soft_rank(&values, reg);

        if a > b + 1e-6 {
            prop_assert!(ranks[0] > ranks[1], "a={}, b={}, ranks={:?}", a, b, ranks);
        } else if b > a + 1e-6 {
            prop_assert!(ranks[1] > ranks[0], "a={}, b={}, ranks={:?}", a, b, ranks);
        }
    }

    #[test]
    fn soft_rank_in_range(
        values in prop::collection::vec(-100.0..100.0_f64, 1..20),
        reg in 0.1..10.0_f64,
    ) {
        let ranks = crate::rank::soft_rank(&values, reg);
        let n = values.len();

        for &r in &ranks {
            prop_assert!(r >= -0.01, "rank {} below 0", r);
            prop_assert!(r <= (n - 1) as f64 + 0.01, "rank {} above n-1={}", r, n - 1);
        }
    }

    #[test]
    fn sigmoid_bounded(x in -1000.0..1000.0_f64) {
        let s = crate::rank::sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&s), "sigmoid({}) = {} out of [0,1]", x, s);
    }

    // -----------------------------------------------------------------------
    // Natural gradient property tests
    // -----------------------------------------------------------------------

    /// Fisher information matrix is symmetric for any probability distribution.
    #[test]
    fn fisher_is_symmetric(
        raw in prop::collection::vec(0.1_f64..10.0, 2..10),
    ) {
        // Normalize to valid probability distribution.
        let sum: f64 = raw.iter().sum();
        let probs: Vec<f64> = raw.iter().map(|&x| x / sum).collect();
        let n = probs.len();

        let fisher = crate::gradients::natural::fisher_information_softmax(&probs);
        for i in 0..n {
            for j in i+1..n {
                let diff = (fisher[i * n + j] - fisher[j * n + i]).abs();
                prop_assert!(diff < 1e-14, "Not symmetric at [{i}][{j}]: diff={diff}");
            }
        }
    }

    /// Fisher information matrix is PSD: v^T F v >= 0 for random v.
    #[test]
    fn fisher_is_psd(
        raw in prop::collection::vec(0.1_f64..10.0, 2..8),
        v in prop::collection::vec(-10.0..10.0_f64, 2..8),
    ) {
        if raw.len() != v.len() {
            return Ok(());
        }
        let sum: f64 = raw.iter().sum();
        let probs: Vec<f64> = raw.iter().map(|&x| x / sum).collect();
        let n = probs.len();

        let fisher = crate::gradients::natural::fisher_information_softmax(&probs);
        let mut vtfv = 0.0;
        for i in 0..n {
            for j in 0..n {
                vtfv += v[i] * fisher[i * n + j] * v[j];
            }
        }
        prop_assert!(vtfv >= -1e-10, "v^T F v = {vtfv} < 0");
    }

    /// Natural gradient is finite for all valid inputs.
    #[test]
    fn natural_gradient_is_finite(
        raw_probs in prop::collection::vec(0.1_f64..10.0, 2..8),
        grad in prop::collection::vec(-5.0..5.0_f64, 2..8),
    ) {
        if raw_probs.len() != grad.len() {
            return Ok(());
        }
        let sum: f64 = raw_probs.iter().sum();
        let probs: Vec<f64> = raw_probs.iter().map(|&x| x / sum).collect();

        let nat_grad = crate::gradients::natural::natural_gradient_softmax(&grad, &probs);

        for (i, &ng) in nat_grad.iter().enumerate() {
            prop_assert!(ng.is_finite(), "natural_grad[{i}] is not finite: {ng}");
        }
    }

    /// Fisher-natural gradient identity: g^T F^{-1} g = sum(g_i^2 / p_i) - (sum g_i)^2.
    /// Verify by computing g^T * nat_grad (which equals g^T F^{-1} g).
    #[test]
    fn natural_gradient_quadratic_form(
        raw_probs in prop::collection::vec(0.1_f64..10.0, 2..8),
        grad in prop::collection::vec(-5.0..5.0_f64, 2..8),
    ) {
        if raw_probs.len() != grad.len() {
            return Ok(());
        }
        let sum: f64 = raw_probs.iter().sum();
        let probs: Vec<f64> = raw_probs.iter().map(|&x| x / sum).collect();

        let nat_grad = crate::gradients::natural::natural_gradient_softmax(&grad, &probs);

        // g^T * nat_grad
        let dot: f64 = grad.iter().zip(nat_grad.iter()).map(|(&g, &ng)| g * ng).sum();

        // Expected: sum(g_i^2 / p_i) - (sum g_i)^2
        let sum_g: f64 = grad.iter().sum();
        let expected: f64 = grad.iter().zip(probs.iter()).map(|(&g, &p)| g * g / p).sum::<f64>() - sum_g * sum_g;

        prop_assert!(
            (dot - expected).abs() < 1e-6,
            "g^T nat_grad ({dot}) != expected ({expected})"
        );
    }

    #[test]
    #[cfg(feature = "losses")]
    fn ranknet_loss_nonnegative(
        predictions in prop::collection::vec(-10.0..10.0_f64, 2..10),
        relevance in prop::collection::vec(0.0..5.0_f64, 2..10),
    ) {
        if predictions.len() != relevance.len() {
            return Ok(());
        }

        let loss = crate::losses::ranknet_loss(&predictions, &relevance);
        prop_assert!(loss >= 0.0, "ranknet_loss should be >= 0, got {}", loss);
    }
}
