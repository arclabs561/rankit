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

    #[test]
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
