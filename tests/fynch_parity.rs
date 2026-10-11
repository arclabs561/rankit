use proptest::prelude::*;

fn assert_close(actual: f64, expected: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff < 1e-10,
        "actual={actual}, expected={expected}, diff={diff}"
    );
}

#[test]
fn fynch_soft_rank_maps_to_rankit_convention() {
    let cases = [
        vec![5.0, 1.0, 2.0, 4.0, 3.0],
        vec![1.0, 1.0, 1.0],
        vec![-3.0, 0.0, 7.5, 2.25],
        vec![42.0],
    ];

    for values in cases {
        for temperature in [0.25, 0.5, 1.0, 2.0] {
            let fynch_ranks = fynch::soft_rank(&values, temperature).unwrap();
            let rankit_ranks = rankit::soft_rank(&values, 1.0 / temperature);
            let n = values.len() as f64;

            for (from_fynch, from_rankit) in fynch_ranks.iter().zip(&rankit_ranks) {
                assert_close(n - from_fynch, *from_rankit);
            }
        }
    }
}

proptest! {
    #[test]
    fn finite_fynch_soft_rank_maps_to_rankit(
        values in prop::collection::vec(-10.0_f64..10.0, 1..16),
        temperature in 0.2_f64..10.0,
    ) {
        let fynch_ranks = fynch::soft_rank(&values, temperature).unwrap();
        let rankit_ranks = rankit::soft_rank(&values, 1.0 / temperature);
        let n = values.len() as f64;

        for (from_fynch, from_rankit) in fynch_ranks.iter().zip(&rankit_ranks) {
            prop_assert!(
                (n - from_fynch - from_rankit).abs() < 1e-10,
                "values={values:?}, temperature={temperature}, fynch={fynch_ranks:?}, rankit={rankit_ranks:?}"
            );
        }
    }
}

#[test]
fn empty_input_semantics_are_not_interchangeable() {
    assert!(rankit::soft_rank(&[], 1.0).is_empty());
    assert!(matches!(
        fynch::soft_rank(&[], 1.0),
        Err(fynch::Error::EmptyInput)
    ));
}

#[test]
fn non_finite_input_semantics_are_not_interchangeable() {
    let values = [1.0, f64::NAN, 2.0];

    let rankit_ranks = rankit::soft_rank(&values, 2.0);
    assert!(rankit_ranks[0].is_finite());
    assert!(rankit_ranks[1].is_nan());
    assert!(rankit_ranks[2].is_finite());

    let fynch_ranks = fynch::soft_rank(&values, 0.5).unwrap();
    assert!(fynch_ranks.iter().all(|rank| rank.is_nan()));
}

/// The pairwise loop `rankit::soft_rank` ran before it delegated to fynch,
/// kept verbatim as an oracle for the delegation.
fn pre_delegation_soft_rank(values: &[f64], strength: f64) -> Vec<f64> {
    fn sigmoid(x: f64) -> f64 {
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
        let mut valid = 0;
        for j in 0..n {
            if i != j && values[j].is_finite() {
                sum += sigmoid((values[i] - values[j]) * strength);
                valid += 1;
            }
        }
        ranks[i] = if valid > 0 {
            sum / valid as f64 * (n - 1) as f64
        } else {
            0.0
        };
    }
    ranks
}

fn assert_same_ranks(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (a, e) in actual.iter().zip(expected) {
        if e.is_nan() {
            assert!(
                a.is_nan(),
                "{context}: actual={actual:?}, expected={expected:?}"
            );
        } else {
            assert!(
                (a - e).abs() < 1e-10,
                "{context}: actual={actual:?}, expected={expected:?}"
            );
        }
    }
}

const DELEGATION_CASES: &[&[f64]] = &[
    &[5.0, 1.0, 2.0, 4.0, 3.0],
    &[2.0, 1.0, 2.0, 3.0, 1.0],
    &[7.0, 7.0, 7.0, 7.0],
    &[0.3, -1.2, 0.3, 4.0, -1.2, 0.0],
    &[1.0, f64::NAN, 2.0, 2.0],
    &[f64::INFINITY, 3.0, f64::NEG_INFINITY, 1.0, 3.0],
    &[f64::NAN, 4.0, f64::NAN],
    &[f64::NAN, f64::NAN],
    &[f64::NAN],
    &[],
];

#[test]
fn soft_rank_matches_pre_delegation_loop() {
    for values in DELEGATION_CASES {
        for strength in [-3.0, -0.5, 0.0, 0.1, 1.0, 4.0, 50.0] {
            let context = format!("values={values:?}, strength={strength}");
            let expected = pre_delegation_soft_rank(values, strength);
            assert_same_ranks(&rankit::soft_rank(values, strength), &expected, &context);
            assert_same_ranks(
                &rankit::methods::soft_rank_sigmoid(values, strength),
                &expected,
                &context,
            );
            assert_same_ranks(
                &rankit::optimized::soft_rank_optimized(values, strength, false),
                &expected,
                &context,
            );
            if strength != 0.0 {
                assert_same_ranks(
                    &rankit::pairwise_logistic_rank(values, 1.0 / strength),
                    &expected,
                    &context,
                );
            }
        }
    }
}

#[test]
fn soft_rank_with_ties_equals_converted_fynch_ranks() {
    // Ties included: tied values must get identical ranks from both crates.
    let values = [2.0, 1.0, 2.0, 3.0, 1.0];
    for temperature in [0.1, 0.5, 2.0] {
        let fynch_ranks = fynch::soft_rank(&values, temperature).unwrap();
        let rankit_ranks = rankit::soft_rank(&values, 1.0 / temperature);
        let converted: Vec<f64> = fynch_ranks.iter().map(|r| 5.0 - r).collect();
        assert_same_ranks(&rankit_ranks, &converted, "ties");
        // Equal up to summation order, not bitwise.
        assert_close(rankit_ranks[0], rankit_ranks[2]);
        assert_close(rankit_ranks[1], rankit_ranks[4]);
    }
}

#[test]
fn infinite_strength_gives_hard_ranks_with_shared_ties() {
    // The pre-delegation loop returned NaN for tied values here (0 * inf);
    // the delegated version gives each tie half a win.
    let ranks = rankit::soft_rank(&[2.0, 1.0, 2.0, 3.0], f64::INFINITY);
    assert_same_ranks(&ranks, &[1.5, 0.0, 1.5, 3.0], "infinite strength");
    let reversed = rankit::soft_rank(&[2.0, 1.0, 2.0, 3.0], f64::NEG_INFINITY);
    assert_same_ranks(
        &reversed,
        &[1.5, 3.0, 1.5, 0.0],
        "negative infinite strength",
    );
}
