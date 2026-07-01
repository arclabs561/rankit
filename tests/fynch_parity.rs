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
