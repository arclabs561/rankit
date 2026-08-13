use rankit::{listmle_loss, listnet_loss, neural_sort, soft_sort};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "expected {expected}, got {actual}"
    );
}

fn normalized_exponentials(logits: [f64; 3]) -> [f64; 3] {
    let weights = logits.map(f64::exp);
    let denominator: f64 = weights.iter().sum();
    weights.map(|weight| weight / denominator)
}

fn assert_matrix_close(actual: &[Vec<f64>], expected: [[f64; 3]; 3]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_row, expected_row) in actual.iter().zip(expected) {
        assert_eq!(actual_row.len(), expected_row.len());
        for (&actual, expected) in actual_row.iter().zip(expected_row) {
            assert_close(actual, expected);
        }
    }
}

#[test]
fn listnet_public_api_matches_top_one_cross_entropy_at_non_unit_strength() {
    let predictions = [2.0_f64.ln(), 3.0_f64.ln()];
    let targets = [4.0_f64.ln(), 0.0];
    let strength = 2.0;

    let target_probability = 16.0 / 17.0;
    let expected = -target_probability * (4.0 / 13.0_f64).ln()
        - (1.0 - target_probability) * (9.0 / 13.0_f64).ln();

    assert_close(listnet_loss(&predictions, &targets, strength), expected);
}

#[test]
fn listmle_public_api_rewards_the_target_order() {
    let targets = [3.0, 2.0, 1.0, 0.0];
    let aligned = listmle_loss(&[3.0, 2.0, 1.0, 0.0], &targets, 4.0);
    let reversed = listmle_loss(&[0.0, 1.0, 2.0, 3.0], &targets, 4.0);

    assert!(
        aligned < reversed,
        "aligned loss {aligned} should be below reversed loss {reversed}"
    );
}

#[test]
fn two_item_sorting_operators_match_their_closed_forms() {
    let scores = [2.0, 0.0];
    let temperature = 2.0;
    let expected_high_probability = 1.0 / (1.0 + (-1.0_f64).exp());

    for matrix in [
        neural_sort(&scores, temperature).unwrap(),
        soft_sort(&scores, temperature).unwrap(),
    ] {
        assert_close(matrix[0][0], expected_high_probability);
        assert_close(matrix[0][1], 1.0 - expected_high_probability);
        assert_close(matrix[1][0], 1.0 - expected_high_probability);
        assert_close(matrix[1][1], expected_high_probability);
    }
}

#[test]
fn three_item_sorting_operators_match_distinct_full_matrix_closed_forms() {
    let scores = [3.0, 1.0, 0.0];
    let temperature = 2.0;

    // NeuralSort logits are ((n + 1 - 2k) * s_j - sum_i |s_j-s_i|) / tau.
    let expected_neural = [
        normalized_exponentials([0.5, -0.5, -2.0]),
        normalized_exponentials([-2.5, -1.5, -2.0]),
        normalized_exponentials([-5.5, -2.5, -2.0]),
    ];
    // SoftSort logits are -|sort(s)_k - s_j| / tau.
    let expected_soft = [
        normalized_exponentials([0.0, -1.0, -1.5]),
        normalized_exponentials([-1.0, 0.0, -0.5]),
        normalized_exponentials([-1.5, -0.5, 0.0]),
    ];

    let neural = neural_sort(&scores, temperature).unwrap();
    let soft = soft_sort(&scores, temperature).unwrap();
    assert_matrix_close(&neural, expected_neural);
    assert_matrix_close(&soft, expected_soft);
    assert_ne!(neural, soft, "three-item relaxations must remain distinct");
}
