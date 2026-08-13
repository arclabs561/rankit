use rankit::{listmle_loss, listnet_loss, neural_sort, soft_sort};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "expected {expected}, got {actual}"
    );
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
