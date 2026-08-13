//! Statistical testing utilities for evaluation results.

/// Result of a paired t-test.
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// Computed t-statistic.
    pub t_statistic: f64,
    /// Two-sided p-value from Student's t-distribution.
    pub p_value: f64,
    /// Degrees of freedom (n-1).
    pub degrees_of_freedom: usize,
    /// Mean of paired differences (A - B).
    pub mean_difference: f64,
    /// Standard error of the mean difference.
    pub std_error: f64,
    /// Whether `p_value < alpha`.
    pub significant: bool,
}

/// Perform a two-sided paired Student's t-test on two sets of scores.
///
/// The test is applied to the paired differences `method_a[i] - method_b[i]`.
/// At least two pairs are required. An insufficient sample, non-finite input, or
/// zero variance with zero mean difference produces undefined (`NaN`) test
/// statistics. A constant nonzero difference produces an infinite t-statistic
/// and a zero p-value.
///
/// # Panics
///
/// Panics if the samples have different lengths or if `alpha` is not strictly
/// between zero and one.
pub fn paired_t_test(method_a: &[f64], method_b: &[f64], alpha: f64) -> TTestResult {
    assert_eq!(
        method_a.len(),
        method_b.len(),
        "method_a and method_b must have same length"
    );
    assert!(
        alpha.is_finite() && alpha > 0.0 && alpha < 1.0,
        "alpha must be finite and strictly between 0 and 1"
    );

    let n = method_a.len();
    let df = n.saturating_sub(1);
    if n < 2 {
        return undefined_t_test(df);
    }

    let differences: Vec<f64> = method_a.iter().zip(method_b).map(|(a, b)| a - b).collect();
    let (mean_diff, variance) = sample_mean_variance(&differences);
    let std_error = (variance / n as f64).sqrt();
    let t_statistic = mean_diff / std_error;
    let p_value = if t_statistic.is_nan() {
        f64::NAN
    } else if t_statistic.is_infinite() {
        0.0
    } else {
        student_t_two_sided_p_value(t_statistic, df as f64)
    };

    TTestResult {
        t_statistic,
        p_value,
        degrees_of_freedom: df,
        mean_difference: mean_diff,
        std_error,
        significant: p_value < alpha,
    }
}

/// Compute a two-sided Student-t confidence interval for a population mean.
///
/// Returns `(NaN, NaN)` for fewer than two observations or non-finite input.
/// A constant finite sample has a zero-width interval at its mean.
///
/// # Panics
///
/// Panics if `confidence` is not strictly between zero and one.
pub fn confidence_interval(scores: &[f64], confidence: f64) -> (f64, f64) {
    assert!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "confidence must be finite and strictly between 0 and 1"
    );
    if scores.len() < 2 {
        return (f64::NAN, f64::NAN);
    }

    let (mean, variance) = sample_mean_variance(scores);
    let se = (variance / scores.len() as f64).sqrt();
    if !mean.is_finite() || !se.is_finite() {
        return (f64::NAN, f64::NAN);
    }

    let critical = student_t_quantile((1.0 + confidence) / 2.0, (scores.len() - 1) as f64);
    let margin = critical * se;
    (mean - margin, mean + margin)
}

/// Compute Cohen's paired-samples effect size (`d_z`).
///
/// This is the mean paired difference divided by the sample standard deviation
/// of the paired differences. It is therefore consistent with [`paired_t_test`]
/// (`t = d_z * sqrt(n)`), rather than the pooled standard deviation used for
/// independent samples. Fewer than two pairs or an undefined `0 / 0` effect
/// returns `NaN`; a constant nonzero difference returns signed infinity.
///
/// # Panics
///
/// Panics if the samples have different lengths.
pub fn cohens_d(method_a: &[f64], method_b: &[f64]) -> f64 {
    assert_eq!(
        method_a.len(),
        method_b.len(),
        "method_a and method_b must have same length"
    );
    if method_a.len() < 2 {
        return f64::NAN;
    }

    let differences: Vec<f64> = method_a.iter().zip(method_b).map(|(a, b)| a - b).collect();
    let (mean, variance) = sample_mean_variance(&differences);
    mean / variance.sqrt()
}

fn undefined_t_test(degrees_of_freedom: usize) -> TTestResult {
    TTestResult {
        t_statistic: f64::NAN,
        p_value: f64::NAN,
        degrees_of_freedom,
        mean_difference: f64::NAN,
        std_error: f64::NAN,
        significant: false,
    }
}

fn sample_mean_variance(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    (mean, variance)
}

fn student_t_two_sided_p_value(t: f64, degrees_of_freedom: f64) -> f64 {
    let x = degrees_of_freedom / (degrees_of_freedom + t * t);
    regularized_beta(x, degrees_of_freedom / 2.0, 0.5)
}

fn student_t_quantile(probability: f64, degrees_of_freedom: f64) -> f64 {
    debug_assert!(probability > 0.5 && probability < 1.0);
    let target_tail = 2.0 * (1.0 - probability);
    let mut low = 0.0;
    let mut high = 1.0;
    while student_t_two_sided_p_value(high, degrees_of_freedom) > target_tail {
        high *= 2.0;
    }
    for _ in 0..80 {
        let mid = (low + high) / 2.0;
        if student_t_two_sided_p_value(mid, degrees_of_freedom) > target_tail {
            low = mid;
        } else {
            high = mid;
        }
    }
    (low + high) / 2.0
}

// Regularized incomplete beta, using the continued-fraction expansion from
// Numerical Recipes. This is the distribution identity used for Student's t.
fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(x, a, b) / a
    } else {
        1.0 - front * beta_continued_fraction(1.0 - x, b, a) / b
    }
}

fn beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITERATIONS: usize = 200;
    const EPSILON: f64 = 3.0e-14;
    const MIN_VALUE: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < MIN_VALUE {
        d = MIN_VALUE;
    }
    d = 1.0 / d;
    let mut result = d;

    for m in 1..=MAX_ITERATIONS {
        let m = m as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_VALUE {
            d = MIN_VALUE;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_VALUE {
            c = MIN_VALUE;
        }
        d = 1.0 / d;
        result *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_VALUE {
            d = MIN_VALUE;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_VALUE {
            c = MIN_VALUE;
        }
        d = 1.0 / d;
        let delta = d * c;
        result *= delta;
        if (delta - 1.0).abs() <= EPSILON {
            break;
        }
    }
    result
}

fn ln_gamma(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let z = value - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        sum += coefficient / (z + index as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Fixed reference values generated with scipy.stats 1.18.0. They are not
    // computed from this module's implementation.
    const A: [f64; 6] = [2.1, 2.5, 3.6, 4.0, 5.2, 5.8];
    const B: [f64; 6] = [1.8, 2.7, 3.1, 3.9, 4.8, 5.1];

    #[test]
    fn paired_t_test_matches_scipy_fixture() {
        let result = paired_t_test(&A, &B, 0.05);
        assert_abs_diff_eq!(result.t_statistic, 2.323_790_007_724_45, epsilon = 1e-12);
        assert_abs_diff_eq!(result.p_value, 0.067_733_017_765_571_79, epsilon = 1e-12);
        assert_abs_diff_eq!(result.mean_difference, 0.3, epsilon = 1e-14);
        assert_abs_diff_eq!(result.std_error, 0.129_099_444_873_580_58, epsilon = 1e-14);
        assert_eq!(result.degrees_of_freedom, 5);
        assert!(!result.significant);
    }

    #[test]
    fn confidence_interval_matches_scipy_fixture() {
        let (lower, upper) = confidence_interval(&A, 0.95);
        assert_abs_diff_eq!(lower, 2.339_145_889_778_353_5, epsilon = 1e-12);
        assert_abs_diff_eq!(upper, 5.394_187_443_554_98, epsilon = 1e-12);
    }

    #[test]
    fn paired_cohens_d_matches_external_fixture_and_t_identity() {
        let d = cohens_d(&A, &B);
        assert_abs_diff_eq!(d, 0.948_683_298_050_513_8, epsilon = 1e-12);
        let t = paired_t_test(&A, &B, 0.05).t_statistic;
        assert_abs_diff_eq!(t, d * (A.len() as f64).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn two_observation_interval_uses_heavy_t_tails() {
        let interval = confidence_interval(&[1.0, 2.0], 0.95);
        assert_abs_diff_eq!(interval.0, -4.853_102_368_087_347, epsilon = 1e-11);
        assert_abs_diff_eq!(interval.1, 7.853_102_368_087_347, epsilon = 1e-11);
    }

    #[test]
    fn constant_samples_have_defined_or_explicitly_undefined_edges() {
        assert_eq!(confidence_interval(&[3.0, 3.0, 3.0], 0.95), (3.0, 3.0));

        let nonzero = paired_t_test(&[2.0, 2.0], &[1.0, 1.0], 0.05);
        assert_eq!(nonzero.t_statistic, f64::INFINITY);
        assert_eq!(nonzero.p_value, 0.0);
        assert!(nonzero.significant);
        assert_eq!(cohens_d(&[2.0, 2.0], &[1.0, 1.0]), f64::INFINITY);

        let zero = paired_t_test(&[1.0, 1.0], &[1.0, 1.0], 0.05);
        assert!(zero.t_statistic.is_nan());
        assert!(zero.p_value.is_nan());
        assert!(!zero.significant);
        assert!(cohens_d(&[1.0, 1.0], &[1.0, 1.0]).is_nan());
    }

    #[test]
    fn insufficient_and_non_finite_samples_are_undefined() {
        let empty = paired_t_test(&[], &[], 0.05);
        assert!(empty.t_statistic.is_nan());
        assert_eq!(empty.degrees_of_freedom, 0);
        assert!(confidence_interval(&[1.0], 0.95).0.is_nan());
        assert!(cohens_d(&[1.0], &[0.0]).is_nan());

        let non_finite = paired_t_test(&[1.0, f64::NAN], &[0.0, 0.0], 0.05);
        assert!(non_finite.t_statistic.is_nan());
        assert!(non_finite.p_value.is_nan());
        assert!(confidence_interval(&[1.0, f64::INFINITY], 0.95).0.is_nan());
        assert!(cohens_d(&[1.0, f64::NAN], &[0.0, 0.0]).is_nan());
    }

    #[test]
    #[should_panic(expected = "alpha must be finite and strictly between 0 and 1")]
    fn rejects_invalid_alpha() {
        paired_t_test(&[1.0, 2.0], &[0.0, 1.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "confidence must be finite and strictly between 0 and 1")]
    fn rejects_invalid_confidence() {
        confidence_interval(&[1.0, 2.0], f64::NAN);
    }
}
