//! Matrix-valued continuous relaxations of descending argsort.
//!
//! Rows represent output positions from highest to lowest score. Columns
//! represent items in their original input order.

use thiserror::Error;

/// Errors returned by differentiable sorting operators.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SortingError {
    /// The temperature was not finite and strictly positive.
    #[error("temperature must be finite and greater than zero, got {0}")]
    InvalidTemperature(f64),
    /// An input score was not finite.
    #[error("score at index {index} must be finite, got {value}")]
    NonFiniteScore {
        /// Index of the invalid score.
        index: usize,
        /// Invalid score value.
        value: f64,
    },
}

fn validate(scores: &[f64], temperature: f64) -> Result<(), SortingError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(SortingError::InvalidTemperature(temperature));
    }
    if let Some((index, &value)) = scores
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SortingError::NonFiniteScore { index, value });
    }
    Ok(())
}

fn softmax(logits: &mut [f64]) {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for value in logits.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in logits {
        *value /= sum;
    }
}

/// Compute the NeuralSort relaxation of descending argsort.
///
/// The returned `n × n` row-stochastic matrix uses rows for descending output
/// positions and columns for items in their original input order. Lower
/// temperatures make the rows sharper. With distinct scores, row-wise argmax
/// equals descending argsort. Tied items receive symmetric mass; the operator
/// does not impose a strict order between them.
///
/// This implements the formula from Grover et al., ["Stochastic Optimization
/// of Sorting Networks via Continuous Relaxations"](https://openreview.net/forum?id=H1eSS3CcKX).
pub fn neural_sort(scores: &[f64], temperature: f64) -> Result<Vec<Vec<f64>>, SortingError> {
    validate(scores, temperature)?;
    let n = scores.len();
    let absolute_difference_sums: Vec<f64> = scores
        .iter()
        .map(|&score| scores.iter().map(|&other| (score - other).abs()).sum())
        .collect();

    let mut matrix = Vec::with_capacity(n);
    for position in 0..n {
        let scale = (n as f64 + 1.0) - 2.0 * (position as f64 + 1.0);
        let mut row: Vec<f64> = scores
            .iter()
            .zip(&absolute_difference_sums)
            .map(|(&score, &difference_sum)| (scale * score - difference_sum) / temperature)
            .collect();
        softmax(&mut row);
        matrix.push(row);
    }
    Ok(matrix)
}

/// Compute the SoftSort relaxation of descending argsort using absolute distance.
///
/// The returned `n × n` row-stochastic matrix uses rows for descending output
/// positions and columns for items in their original input order. Lower
/// temperatures make the rows sharper. With distinct scores, row-wise argmax
/// equals descending argsort. Tied items receive symmetric mass; the operator
/// does not impose a strict order between them.
///
/// This implements the absolute-distance formula from Prillo and Eisenschlos,
/// ["SoftSort: A Continuous Relaxation for the argsort
/// Operator"](https://proceedings.mlr.press/v119/prillo20a.html).
pub fn soft_sort(scores: &[f64], temperature: f64) -> Result<Vec<Vec<f64>>, SortingError> {
    validate(scores, temperature)?;
    let mut sorted = scores.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));

    let mut matrix = Vec::with_capacity(scores.len());
    for order_statistic in sorted {
        let mut row: Vec<f64> = scores
            .iter()
            .map(|&score| -(order_statistic - score).abs() / temperature)
            .collect();
        softmax(&mut row);
        matrix.push(row);
    }
    Ok(matrix)
}

fn expected_ranks(matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = matrix.len();
    let mut ranks = vec![0.0; n];
    for (position, row) in matrix.iter().enumerate() {
        let rank = (n - 1 - position) as f64;
        for (item, &probability) in row.iter().enumerate() {
            ranks[item] += rank * probability;
        }
    }
    ranks
}

/// Compute expected ranks from [`neural_sort`].
///
/// Rank zero denotes the lowest score and rank `n - 1` the highest.
pub fn neural_sort_ranks(scores: &[f64], temperature: f64) -> Result<Vec<f64>, SortingError> {
    neural_sort(scores, temperature).map(|matrix| expected_ranks(&matrix))
}

/// Compute expected ranks from [`soft_sort`].
///
/// Rank zero denotes the lowest score and rank `n - 1` the highest.
pub fn soft_sort_ranks(scores: &[f64], temperature: f64) -> Result<Vec<f64>, SortingError> {
    soft_sort(scores, temperature).map(|matrix| expected_ranks(&matrix))
}

#[cfg(test)]
mod tests {
    use super::*;

    type Operator = fn(&[f64], f64) -> Result<Vec<Vec<f64>>, SortingError>;

    fn operators() -> [Operator; 2] {
        [neural_sort, soft_sort]
    }

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    #[test]
    fn matrices_are_row_stochastic() {
        for operator in operators() {
            let matrix = operator(&[2.0, -1.0, 4.0, 0.5], 0.7).unwrap();
            for row in matrix {
                assert!(row.iter().all(|value| value.is_finite() && *value >= 0.0));
                assert_close(row.iter().sum(), 1.0, 1e-12);
            }
        }
    }

    #[test]
    fn sharp_row_argmax_is_descending_argsort() {
        let scores = [2.0, -1.0, 4.0, 0.5];
        let expected = [2, 0, 3, 1];
        for operator in operators() {
            let matrix = operator(&scores, 1e-3).unwrap();
            let actual: Vec<usize> = matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|left, right| left.1.total_cmp(right.1))
                        .unwrap()
                        .0
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn translation_invariant() {
        let scores = [2.0, -1.0, 4.0, 0.5];
        let translated: Vec<_> = scores.iter().map(|score| score + 17.0).collect();
        for operator in operators() {
            let expected = operator(&scores, 0.7).unwrap();
            let actual = operator(&translated, 0.7).unwrap();
            for (actual_row, expected_row) in actual.iter().zip(expected) {
                for (&actual, expected) in actual_row.iter().zip(expected_row) {
                    assert_close(actual, expected, 1e-12);
                }
            }
        }
    }

    #[test]
    fn permutation_equivariant() {
        let scores = [2.0, -1.0, 4.0, 0.5];
        let permutation = [2, 0, 3, 1];
        let permuted: Vec<_> = permutation.iter().map(|&index| scores[index]).collect();
        for operator in operators() {
            let expected = operator(&scores, 0.7).unwrap();
            let actual = operator(&permuted, 0.7).unwrap();
            for (actual_row, expected_row) in actual.iter().zip(expected) {
                for (new_index, &old_index) in permutation.iter().enumerate() {
                    assert_close(actual_row[new_index], expected_row[old_index], 1e-12);
                }
            }
        }
    }

    #[test]
    fn ties_have_equal_expected_ranks() {
        let scores = [3.0, 1.0, 3.0, -2.0];
        for ranks in [
            neural_sort_ranks(&scores, 0.5).unwrap(),
            soft_sort_ranks(&scores, 0.5).unwrap(),
        ] {
            assert_close(ranks[0], ranks[2], 1e-12);
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        for operator in operators() {
            assert!(matches!(
                operator(&[1.0], 0.0),
                Err(SortingError::InvalidTemperature(0.0))
            ));
            assert!(matches!(
                operator(&[f64::NAN], 1.0),
                Err(SortingError::NonFiniteScore { index: 0, .. })
            ));
        }
    }
}
