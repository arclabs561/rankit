//! Batch processing utilities for efficient multi-query ranking.
//!
//! With `parallel` feature, batches process in parallel via rayon.

use crate::rank::soft_rank;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Compute soft ranks for a batch of value vectors.
///
/// Each query can have a different number of items.
pub fn soft_rank_batch(batch_values: &[Vec<f64>], regularization_strength: f64) -> Vec<Vec<f64>> {
    #[cfg(feature = "parallel")]
    {
        batch_values
            .par_iter()
            .map(|values| soft_rank(values, regularization_strength))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        batch_values
            .iter()
            .map(|values| soft_rank(values, regularization_strength))
            .collect()
    }
}

/// Compute Spearman loss for a batch of prediction-target pairs.
pub fn spearman_loss_batch(
    batch_predictions: &[Vec<f64>],
    batch_targets: &[Vec<f64>],
    regularization_strength: f64,
) -> Vec<f64> {
    #[cfg(feature = "parallel")]
    {
        batch_predictions
            .par_iter()
            .zip(batch_targets.par_iter())
            .map(|(pred, targ)| fynch::loss::spearman_loss(pred, targ, regularization_strength))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        batch_predictions
            .iter()
            .zip(batch_targets.iter())
            .map(|(pred, targ)| fynch::loss::spearman_loss(pred, targ, regularization_strength))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_rank_batch() {
        let batch = vec![
            vec![5.0, 1.0, 2.0, 4.0, 3.0],
            vec![3.0, 1.0, 2.0],
            vec![10.0, 5.0, 8.0, 7.0],
        ];

        let ranks = soft_rank_batch(&batch, 1.0);

        assert_eq!(ranks.len(), 3);
        assert_eq!(ranks[0].len(), 5);
        assert_eq!(ranks[1].len(), 3);
        assert_eq!(ranks[2].len(), 4);
    }

    #[test]
    fn test_spearman_loss_batch() {
        let predictions = vec![vec![0.1, 0.9, 0.3], vec![1.0, 2.0, 3.0]];
        let targets = vec![vec![0.0, 1.0, 0.2], vec![1.0, 2.0, 3.0]];

        let losses = spearman_loss_batch(&predictions, &targets, 1.0);

        assert_eq!(losses.len(), 2);
        assert!(losses[0] >= 0.0 && losses[0] <= 2.0);
        assert!(losses[1] < losses[0]);
    }
}
