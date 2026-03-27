#![warn(missing_docs)]
//! Learning to Rank for Rust: differentiable ranking, LTR losses, trainers,
//! and IR evaluation metrics.
//!
//! rankit provides everything needed to train and evaluate ranking models:
//!
//! - **Differentiable ranking**: sigmoid-based soft ranking with multiple method
//!   variants (NeuralSort, SoftRank, SmoothI). O(n^2) complexity, suitable for
//!   n < 1000.
//! - **LTR losses**: RankNet, LambdaLoss (NDCG-weighted), ApproxNDCG, ListNet,
//!   ListMLE. Pairwise and listwise paradigms.
//! - **Trainers**: LambdaRank and Ranking SVM with query normalization, cost
//!   sensitivity, and score normalization options.
//! - **Evaluation** (feature `eval`): NDCG, MAP, MRR, Precision/Recall@K, ERR,
//!   RBP, F-measure. TREC format parsing. Batch evaluation. Statistical testing
//!   (paired t-test, confidence intervals, Cohen's d).
//!
//! # Quick start
//!
//! ```rust
//! use rankit::{soft_rank, ranknet_loss};
//!
//! // Differentiable ranking
//! let scores = vec![5.0, 1.0, 2.0, 4.0, 3.0];
//! let ranks = soft_rank(&scores, 1.0);
//! // ranks[0] is highest (~4.0), ranks[1] is lowest (~0.0)
//!
//! // RankNet pairwise loss
//! let predictions = vec![0.8, 0.3, 0.6];
//! let relevance = vec![2.0, 0.0, 1.0];
//! let loss = ranknet_loss(&predictions, &relevance);
//! ```
//!
//! # Feature flags
//!
//! | Feature | Default | What it adds |
//! |---------|---------|-------------|
//! | `eval` | yes | IR evaluation metrics, TREC parsing, batch eval, statistics |
//! | `losses` | yes | LTR loss functions (RankNet, LambdaLoss, ApproxNDCG, ListNet, ListMLE) |
//! | `gumbel` | no | Gumbel-Softmax, relaxed top-k (requires `rand`) |
//! | `parallel` | no | Rayon parallelization for batch operations |
//! | `serde` | no | Serialization for eval result types |

/// Differentiable ranking operations (sigmoid-based, O(n^2)).
pub mod rank;

/// Multiple ranking method variants from research papers.
pub mod methods;

/// Analytical gradient computation for soft ranking and Spearman loss.
pub mod gradients;

/// Batch processing utilities.
pub mod batch;

/// Performance-optimized implementations.
pub mod optimized;

/// LTR loss functions and advanced ranking operations.
#[cfg(feature = "losses")]
pub mod losses;

/// Differentiable top-k selection.
pub mod topk;

/// Top-k cross-entropy loss for classification.
pub mod topk_ce;

/// Gumbel-Softmax sampling and relaxed top-k.
#[cfg(feature = "gumbel")]
pub mod sampling;

/// IR evaluation metrics, TREC parsing, batch evaluation, statistical testing.
#[cfg(feature = "eval")]
pub mod eval;

/// End-to-end retrieval pipeline: tokenize, index, score, rank.
#[cfg(feature = "pipeline")]
pub mod pipeline;

// --- Re-exports: core ---

pub use batch::{soft_rank_batch, spearman_loss_batch};
pub use gradients::{
    compute_lambdarank_gradients, compute_ranking_svm_gradients, fisher_information_softmax,
    natural_gradient_softmax, ndcg_at_k, pairwise_hinge_loss, sigmoid_derivative,
    soft_rank_gradient, spearman_loss_gradient, with_natural_gradient, GradientError,
    LambdaRankParams, LambdaRankTrainer, RankingSVMParams, RankingSVMTrainer,
};
pub use methods::{
    soft_rank_neural_sort, soft_rank_probabilistic, soft_rank_sigmoid, soft_rank_smooth_i,
    RankingMethod,
};
pub use optimized::{soft_rank_gradient_sparse, soft_rank_optimized};
pub use rank::soft_rank;
pub use topk::differentiable_topk;

#[cfg(feature = "parallel")]
pub use optimized::soft_rank_batch_parallel;

// --- Re-exports: losses ---

#[cfg(feature = "losses")]
pub use losses::{
    approx_ndcg, approx_ndcg_loss, lambda_loss, listmle_loss, listnet_loss, ranknet_loss,
    soft_rank_softsort,
};

// --- Re-exports: gumbel ---

#[cfg(feature = "gumbel")]
pub use sampling::{gumbel_attention_mask, gumbel_softmax, relaxed_topk_gumbel};

// --- Re-exports from fynch (primitives layer) ---

/// Re-export fynch's Spearman loss.
pub use fynch::loss::spearman_loss;

// --- Re-exports: eval ---

#[cfg(feature = "eval")]
pub use eval::{batch as eval_batch, binary, export, graded, statistics, trec, validation};

#[cfg(test)]
mod proptests;

#[cfg(test)]
mod gradient_tests;
