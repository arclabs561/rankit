//! Error types for gradient computation operations.

use std::fmt;

/// Errors that can occur during gradient computation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GradientError {
    /// Empty input provided.
    EmptyInput,
    /// Length mismatch between scores and relevance.
    LengthMismatch {
        scores_len: usize,
        relevance_len: usize,
    },
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Invalid relevance scores (e.g., negative values when not allowed).
    InvalidRelevance(String),
    /// Invalid NDCG computation (e.g., k=0 or k > length).
    InvalidNDCG { k: usize, length: usize },
}

impl fmt::Display for GradientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GradientError::EmptyInput => write!(f, "Input is empty"),
            GradientError::LengthMismatch {
                scores_len,
                relevance_len,
            } => {
                write!(
                    f,
                    "Length mismatch: scores has {} elements, relevance has {}",
                    scores_len, relevance_len
                )
            }
            GradientError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            GradientError::InvalidRelevance(msg) => write!(f, "Invalid relevance: {}", msg),
            GradientError::InvalidNDCG { k, length } => {
                write!(f, "Invalid NDCG@k: k={} but length={}", k, length)
            }
        }
    }
}

impl std::error::Error for GradientError {}
