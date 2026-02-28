//! IR evaluation metrics, TREC parsing, batch evaluation, statistical testing.
//!
//! - **Binary relevance**: Precision, Recall, MRR, NDCG, AP, ERR, RBP, F-measure
//! - **Graded relevance**: NDCG and MAP with multi-level relevance
//! - **TREC format**: Load and parse run files and qrels
//! - **Batch evaluation**: Process multiple queries at once
//! - **Statistical testing**: Paired t-test, confidence intervals, Cohen's d
//! - **Export**: CSV and JSON output

pub mod batch;
pub mod binary;
pub mod export;
pub mod graded;
pub mod statistics;
pub mod trec;
pub mod validation;

pub use batch::{evaluate_batch_binary, evaluate_trec_batch, BatchResults, QueryResults};
pub use binary::DegradationMetrics;
pub use export::export_to_csv;
pub use statistics::{cohens_d, confidence_interval, paired_t_test, TTestResult};
pub use trec::{
    group_qrels_by_query, group_runs_by_query, load_qrels, load_trec_runs, Qrel, TrecRun,
};
pub use validation::{
    validate_beta, validate_metric_inputs, validate_persistence, ValidationError,
};

#[cfg(feature = "serde")]
pub use binary::Metrics;
#[cfg(feature = "serde")]
pub use export::export_to_json;
