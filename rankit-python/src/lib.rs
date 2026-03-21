//! Python bindings for rankit (Rust) using PyO3.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use rankit::eval::{binary, graded};
use rankit::gradients::{
    compute_lambdarank_gradients, compute_ranking_svm_gradients, LambdaRankParams, RankingSVMParams,
};
use rankit::losses::{approx_ndcg, lambda_loss, listmle_loss, listnet_loss, ranknet_loss};
use rankit::methods::{soft_rank_neural_sort, soft_rank_sigmoid, soft_rank_smooth_i};
use rankit::{differentiable_topk, soft_rank};

// ---------------------------------------------------------------------------
// Differentiable ranking
// ---------------------------------------------------------------------------

#[pyfunction(name = "soft_rank")]
#[pyo3(signature = (scores, temperature = 1.0))]
fn soft_rank_py(scores: Vec<f64>, temperature: f64) -> Vec<f64> {
    soft_rank(&scores, temperature)
}

#[pyfunction(name = "soft_rank_neural_sort")]
#[pyo3(signature = (scores, temperature = 1.0))]
fn soft_rank_neural_sort_py(scores: Vec<f64>, temperature: f64) -> Vec<f64> {
    soft_rank_neural_sort(&scores, temperature)
}

#[pyfunction(name = "soft_rank_sigmoid")]
#[pyo3(signature = (scores, temperature = 1.0))]
fn soft_rank_sigmoid_py(scores: Vec<f64>, temperature: f64) -> Vec<f64> {
    soft_rank_sigmoid(&scores, temperature)
}

#[pyfunction(name = "soft_rank_smooth_i")]
#[pyo3(signature = (scores, temperature = 1.0))]
fn soft_rank_smooth_i_py(scores: Vec<f64>, temperature: f64) -> Vec<f64> {
    soft_rank_smooth_i(&scores, temperature)
}

#[pyfunction(name = "differentiable_topk")]
#[pyo3(signature = (scores, k, temperature = 1.0))]
fn differentiable_topk_py(scores: Vec<f64>, k: usize, temperature: f64) -> (Vec<f64>, Vec<f64>) {
    differentiable_topk(&scores, k, temperature)
}

// ---------------------------------------------------------------------------
// LTR losses
// ---------------------------------------------------------------------------

#[pyfunction(name = "ranknet_loss")]
#[pyo3(signature = (predictions, relevance))]
fn ranknet_loss_py(predictions: Vec<f64>, relevance: Vec<f64>) -> f64 {
    ranknet_loss(&predictions, &relevance)
}

#[pyfunction(name = "approx_ndcg")]
#[pyo3(signature = (predictions, relevance, temperature = 1.0, k = None))]
fn approx_ndcg_py(
    predictions: Vec<f64>,
    relevance: Vec<f64>,
    temperature: f64,
    k: Option<usize>,
) -> f64 {
    approx_ndcg(&predictions, &relevance, temperature, k)
}

#[pyfunction(name = "lambda_loss")]
#[pyo3(signature = (predictions, relevance, k = None))]
fn lambda_loss_py(predictions: Vec<f64>, relevance: Vec<f64>, k: Option<usize>) -> f64 {
    lambda_loss(&predictions, &relevance, k)
}

#[pyfunction(name = "listnet_loss")]
#[pyo3(signature = (predictions, relevance, temperature = 1.0))]
fn listnet_loss_py(predictions: Vec<f64>, relevance: Vec<f64>, temperature: f64) -> f64 {
    listnet_loss(&predictions, &relevance, temperature)
}

#[pyfunction(name = "listmle_loss")]
#[pyo3(signature = (predictions, relevance, temperature = 1.0))]
fn listmle_loss_py(predictions: Vec<f64>, relevance: Vec<f64>, temperature: f64) -> f64 {
    listmle_loss(&predictions, &relevance, temperature)
}

// ---------------------------------------------------------------------------
// Gradient computation
// ---------------------------------------------------------------------------

#[pyfunction(name = "compute_lambdarank_gradients")]
#[pyo3(signature = (scores, relevance, k = None, sigma = 1.0, cost_sensitive = false))]
fn compute_lambdarank_gradients_py(
    scores: Vec<f32>,
    relevance: Vec<f32>,
    k: Option<usize>,
    sigma: f32,
    cost_sensitive: bool,
) -> PyResult<Vec<f32>> {
    let params = LambdaRankParams {
        sigma,
        cost_sensitivity: cost_sensitive,
        ..LambdaRankParams::default()
    };
    compute_lambdarank_gradients(&scores, &relevance, params, k)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction(name = "compute_ranking_svm_gradients")]
#[pyo3(signature = (scores, relevance, c = 1.0, normalize_queries = false))]
fn compute_ranking_svm_gradients_py(
    scores: Vec<f32>,
    relevance: Vec<f32>,
    c: f32,
    normalize_queries: bool,
) -> PyResult<Vec<f32>> {
    let params = RankingSVMParams {
        c,
        query_normalization: normalize_queries,
        ..RankingSVMParams::default()
    };
    compute_ranking_svm_gradients(&scores, &relevance, params)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

// ---------------------------------------------------------------------------
// Eval metrics (graded)
// ---------------------------------------------------------------------------

/// Convert Python qrels dict to HashMap<String, u32>.
fn qrels_to_hashmap(qrels: HashMap<String, u32>) -> HashMap<String, u32> {
    qrels
}

/// Convert Python qrels dict to HashSet of relevant doc IDs (value > 0).
fn qrels_to_hashset(qrels: &HashMap<String, u32>) -> HashSet<String> {
    qrels
        .iter()
        .filter(|(_, &v)| v > 0)
        .map(|(k, _)| k.clone())
        .collect()
}

/// Convert Python ranked list to Vec<(String, f32)>.
fn convert_ranked(ranked: Vec<(String, f32)>) -> Vec<(String, f32)> {
    ranked
}

#[pyfunction(name = "ndcg")]
#[pyo3(signature = (ranked, qrels, k))]
fn ndcg_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>, k: usize) -> f64 {
    let ranked = convert_ranked(ranked);
    let qrels = qrels_to_hashmap(qrels);
    graded::compute_ndcg(&ranked, &qrels, k)
}

#[pyfunction(name = "map_score")]
#[pyo3(signature = (ranked, qrels))]
fn map_score_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>) -> f64 {
    let ranked = convert_ranked(ranked);
    let qrels = qrels_to_hashmap(qrels);
    graded::compute_map(&ranked, &qrels)
}

// ---------------------------------------------------------------------------
// Eval metrics (binary)
// ---------------------------------------------------------------------------

#[pyfunction(name = "mrr")]
#[pyo3(signature = (ranked, qrels))]
fn mrr_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::mrr(&ids, &relevant)
}

#[pyfunction(name = "precision_at_k")]
#[pyo3(signature = (ranked, qrels, k))]
fn precision_at_k_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>, k: usize) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::precision_at_k(&ids, &relevant, k)
}

#[pyfunction(name = "recall_at_k")]
#[pyo3(signature = (ranked, qrels, k))]
fn recall_at_k_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>, k: usize) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::recall_at_k(&ids, &relevant, k)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Registers all functions in the module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Differentiable ranking
    m.add_function(wrap_pyfunction!(soft_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(soft_rank_neural_sort_py, m)?)?;
    m.add_function(wrap_pyfunction!(soft_rank_sigmoid_py, m)?)?;
    m.add_function(wrap_pyfunction!(soft_rank_smooth_i_py, m)?)?;
    m.add_function(wrap_pyfunction!(differentiable_topk_py, m)?)?;

    // LTR losses
    m.add_function(wrap_pyfunction!(ranknet_loss_py, m)?)?;
    m.add_function(wrap_pyfunction!(approx_ndcg_py, m)?)?;
    m.add_function(wrap_pyfunction!(lambda_loss_py, m)?)?;
    m.add_function(wrap_pyfunction!(listnet_loss_py, m)?)?;
    m.add_function(wrap_pyfunction!(listmle_loss_py, m)?)?;

    // Gradients
    m.add_function(wrap_pyfunction!(compute_lambdarank_gradients_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ranking_svm_gradients_py, m)?)?;

    // Eval metrics
    m.add_function(wrap_pyfunction!(ndcg_py, m)?)?;
    m.add_function(wrap_pyfunction!(map_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(mrr_py, m)?)?;
    m.add_function(wrap_pyfunction!(precision_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(recall_at_k_py, m)?)?;

    Ok(())
}

#[pymodule]
fn ranklab(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(py, m)
}
