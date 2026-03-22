//! Python bindings for rankit (Rust) using PyO3.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
// Helpers: accept numpy 1D array OR Python list
// ---------------------------------------------------------------------------

/// Extract a `Vec<f64>` from either a numpy array (f64 or f32) or a Python list.
fn extract_f64_vec(input: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = input.extract::<PyReadonlyArray1<f64>>() {
        Ok(arr.as_array().to_vec())
    } else if let Ok(arr) = input.extract::<PyReadonlyArray1<f32>>() {
        Ok(arr.as_array().iter().map(|&v| v as f64).collect())
    } else {
        input.extract::<Vec<f64>>()
    }
}

/// Extract a `Vec<f32>` from either a numpy array (f32 or f64) or a Python list.
fn extract_f32_vec(input: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(arr) = input.extract::<PyReadonlyArray1<f32>>() {
        Ok(arr.as_array().to_vec())
    } else if let Ok(arr) = input.extract::<PyReadonlyArray1<f64>>() {
        Ok(arr.as_array().iter().map(|&v| v as f32).collect())
    } else {
        input.extract::<Vec<f32>>()
    }
}

// ---------------------------------------------------------------------------
// Differentiable ranking
// ---------------------------------------------------------------------------

/// Differentiable soft ranking using optimal-transport relaxation.
///
/// Args:
///     scores: 1D array of scores to rank.
///     temperature: Smoothing temperature (higher = smoother). Default 1.0.
///
/// Returns:
///     numpy array of soft ranks (0-indexed, fractional).
#[pyfunction(name = "soft_rank")]
#[pyo3(signature = (scores, temperature = 1.0))]
fn soft_rank_py<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    temperature: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let scores = extract_f64_vec(scores)?;
    let result = soft_rank(&scores, temperature);
    Ok(result.into_pyarray(py))
}

/// Differentiable soft ranking using the NeuralSort relaxation.
///
/// Args:
///     scores: 1D array of scores to rank.
///     temperature: Smoothing temperature (higher = smoother). Default 1.0.
///
/// Returns:
///     numpy array of soft ranks (0-indexed, fractional).
#[pyfunction(name = "soft_rank_neural_sort")]
#[pyo3(signature = (scores, temperature = 1.0))]
fn soft_rank_neural_sort_py<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    temperature: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let scores = extract_f64_vec(scores)?;
    let result = soft_rank_neural_sort(&scores, temperature);
    Ok(result.into_pyarray(py))
}

/// Differentiable soft ranking using sigmoid-based pairwise comparisons.
///
/// Args:
///     scores: 1D array of scores to rank.
///     temperature: Smoothing temperature (higher = smoother). Default 1.0.
///
/// Returns:
///     numpy array of soft ranks (0-indexed, fractional).
#[pyfunction(name = "soft_rank_sigmoid")]
#[pyo3(signature = (scores, temperature = 1.0))]
fn soft_rank_sigmoid_py<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    temperature: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let scores = extract_f64_vec(scores)?;
    let result = soft_rank_sigmoid(&scores, temperature);
    Ok(result.into_pyarray(py))
}

/// Differentiable soft ranking using smooth indicator functions.
///
/// Args:
///     scores: 1D array of scores to rank.
///     temperature: Smoothing temperature (higher = smoother). Default 1.0.
///
/// Returns:
///     numpy array of soft ranks (0-indexed, fractional).
#[pyfunction(name = "soft_rank_smooth_i")]
#[pyo3(signature = (scores, temperature = 1.0))]
fn soft_rank_smooth_i_py<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    temperature: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let scores = extract_f64_vec(scores)?;
    let result = soft_rank_smooth_i(&scores, temperature);
    Ok(result.into_pyarray(py))
}

/// Differentiable top-k selection via relaxed permutation matrices.
///
/// Args:
///     scores: 1D array of scores.
///     k: Number of top elements to select.
///     temperature: Smoothing temperature. Default 1.0.
///
/// Returns:
///     Tuple of (values, indicators) as numpy arrays. `values` contains
///     relaxed top-k scores; `indicators` contains soft selection weights.
#[pyfunction(name = "differentiable_topk")]
#[pyo3(signature = (scores, k, temperature = 1.0))]
fn differentiable_topk_py<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    k: usize,
    temperature: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let scores = extract_f64_vec(scores)?;
    let (values, indicators) = differentiable_topk(&scores, k, temperature);
    Ok((values.into_pyarray(py), indicators.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// LTR losses
// ---------------------------------------------------------------------------

/// RankNet pairwise cross-entropy loss.
///
/// Args:
///     predictions: 1D array of predicted scores.
///     relevance: 1D array of ground-truth relevance labels.
///
/// Returns:
///     Scalar loss value.
#[pyfunction(name = "ranknet_loss")]
#[pyo3(signature = (predictions, relevance))]
fn ranknet_loss_py(predictions: &Bound<'_, PyAny>, relevance: &Bound<'_, PyAny>) -> PyResult<f64> {
    let predictions = extract_f64_vec(predictions)?;
    let relevance = extract_f64_vec(relevance)?;
    if predictions.len() != relevance.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("predictions length ({}) != relevance length ({})", predictions.len(), relevance.len())
        ));
    }
    Ok(ranknet_loss(&predictions, &relevance))
}

/// ApproxNDCG: differentiable approximation of NDCG via softmax.
///
/// Args:
///     predictions: 1D array of predicted scores.
///     relevance: 1D array of ground-truth relevance labels.
///     temperature: Softmax temperature. Default 1.0.
///     k: Truncation depth. None for full list. Default None.
///
/// Returns:
///     Scalar loss value (negative approximate NDCG).
#[pyfunction(name = "approx_ndcg")]
#[pyo3(signature = (predictions, relevance, temperature = 1.0, k = None))]
fn approx_ndcg_py(
    predictions: &Bound<'_, PyAny>,
    relevance: &Bound<'_, PyAny>,
    temperature: f64,
    k: Option<usize>,
) -> PyResult<f64> {
    let predictions = extract_f64_vec(predictions)?;
    let relevance = extract_f64_vec(relevance)?;
    if predictions.len() != relevance.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("predictions length ({}) != relevance length ({})", predictions.len(), relevance.len())
        ));
    }
    Ok(approx_ndcg(&predictions, &relevance, temperature, k))
}

/// LambdaLoss: a general framework for ranking losses.
///
/// Args:
///     predictions: 1D array of predicted scores.
///     relevance: 1D array of ground-truth relevance labels.
///     k: Truncation depth. None for full list. Default None.
///
/// Returns:
///     Scalar loss value.
#[pyfunction(name = "lambda_loss")]
#[pyo3(signature = (predictions, relevance, k = None))]
fn lambda_loss_py(
    predictions: &Bound<'_, PyAny>,
    relevance: &Bound<'_, PyAny>,
    k: Option<usize>,
) -> PyResult<f64> {
    let predictions = extract_f64_vec(predictions)?;
    let relevance = extract_f64_vec(relevance)?;
    if predictions.len() != relevance.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("predictions length ({}) != relevance length ({})", predictions.len(), relevance.len())
        ));
    }
    Ok(lambda_loss(&predictions, &relevance, k))
}

/// ListNet loss using top-1 probability distribution (KL divergence).
///
/// Args:
///     predictions: 1D array of predicted scores.
///     relevance: 1D array of ground-truth relevance labels.
///     temperature: Softmax temperature. Default 1.0.
///
/// Returns:
///     Scalar loss value.
#[pyfunction(name = "listnet_loss")]
#[pyo3(signature = (predictions, relevance, temperature = 1.0))]
fn listnet_loss_py(
    predictions: &Bound<'_, PyAny>,
    relevance: &Bound<'_, PyAny>,
    temperature: f64,
) -> PyResult<f64> {
    let predictions = extract_f64_vec(predictions)?;
    let relevance = extract_f64_vec(relevance)?;
    if predictions.len() != relevance.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("predictions length ({}) != relevance length ({})", predictions.len(), relevance.len())
        ));
    }
    Ok(listnet_loss(&predictions, &relevance, temperature))
}

/// ListMLE loss: likelihood loss for permutation learning.
///
/// Args:
///     predictions: 1D array of predicted scores.
///     relevance: 1D array of ground-truth relevance labels.
///     temperature: Softmax temperature. Default 1.0.
///
/// Returns:
///     Scalar loss value.
#[pyfunction(name = "listmle_loss")]
#[pyo3(signature = (predictions, relevance, temperature = 1.0))]
fn listmle_loss_py(
    predictions: &Bound<'_, PyAny>,
    relevance: &Bound<'_, PyAny>,
    temperature: f64,
) -> PyResult<f64> {
    let predictions = extract_f64_vec(predictions)?;
    let relevance = extract_f64_vec(relevance)?;
    if predictions.len() != relevance.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("predictions length ({}) != relevance length ({})", predictions.len(), relevance.len())
        ));
    }
    Ok(listmle_loss(&predictions, &relevance, temperature))
}

// ---------------------------------------------------------------------------
// Gradient computation
// ---------------------------------------------------------------------------

/// Compute LambdaRank gradients for a single query.
///
/// Args:
///     scores: 1D array of model scores (f32 precision).
///     relevance: 1D array of relevance labels (f32 precision).
///     k: Truncation depth. None for full list. Default None.
///     sigma: Sigmoid scaling factor. Default 1.0.
///     cost_sensitive: Use cost-sensitive variant. Default False.
///
/// Returns:
///     numpy array of per-document gradient values (f32).
#[pyfunction(name = "compute_lambdarank_gradients")]
#[pyo3(signature = (scores, relevance, k = None, sigma = 1.0, cost_sensitive = false))]
fn compute_lambdarank_gradients_py<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    relevance: &Bound<'py, PyAny>,
    k: Option<usize>,
    sigma: f32,
    cost_sensitive: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let scores = extract_f32_vec(scores)?;
    let relevance = extract_f32_vec(relevance)?;
    if scores.len() != relevance.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("scores length ({}) != relevance length ({})", scores.len(), relevance.len())
        ));
    }
    let params = LambdaRankParams {
        sigma,
        cost_sensitivity: cost_sensitive,
        ..LambdaRankParams::default()
    };
    let result = compute_lambdarank_gradients(&scores, &relevance, params, k)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Compute RankingSVM gradients for a single query.
///
/// Args:
///     scores: 1D array of model scores (f32 precision).
///     relevance: 1D array of relevance labels (f32 precision).
///     c: Regularization parameter. Default 1.0.
///     normalize_queries: Normalize gradients per query. Default False.
///
/// Returns:
///     numpy array of per-document gradient values (f32).
#[pyfunction(name = "compute_ranking_svm_gradients")]
#[pyo3(signature = (scores, relevance, c = 1.0, normalize_queries = false))]
fn compute_ranking_svm_gradients_py<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    relevance: &Bound<'py, PyAny>,
    c: f32,
    normalize_queries: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let scores = extract_f32_vec(scores)?;
    let relevance = extract_f32_vec(relevance)?;
    if scores.len() != relevance.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("scores length ({}) != relevance length ({})", scores.len(), relevance.len())
        ));
    }
    let params = RankingSVMParams {
        c,
        query_normalization: normalize_queries,
        ..RankingSVMParams::default()
    };
    let result = compute_ranking_svm_gradients(&scores, &relevance, params)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(result.into_pyarray(py))
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

/// Normalized Discounted Cumulative Gain at depth k.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance grade.
///     k: Evaluation depth.
///
/// Returns:
///     NDCG@k score in [0, 1].
#[pyfunction(name = "ndcg")]
#[pyo3(signature = (ranked, qrels, k))]
fn ndcg_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>, k: usize) -> f64 {
    let ranked = convert_ranked(ranked);
    let qrels = qrels_to_hashmap(qrels);
    graded::compute_ndcg(&ranked, &qrels, k)
}

/// Mean Average Precision over graded relevance judgments.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance grade (> 0 = relevant).
///
/// Returns:
///     MAP score in [0, 1].
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

/// Mean Reciprocal Rank.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///
/// Returns:
///     MRR score (1/rank of first relevant doc), or 0.0 if none found.
#[pyfunction(name = "mrr")]
#[pyo3(signature = (ranked, qrels))]
fn mrr_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::mrr(&ids, &relevant)
}

/// Precision at depth k.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///     k: Evaluation depth.
///
/// Returns:
///     Fraction of top-k documents that are relevant.
#[pyfunction(name = "precision_at_k")]
#[pyo3(signature = (ranked, qrels, k))]
fn precision_at_k_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>, k: usize) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::precision_at_k(&ids, &relevant, k)
}

/// Recall at depth k.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///     k: Evaluation depth.
///
/// Returns:
///     Fraction of relevant documents found in the top-k.
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
    m.add("__version__", "0.1.0")?;
    register(py, m)
}
