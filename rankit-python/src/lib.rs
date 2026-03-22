//! Python bindings for rankit (Rust) using PyO3.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use rankit::eval::{binary, graded, statistics, trec, batch};
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
// Eval metrics (binary) -- additional
// ---------------------------------------------------------------------------

/// Expected Reciprocal Rank at depth k.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///     k: Evaluation depth.
///
/// Returns:
///     ERR@k score in [0, 1].
#[pyfunction(name = "err_at_k")]
#[pyo3(signature = (ranked, qrels, k))]
fn err_at_k_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>, k: usize) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::err_at_k(&ids, &relevant, k)
}

/// Rank-Biased Precision at depth k.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///     k: Evaluation depth.
///     persistence: User persistence parameter in (0, 1). Default 0.95.
///
/// Returns:
///     RBP@k score in [0, 1].
#[pyfunction(name = "rbp_at_k")]
#[pyo3(signature = (ranked, qrels, k, persistence = 0.95))]
fn rbp_at_k_py(
    ranked: Vec<(String, f32)>,
    qrels: HashMap<String, u32>,
    k: usize,
    persistence: f64,
) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::rbp_at_k(&ids, &relevant, k, persistence)
}

/// F-measure at depth k: harmonic mean of precision and recall.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///     k: Evaluation depth.
///     beta: Weight of recall relative to precision. Default 1.0 (F1).
///
/// Returns:
///     F-measure at k.
#[pyfunction(name = "f_measure_at_k")]
#[pyo3(signature = (ranked, qrels, k, beta = 1.0))]
fn f_measure_at_k_py(
    ranked: Vec<(String, f32)>,
    qrels: HashMap<String, u32>,
    k: usize,
    beta: f64,
) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::f_measure_at_k(&ids, &relevant, k, beta)
}

/// Success at k: whether at least one relevant document is in top-k.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///     k: Evaluation depth.
///
/// Returns:
///     1.0 if any relevant document found in top-k, else 0.0.
#[pyfunction(name = "success_at_k")]
#[pyo3(signature = (ranked, qrels, k))]
fn success_at_k_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>, k: usize) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::success_at_k(&ids, &relevant, k)
}

/// R-Precision: precision at R, where R is the number of relevant documents.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///
/// Returns:
///     R-precision score.
#[pyfunction(name = "r_precision")]
#[pyo3(signature = (ranked, qrels))]
fn r_precision_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::r_precision(&ids, &relevant)
}

/// Average Precision for binary relevance.
///
/// Args:
///     ranked: List of (doc_id, score) tuples, ordered by score descending.
///     qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
///
/// Returns:
///     Average precision score in [0, 1].
#[pyfunction(name = "average_precision")]
#[pyo3(signature = (ranked, qrels))]
fn average_precision_py(ranked: Vec<(String, f32)>, qrels: HashMap<String, u32>) -> f64 {
    let relevant = qrels_to_hashset(&qrels);
    let ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
    binary::average_precision(&ids, &relevant)
}

// ---------------------------------------------------------------------------
// Statistical testing
// ---------------------------------------------------------------------------

/// Paired t-test on two sets of per-query scores.
///
/// Args:
///     scores_a: 1D array of scores from method A.
///     scores_b: 1D array of scores from method B.
///     alpha: Significance level. Default 0.05.
///
/// Returns:
///     Dict with keys: t_statistic, p_value, significant, degrees_of_freedom,
///     mean_difference, std_error.
///
/// Raises:
///     ValueError: If inputs have different lengths or fewer than 2 elements.
#[pyfunction(name = "paired_t_test")]
#[pyo3(signature = (scores_a, scores_b, alpha = 0.05))]
fn paired_t_test_py<'py>(
    py: Python<'py>,
    scores_a: &Bound<'py, PyAny>,
    scores_b: &Bound<'py, PyAny>,
    alpha: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let a = extract_f64_vec(scores_a)?;
    let b = extract_f64_vec(scores_b)?;
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("scores_a length ({}) != scores_b length ({})", a.len(), b.len()),
        ));
    }
    let result = statistics::paired_t_test(&a, &b, alpha);
    let d = pyo3::types::PyDict::new(py);
    d.set_item("t_statistic", result.t_statistic)?;
    d.set_item("p_value", result.p_value)?;
    d.set_item("significant", result.significant)?;
    d.set_item("degrees_of_freedom", result.degrees_of_freedom)?;
    d.set_item("mean_difference", result.mean_difference)?;
    d.set_item("std_error", result.std_error)?;
    Ok(d)
}

/// Confidence interval for a set of scores.
///
/// Args:
///     scores: 1D array of scores.
///     confidence: Confidence level (e.g. 0.95 for 95%). Default 0.95.
///
/// Returns:
///     Tuple of (lower, upper) bounds.
#[pyfunction(name = "confidence_interval")]
#[pyo3(signature = (scores, confidence = 0.95))]
fn confidence_interval_py(
    scores: &Bound<'_, PyAny>,
    confidence: f64,
) -> PyResult<(f64, f64)> {
    let s = extract_f64_vec(scores)?;
    Ok(statistics::confidence_interval(&s, confidence))
}

/// Cohen's d effect size between two sets of scores.
///
/// Interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >= 0.8 large.
///
/// Args:
///     scores_a: 1D array of scores from method A.
///     scores_b: 1D array of scores from method B.
///
/// Returns:
///     Effect size (positive means A > B).
///
/// Raises:
///     ValueError: If inputs have different lengths.
#[pyfunction(name = "cohens_d")]
#[pyo3(signature = (scores_a, scores_b))]
fn cohens_d_py(
    scores_a: &Bound<'_, PyAny>,
    scores_b: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let a = extract_f64_vec(scores_a)?;
    let b = extract_f64_vec(scores_b)?;
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("scores_a length ({}) != scores_b length ({})", a.len(), b.len()),
        ));
    }
    Ok(statistics::cohens_d(&a, &b))
}

// ---------------------------------------------------------------------------
// TREC file I/O
// ---------------------------------------------------------------------------

/// Load a TREC run file.
///
/// Format: ``query_id Q0 doc_id rank score run_tag``
///
/// Args:
///     path: Path to the TREC run file.
///
/// Returns:
///     Dict mapping query_id to list of (doc_id, score) tuples, sorted by
///     score descending within each query.
#[pyfunction(name = "load_trec_run")]
#[pyo3(signature = (path))]
fn load_trec_run_py(path: &str) -> PyResult<HashMap<String, Vec<(String, f32)>>> {
    let runs = trec::load_trec_runs(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let grouped = trec::group_runs_by_query(&runs);
    // Flatten: for each query, merge all run tags into one list (first tag wins if multiple).
    let mut result: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    for (query_id, tag_map) in grouped {
        // Take the first (or only) run tag's results.
        if let Some(docs) = tag_map.into_values().next() {
            result.insert(query_id, docs);
        }
    }
    Ok(result)
}

/// Load a TREC qrels file.
///
/// Format: ``query_id 0 doc_id relevance``
///
/// Args:
///     path: Path to the qrels file.
///
/// Returns:
///     Dict mapping query_id to dict of {doc_id: relevance_grade}.
#[pyfunction(name = "load_qrels")]
#[pyo3(signature = (path))]
fn load_qrels_py(path: &str) -> PyResult<HashMap<String, HashMap<String, u32>>> {
    let qrels = trec::load_qrels(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(trec::group_qrels_by_query(&qrels))
}

// ---------------------------------------------------------------------------
// Batch evaluation
// ---------------------------------------------------------------------------

/// Evaluate multiple queries using binary relevance metrics.
///
/// Args:
///     rankings: List of rankings, where each ranking is a list of doc_id strings
///         in ranked order (best first).
///     qrels: List of relevance sets, where each is a set/list of relevant doc_ids.
///     metrics: List of metric names to compute. Supported:
///         "ndcg@10", "ndcg@5", "precision@10", "precision@5", "precision@1",
///         "recall@10", "recall@5", "mrr", "ap"/"map", "err@10", "rbp@10",
///         "f1@10", "success@10", "r_precision".
///
/// Returns:
///     Dict with keys:
///         "per_query": list of dicts, each mapping metric name to value.
///         "aggregated": dict mapping metric name to mean value across queries.
#[pyfunction(name = "evaluate_batch")]
#[pyo3(signature = (rankings, qrels, metrics))]
fn evaluate_batch_py<'py>(
    py: Python<'py>,
    rankings: Vec<Vec<String>>,
    qrels: Vec<Vec<String>>,
    metrics: Vec<String>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let qrel_sets: Vec<HashSet<String>> = qrels.into_iter().map(|v| v.into_iter().collect()).collect();
    let metric_strs: Vec<&str> = metrics.iter().map(|s| s.as_str()).collect();
    let results = batch::evaluate_batch_binary(&rankings, &qrel_sets, &metric_strs);

    let per_query: Vec<HashMap<String, f64>> = results
        .query_results
        .into_iter()
        .map(|qr| qr.metrics)
        .collect();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("per_query", per_query)?;
    d.set_item("aggregated", results.aggregated)?;
    Ok(d)
}

/// Evaluate TREC run and qrels files with batch metrics.
///
/// Args:
///     run_path: Path to a TREC run file.
///     qrels_path: Path to a TREC qrels file.
///     metrics: List of metric names (same as ``evaluate_batch``).
///
/// Returns:
///     Dict with "per_query" and "aggregated" (same structure as ``evaluate_batch``).
#[pyfunction(name = "evaluate_trec")]
#[pyo3(signature = (run_path, qrels_path, metrics))]
fn evaluate_trec_py<'py>(
    py: Python<'py>,
    run_path: &str,
    qrels_path: &str,
    metrics: Vec<String>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let runs = trec::load_trec_runs(run_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let qrels = trec::load_qrels(qrels_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let metric_strs: Vec<&str> = metrics.iter().map(|s| s.as_str()).collect();
    let results = batch::evaluate_trec_batch(&runs, &qrels, &metric_strs);

    let per_query = pyo3::types::PyList::empty(py);
    for qr in &results.query_results {
        let qd = pyo3::types::PyDict::new(py);
        qd.set_item("query_id", &qr.query_id)?;
        for (k, v) in &qr.metrics {
            qd.set_item(k, v)?;
        }
        per_query.append(qd)?;
    }
    let d = pyo3::types::PyDict::new(py);
    d.set_item("per_query", per_query)?;
    d.set_item("aggregated", results.aggregated)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// Spearman loss (from fynch)
// ---------------------------------------------------------------------------

/// Spearman rank correlation loss (differentiable).
///
/// Computes a soft Spearman loss between predicted scores and target ranks
/// using the fynch differentiable sorting primitives.
///
/// Args:
///     predictions: 1D array of predicted scores.
///     targets: 1D array of target values (higher = better).
///     regularization_strength: Temperature for soft sorting. Default 1.0.
///
/// Returns:
///     Scalar loss value.
#[pyfunction(name = "spearman_loss")]
#[pyo3(signature = (predictions, targets, regularization_strength = 1.0))]
fn spearman_loss_py(
    predictions: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    regularization_strength: f64,
) -> PyResult<f64> {
    let preds = extract_f64_vec(predictions)?;
    let targs = extract_f64_vec(targets)?;
    if preds.len() != targs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("predictions length ({}) != targets length ({})", preds.len(), targs.len())
        ));
    }
    Ok(rankit::spearman_loss(&preds, &targs, regularization_strength))
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
    m.add_function(wrap_pyfunction!(err_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(rbp_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(f_measure_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(success_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(r_precision_py, m)?)?;
    m.add_function(wrap_pyfunction!(average_precision_py, m)?)?;

    // Statistical testing
    m.add_function(wrap_pyfunction!(paired_t_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(confidence_interval_py, m)?)?;
    m.add_function(wrap_pyfunction!(cohens_d_py, m)?)?;

    // TREC I/O
    m.add_function(wrap_pyfunction!(load_trec_run_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_qrels_py, m)?)?;

    // Batch evaluation
    m.add_function(wrap_pyfunction!(evaluate_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_trec_py, m)?)?;

    // Spearman loss (from fynch)
    m.add_function(wrap_pyfunction!(spearman_loss_py, m)?)?;

    Ok(())
}

#[pymodule]
fn ranklab(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    register(py, m)
}
