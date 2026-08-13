"""Type stubs for ranklab Python bindings.

Note: NaN values in numeric inputs will propagate to the output silently.
Filter NaN before calling if this is not desired.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__version__: str

# Type alias: inputs accept numpy arrays or Python lists.
_Scores = Union[NDArray[np.floating], List[float]]

# ---------------------------------------------------------------------------
# Differentiable ranking
# ---------------------------------------------------------------------------

def soft_rank(scores: _Scores, temperature: float = 1.0) -> NDArray[np.float64]:
    """Differentiable soft ranking using optimal-transport relaxation.

    NaN values in scores will propagate to the output. Filter NaN before
    calling if this is not desired.

    Args:
        scores: 1D array of scores to rank.
        temperature: Smoothing temperature (higher = smoother). Default 1.0.

    Returns:
        numpy array of soft ranks (0-indexed, fractional).
    """
    ...

def soft_rank_neural_sort(scores: _Scores, temperature: float = 1.0) -> NDArray[np.float64]:
    """Differentiable soft ranking using the NeuralSort relaxation.

    Args:
        scores: 1D array of scores to rank.
        temperature: Smoothing temperature (higher = smoother). Default 1.0.

    Returns:
        numpy array of soft ranks (0-indexed, fractional).
    """
    ...

def soft_rank_sigmoid(scores: _Scores, temperature: float = 1.0) -> NDArray[np.float64]:
    """Differentiable soft ranking using sigmoid-based pairwise comparisons.

    Args:
        scores: 1D array of scores to rank.
        temperature: Smoothing temperature (higher = smoother). Default 1.0.

    Returns:
        numpy array of soft ranks (0-indexed, fractional).
    """
    ...

def soft_rank_smooth_i(scores: _Scores, temperature: float = 1.0) -> NDArray[np.float64]:
    """Differentiable soft ranking using smooth indicator functions.

    Args:
        scores: 1D array of scores to rank.
        temperature: Smoothing temperature (higher = smoother). Default 1.0.

    Returns:
        numpy array of soft ranks (0-indexed, fractional).
    """
    ...

def differentiable_topk(
    scores: _Scores, k: int, temperature: float = 1.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Differentiable top-k selection via relaxed permutation matrices.

    Args:
        scores: 1D array of scores.
        k: Number of top elements to select.
        temperature: Smoothing temperature. Default 1.0.

    Returns:
        Tuple of (values, indicators) as numpy arrays. ``values`` contains
        relaxed top-k scores; ``indicators`` contains soft selection weights.
    """
    ...

# ---------------------------------------------------------------------------
# LTR losses
# ---------------------------------------------------------------------------

def ranknet_loss(predictions: _Scores, relevance: _Scores) -> float:
    """RankNet pairwise cross-entropy loss.

    Args:
        predictions: 1D array of predicted scores.
        relevance: 1D array of ground-truth relevance labels.

    Returns:
        Scalar loss value.

    Raises:
        ValueError: If predictions and relevance have different lengths.
    """
    ...

def approx_ndcg(
    predictions: _Scores,
    relevance: _Scores,
    temperature: float = 1.0,
    k: Optional[int] = None,
) -> float:
    """ApproxNDCG: differentiable approximation of NDCG via softmax.

    Args:
        predictions: 1D array of predicted scores.
        relevance: 1D array of ground-truth relevance labels.
        temperature: Softmax temperature. Default 1.0.
        k: Truncation depth. None for full list. Default None.

    Returns:
        Scalar loss value (negative approximate NDCG).

    Raises:
        ValueError: If predictions and relevance have different lengths.
    """
    ...

def lambda_loss(
    predictions: _Scores, relevance: _Scores, k: Optional[int] = None
) -> float:
    """LambdaLoss: a general framework for ranking losses.

    Args:
        predictions: 1D array of predicted scores.
        relevance: 1D array of ground-truth relevance labels.
        k: Truncation depth. None for full list. Default None.

    Returns:
        Scalar loss value.

    Raises:
        ValueError: If predictions and relevance have different lengths.
    """
    ...

def listnet_loss(
    predictions: _Scores, relevance: _Scores, temperature: float = 1.0
) -> float:
    """ListNet cross-entropy over top-one probability distributions.

    Args:
        predictions: 1D array of predicted scores.
        relevance: 1D array of ground-truth relevance labels.
        temperature: Softmax temperature. Default 1.0.

    Returns:
        Scalar loss value.

    Raises:
        ValueError: If predictions and relevance have different lengths.
    """
    ...

def listmle_loss(
    predictions: _Scores, relevance: _Scores, temperature: float = 1.0
) -> float:
    """ListMLE-inspired likelihood computed from sigmoid soft ranks.

    Args:
        predictions: 1D array of predicted scores.
        relevance: 1D array of ground-truth relevance labels.
        temperature: Softmax temperature. Default 1.0.

    Returns:
        Scalar loss value.

    Raises:
        ValueError: If predictions and relevance have different lengths.
    """
    ...

# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def compute_lambdarank_gradients(
    scores: _Scores,
    relevance: _Scores,
    k: Optional[int] = None,
    sigma: float = 1.0,
    cost_sensitive: bool = False,
) -> NDArray[np.float32]:
    """Compute LambdaRank gradients for a single query.

    Args:
        scores: 1D array of model scores.
        relevance: 1D array of relevance labels.
        k: Truncation depth. None for full list. Default None.
        sigma: Sigmoid scaling factor. Default 1.0.
        cost_sensitive: Use cost-sensitive variant. Default False.

    Returns:
        numpy array of per-document gradient values (float32).

    Raises:
        ValueError: If scores and relevance have different lengths.
    """
    ...

def compute_ranking_svm_gradients(
    scores: _Scores,
    relevance: _Scores,
    c: float = 1.0,
    normalize_queries: bool = False,
) -> NDArray[np.float32]:
    """Compute RankingSVM gradients for a single query.

    Args:
        scores: 1D array of model scores.
        relevance: 1D array of relevance labels.
        c: Regularization parameter. Default 1.0.
        normalize_queries: Normalize gradients per query. Default False.

    Returns:
        numpy array of per-document gradient values (float32).

    Raises:
        ValueError: If scores and relevance have different lengths.
    """
    ...

# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------

def ndcg(ranked: List[Tuple[str, float]], qrels: Dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at depth k.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance grade.
        k: Evaluation depth.

    Returns:
        NDCG@k score in [0, 1].
    """
    ...

def map_score(ranked: List[Tuple[str, float]], qrels: Dict[str, int]) -> float:
    """Mean Average Precision over graded relevance judgments.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance grade (> 0 = relevant).

    Returns:
        MAP score in [0, 1].
    """
    ...

def mrr(ranked: List[Tuple[str, float]], qrels: Dict[str, int]) -> float:
    """Mean Reciprocal Rank.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).

    Returns:
        MRR score (1/rank of first relevant doc), or 0.0 if none found.
    """
    ...

def precision_at_k(
    ranked: List[Tuple[str, float]], qrels: Dict[str, int], k: int
) -> float:
    """Precision at depth k.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
        k: Evaluation depth.

    Returns:
        Fraction of top-k documents that are relevant.
    """
    ...

def recall_at_k(
    ranked: List[Tuple[str, float]], qrels: Dict[str, int], k: int
) -> float:
    """Recall at depth k.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
        k: Evaluation depth.

    Returns:
        Fraction of relevant documents found in the top-k.
    """
    ...

def err_at_k(
    ranked: List[Tuple[str, float]], qrels: Dict[str, int], k: int
) -> float:
    """Expected Reciprocal Rank at depth k.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
        k: Evaluation depth.

    Returns:
        ERR@k score in [0, 1].
    """
    ...

def rbp_at_k(
    ranked: List[Tuple[str, float]],
    qrels: Dict[str, int],
    k: int,
    persistence: float = 0.95,
) -> float:
    """Rank-Biased Precision at depth k.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
        k: Evaluation depth.
        persistence: User persistence parameter in (0, 1). Default 0.95.

    Returns:
        RBP@k score in [0, 1].
    """
    ...

def f_measure_at_k(
    ranked: List[Tuple[str, float]],
    qrels: Dict[str, int],
    k: int,
    beta: float = 1.0,
) -> float:
    """F-measure at depth k: harmonic mean of precision and recall.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
        k: Evaluation depth.
        beta: Weight of recall relative to precision. Default 1.0 (F1).

    Returns:
        F-measure at k.
    """
    ...

def success_at_k(
    ranked: List[Tuple[str, float]], qrels: Dict[str, int], k: int
) -> float:
    """Success at k: 1.0 if any relevant document in top-k, else 0.0.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).
        k: Evaluation depth.

    Returns:
        1.0 if any relevant document found in top-k, else 0.0.
    """
    ...

def r_precision(
    ranked: List[Tuple[str, float]], qrels: Dict[str, int]
) -> float:
    """R-Precision: precision at R, where R is the number of relevant documents.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).

    Returns:
        R-precision score.
    """
    ...

def average_precision(
    ranked: List[Tuple[str, float]], qrels: Dict[str, int]
) -> float:
    """Average Precision for binary relevance.

    Args:
        ranked: List of (doc_id, score) tuples, ordered by score descending.
        qrels: Dict mapping doc_id to integer relevance (> 0 = relevant).

    Returns:
        Average precision score in [0, 1].
    """
    ...

# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def paired_t_test(
    scores_a: _Scores, scores_b: _Scores, alpha: float = 0.05
) -> Dict[str, object]:
    """Paired t-test on two sets of per-query scores.

    Args:
        scores_a: 1D array of scores from method A.
        scores_b: 1D array of scores from method B.
        alpha: Significance level. Default 0.05.

    Returns:
        Dict with keys: t_statistic (float), p_value (float),
        significant (bool), degrees_of_freedom (int),
        mean_difference (float), std_error (float).

    Raises:
        ValueError: If inputs have different lengths.
    """
    ...

def confidence_interval(
    scores: _Scores, confidence: float = 0.95
) -> Tuple[float, float]:
    """Confidence interval for a set of scores.

    Args:
        scores: 1D array of scores.
        confidence: Confidence level (e.g. 0.95 for 95%). Default 0.95.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    ...

def cohens_d(scores_a: _Scores, scores_b: _Scores) -> float:
    """Cohen's d effect size between two sets of scores.

    Interpretation: |d| < 0.2 negligible, 0.2-0.5 small,
    0.5-0.8 medium, >= 0.8 large.

    Args:
        scores_a: 1D array of scores from method A.
        scores_b: 1D array of scores from method B.

    Returns:
        Effect size (positive means A > B).

    Raises:
        ValueError: If inputs have different lengths.
    """
    ...

# ---------------------------------------------------------------------------
# TREC file I/O
# ---------------------------------------------------------------------------

def load_trec_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load a TREC run file.

    Format: ``query_id Q0 doc_id rank score run_tag``

    Args:
        path: Path to the TREC run file.

    Returns:
        Dict mapping query_id to list of (doc_id, score) tuples,
        sorted by score descending within each query.

    Raises:
        IOError: If the file cannot be read or has invalid format.
    """
    ...

def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Load a TREC qrels file.

    Format: ``query_id 0 doc_id relevance``

    Args:
        path: Path to the qrels file.

    Returns:
        Dict mapping query_id to dict of {doc_id: relevance_grade}.

    Raises:
        IOError: If the file cannot be read or has invalid format.
    """
    ...

# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    rankings: List[List[str]],
    qrels: List[List[str]],
    metrics: List[str],
) -> Dict[str, object]:
    """Evaluate multiple queries using binary relevance metrics.

    Args:
        rankings: List of rankings, each a list of doc_id strings (best first).
        qrels: List of relevance sets, each a list of relevant doc_ids.
        metrics: Metric names. Supported: "ndcg@10", "ndcg@5", "precision@10",
            "precision@5", "precision@1", "recall@10", "recall@5", "mrr",
            "ap"/"map", "err@10", "rbp@10", "f1@10", "success@10", "r_precision".

    Returns:
        Dict with "per_query" (list of metric dicts) and "aggregated"
        (dict of metric name to mean value).
    """
    ...

def evaluate_trec(
    run_path: str, qrels_path: str, metrics: List[str]
) -> Dict[str, object]:
    """Evaluate TREC run and qrels files with batch metrics.

    Args:
        run_path: Path to a TREC run file.
        qrels_path: Path to a TREC qrels file.
        metrics: Metric names (same as ``evaluate_batch``).

    Returns:
        Dict with "per_query" (list of dicts with query_id and metrics)
        and "aggregated" (dict of metric name to mean value).

    Raises:
        IOError: If files cannot be read or have invalid format.
    """
    ...

def spearman_loss(
    predictions: Union[NDArray[np.floating], List[float]],
    targets: Union[NDArray[np.floating], List[float]],
    regularization_strength: float = 1.0,
) -> float:
    """Differentiable Spearman rank correlation loss (via fynch).

    Computes a soft Spearman loss between predicted scores and target ranks.
    Returns 0 for perfectly correlated inputs, higher for worse correlation.

    Args:
        predictions: 1D array of predicted scores.
        targets: 1D array of target values.
        regularization_strength: Temperature for soft sorting.

    Returns:
        Scalar loss value.

    Raises:
        ValueError: If inputs have different lengths.
    """
    ...
