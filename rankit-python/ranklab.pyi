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
    """ListNet loss using top-1 probability distribution (KL divergence).

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
    """ListMLE loss: likelihood loss for permutation learning.

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
