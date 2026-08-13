# ranklab

Differentiable ranking, learning-to-rank losses, and IR evaluation metrics, backed by a Rust implementation for low overhead per call.

Python bindings for the [rankit](https://crates.io/crates/rankit) Rust crate.

## Install

    pip install ranklab

## Quick start

```python
import ranklab

# Differentiable soft ranking
scores = [5.0, 1.0, 2.0, 4.0, 3.0]
ranks = ranklab.soft_rank(scores, temperature=1.0)

# RankNet pairwise loss
predictions = [0.8, 0.3, 0.6]
relevance = [2.0, 0.0, 1.0]
loss = ranklab.ranknet_loss(predictions, relevance)

# NDCG evaluation
ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
qrels = {"doc1": 2, "doc2": 1, "doc3": 0}
score = ranklab.ndcg(ranked, qrels, k=3)
```

## API

| Name | Description |
|------|-------------|
| `soft_rank` | Pairwise sigmoid soft ranks; higher strength is sharper |
| `pairwise_logistic_rank` | Pairwise logistic rank heuristic |
| `neural_sort` | NeuralSort relaxation matrix |
| `soft_sort` | SoftSort relaxation matrix |
| `soft_rank_neural_sort` | Compatibility name for `pairwise_logistic_rank`; not NeuralSort |
| `soft_rank_sigmoid` | Soft ranking via sigmoid pairwise comparisons |
| `soft_rank_smooth_i` | Compatibility name for `soft_rank_sigmoid`; not SmoothI |
| `differentiable_topk` | Pairwise soft-rank thresholding, returns (values, indicators) |
| `ranknet_loss` | RankNet pairwise cross-entropy loss |
| `approx_ndcg` | Differentiable NDCG approximation via softmax |
| `lambda_loss` | LambdaLoss ranking loss |
| `listnet_loss` | ListNet cross-entropy over top-one probabilities |
| `listmle_loss` | ListMLE-inspired likelihood over sigmoid soft ranks |
| `compute_lambdarank_gradients` | Per-document LambdaRank gradients |
| `compute_ranking_svm_gradients` | Per-document RankingSVM gradients |
| `ndcg` | Normalized Discounted Cumulative Gain at k |
| `map_score` | Mean Average Precision |
| `mrr` | Mean Reciprocal Rank |
| `precision_at_k` | Precision at depth k |
| `recall_at_k` | Recall at depth k |
| `err_at_k` | Expected Reciprocal Rank at depth k |
| `rbp_at_k` | Rank-Biased Precision at depth k |
| `f_measure_at_k` | F-measure (harmonic mean of P and R) at depth k |
| `success_at_k` | Whether any relevant doc appears in top-k |
| `r_precision` | Precision at R (R = number of relevant docs) |
| `average_precision` | Average Precision for binary relevance |
| `paired_t_test` | Paired t-test on two score sets, returns dict |
| `confidence_interval` | Confidence interval for a score set |
| `cohens_d` | Cohen's d effect size |
| `load_trec_run` | Parse a TREC run file into {query_id: [(doc_id, score)]} |
| `load_qrels` | Parse a TREC qrels file into {query_id: {doc_id: rel}} |
| `evaluate_batch` | Batch evaluation across multiple queries |
| `evaluate_trec` | Evaluate TREC run+qrels files with batch metrics |

## numpy support

Scoring and loss functions accept numpy arrays or Python lists. Gradient functions return numpy float32 arrays. Eval metrics take `list[tuple[str, float]]` for ranked results and `dict[str, int]` for relevance judgments.

## License

MIT OR Apache-2.0
