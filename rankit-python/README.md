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
| `soft_rank` | Soft ranking via optimal-transport relaxation |
| `soft_rank_neural_sort` | Soft ranking via NeuralSort relaxation |
| `soft_rank_sigmoid` | Soft ranking via sigmoid pairwise comparisons |
| `soft_rank_smooth_i` | Soft ranking via smooth indicator functions |
| `differentiable_topk` | Differentiable top-k selection, returns (values, indicators) |
| `ranknet_loss` | RankNet pairwise cross-entropy loss |
| `approx_ndcg` | Differentiable NDCG approximation via softmax |
| `lambda_loss` | LambdaLoss ranking loss |
| `listnet_loss` | ListNet top-1 probability loss (KL divergence) |
| `listmle_loss` | ListMLE likelihood loss for permutation learning |
| `compute_lambdarank_gradients` | Per-document LambdaRank gradients |
| `compute_ranking_svm_gradients` | Per-document RankingSVM gradients |
| `ndcg` | Normalized Discounted Cumulative Gain at k |
| `map_score` | Mean Average Precision |
| `mrr` | Mean Reciprocal Rank |
| `precision_at_k` | Precision at depth k |
| `recall_at_k` | Recall at depth k |

## numpy support

Scoring and loss functions accept numpy arrays or Python lists. Gradient functions return numpy float32 arrays. Eval metrics take `list[tuple[str, float]]` for ranked results and `dict[str, int]` for relevance judgments.

## License

MIT OR Apache-2.0
