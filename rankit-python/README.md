# ranklab

Python bindings for [rankit](https://docs.rs/rankit) -- differentiable ranking, LTR losses, and IR evaluation metrics.

## Install

```
pip install ranklab
```

## Usage

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

**Differentiable ranking:** `soft_rank`, `soft_rank_neural_sort`, `soft_rank_sigmoid`, `soft_rank_smooth_i`, `differentiable_topk`

**LTR losses:** `ranknet_loss`, `approx_ndcg`, `lambda_loss`, `listnet_loss`, `listmle_loss`

**Gradients:** `compute_lambdarank_gradients`, `compute_ranking_svm_gradients`

**Eval metrics:** `ndcg`, `map_score`, `mrr`, `precision_at_k`, `recall_at_k`

## License

MIT OR Apache-2.0
