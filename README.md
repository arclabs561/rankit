# rankit

[![crates.io](https://img.shields.io/crates/v/rankit.svg)](https://crates.io/crates/rankit)
[![Documentation](https://docs.rs/rankit/badge.svg)](https://docs.rs/rankit)
[![CI](https://github.com/arclabs561/rankit/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rankit/actions/workflows/ci.yml)

Learning-to-rank losses and evaluation.

## What it does

- **Differentiable ranking**: sigmoid-based soft ranking $\hat{R}_i(\mathbf{s}) = \sum_{j \neq i} \sigma\bigl(\tau(s_j - s_i)\bigr)$. Variants: NeuralSort, SoftRank/Probabilistic, SmoothI. $O(n^2)$, suitable for lists up to ~1000 items.
- **LTR loss functions**: RankNet, LambdaLoss, ApproxNDCG, ListNet, ListMLE (see formulas below).
- **Gradient trainers**: LambdaRank and Ranking SVM with configurable query normalization, cost sensitivity, and score normalization.
- **IR evaluation metrics**: NDCG, MAP, MRR, Precision@K, Recall@K, ERR, RBP, F-measure, R-Precision, Success@K. Binary and graded relevance.
- **TREC format parsing**: load standard TREC run files and qrels, batch evaluate, export CSV/JSON.
- **Statistical testing**: paired t-test, confidence intervals, Cohen's d effect size.

### Loss functions

| Loss | Formula |
|------|---------|
| RankNet | $\mathcal{L} = \sum_{(i,j): y_i > y_j} \log\bigl(1 + e^{-(s_i - s_j)}\bigr)$ |
| LambdaLoss | RankNet weighted by $\lvert\Delta\text{NDCG}\_{ij}\rvert$ per swapped pair |
| ApproxNDCG | $-\sum_i G(y_i) \cdot D\bigl(\hat{\pi}_i(\mathbf{s})\bigr)$ with soft rank $\hat{\pi}$ |
| ListNet | $\text{KL}\bigl(P_y \;\lVert\; P_s\bigr)$ where $P_z(i) = e^{z_i} / \sum_j e^{z_j}$ |
| ListMLE | $-\sum\_{k=1}^{n} \log \frac{e^{s\_{\pi(k)}}}{\sum\_{j=k}^{n} e^{s\_{\pi(j)}}}$ (likelihood of ground-truth permutation $\pi$) |

## Quick start

```rust
use rankit::{soft_rank, ranknet_loss};

// Differentiable ranking
let scores = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&scores, 1.0);
// ranks[0] ≈ 4.0 (highest), ranks[1] ≈ 0.0 (lowest)

// RankNet pairwise loss
let predictions = vec![0.8, 0.3, 0.6];
let relevance = vec![2.0, 0.0, 1.0];
let loss = ranknet_loss(&predictions, &relevance);
```

## Feature flags

| Feature    | Default | Description |
|------------|---------|-------------|
| `eval`     | yes     | IR evaluation metrics, TREC parsing, batch eval, statistics |
| `losses`   | yes     | LTR loss functions (RankNet, LambdaLoss, ApproxNDCG, ListNet, ListMLE) |
| `gumbel`   | no      | Gumbel-Softmax sampling, relaxed top-k (delegates to `drawset`, pulls `rand`) |
| `pipeline` | no      | End-to-end retrieval pipeline: tokenize, index, score, rank (pulls `textprep`, `postings`, `rankfns`) |
| `parallel` | no      | Rayon parallelization for batch operations |
| `serde`    | no      | Serialization for eval result types |

## Examples

See [`examples/README.md`](examples/README.md) for the full gallery: each
example states the question it answers, the run command, feature flags or data
requirements, and real sample output.

| Example | What it shows |
|---------|---------------|
| `eval_metrics` | Binary and graded IR metrics |
| `soft_rank` | Differentiable ranks at different sharpness values |
| `lambdarank` | Per-document LambdaRank gradients |
| `train_with_schedule` | RankNet training with a warmup-cosine schedule |
| `ltr_lightgbm_train_eval` | LambdaRank on the LightGBM ranking dataset |
| `retrieval_pipeline` | BM25 retrieval with the `pipeline` and `eval` features |

## Crate topology

`rankit` builds on [`fynch`](https://crates.io/crates/fynch) (Fenchel-Young losses, differentiable sorting primitives). Related crates:

- [`rankfns`](https://crates.io/crates/rankfns): scoring functions (BM25, TF-IDF, language models)
- [`rankops`](https://crates.io/crates/rankops): rank fusion and reranking (RRF, CombMNZ, MMR, IR metrics)

## References

- Burges et al. "Learning to Rank using Gradient Descent" (ICML 2005): RankNet
- Qin & Liu. "A General Approximation Framework for Direct Optimization of Information Retrieval Measures" (2010): ApproxNDCG
- Cao et al. "Learning to Rank: From Pairwise Approach to Listwise Approach" (ICML 2007): ListNet
- Xia et al. "Listwise Approach to Learning to Rank" (ICML 2008): ListMLE
- Blondel et al. "Fast Differentiable Sorting and Ranking" (ICML 2020): soft ranking methods

## License

MIT OR Apache-2.0
