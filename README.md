# rankit

[![crates.io](https://img.shields.io/crates/v/rankit.svg)](https://crates.io/crates/rankit)
[![Documentation](https://docs.rs/rankit/badge.svg)](https://docs.rs/rankit)

Learning-to-rank losses and evaluation.

## What it does

- **Differentiable ranking**: sigmoid-based soft ranking $\hat{R}_i(\mathbf{s}) = \sum_{j \neq i} \sigma\bigl(\tau(s_i - s_j)\bigr)$, in $[0, n-1]$ with 0 for the lowest score. The paper-named functions in `methods` are pairwise heuristics, not the published operators. All are $O(n^2)$.
- **Sorting relaxations**: `neural_sort` and `soft_sort` return row-stochastic matrices following the published NeuralSort and SoftSort formulas; expected-rank adapters are also provided.
- **LTR objectives**: RankNet, an NDCG-weighted pairwise loss, ApproxNDCG, top-one ListNet, and a ListMLE-inspired objective (see formulas below).
- **Gradient trainers**: LambdaRank and Ranking SVM with configurable query normalization, cost sensitivity, and score normalization. LambdaRank's defaults use gain $2^{rel}-1$ and turn on two factors that Burges (2010) does not have (cost sensitivity and query normalization); turn them off for Burges' gradients. The evaluation metrics use linear gain, as `trec_eval` does.
- **IR evaluation metrics**: NDCG, MAP, MRR, Precision@K, Recall@K, ERR, RBP, F-measure, R-Precision, Success@K. Binary and graded relevance.
- **TREC format parsing**: load standard TREC run files and qrels, batch evaluate, export CSV/JSON.
- **Statistical testing**: paired t-test, confidence intervals, Cohen's d effect size.

### Loss functions

| Loss | Formula |
|------|---------|
| RankNet | $\mathcal{L} = \frac{1}{\lvert P\rvert}\sum_{(i,j)\in P} \log\bigl(1 + e^{-(s_i - s_j)}\bigr)$ |
| NDCG-weighted pairwise (`lambda_loss`) | RankNet weighted by $\lvert\Delta\text{NDCG}\_{ij}\rvert$ per swapped pair |
| ApproxNDCG | $-\sum_i G(y_i) \cdot D\bigl(\hat{\pi}_i(\mathbf{s})\bigr)$ with soft rank $\hat{\pi}$ |
| ListNet (`listnet_loss`) | Cross-entropy between top-one softmax distributions |
| ListMLE-style (`listmle_loss`) | A permutation likelihood computed from soft ranks in target order |

Here $P$ contains oriented pairs with $y_i > y_j$, excluding relevance differences below $10^{-10}$. RankNet returns zero when $P$ is empty.

Requires Rust 1.89 or newer for the default and no-default-feature builds.
Neither build enables the optional `innr` backend in `rankops`.

## Quick start

```toml
[dependencies]
rankit = "0.1.8"
```

```rust
use rankit::{soft_rank, ranknet_loss};

// Differentiable ranking
let scores = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&scores, 1.0);
// ranks[0] ≈ 3.546 (highest), ranks[1] ≈ 0.454 (lowest)

// RankNet pairwise loss
let predictions = vec![0.8, 0.3, 0.6];
let relevance = vec![2.0, 0.0, 1.0];
let loss = ranknet_loss(&predictions, &relevance);
```

## Feature flags

| Feature    | Default | Description |
|------------|---------|-------------|
| `eval`     | yes     | IR evaluation metrics, TREC parsing, batch eval, statistics |
| `losses`   | yes     | Pairwise, approximate NDCG, and listwise-style objectives |
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

`cargo run --release --example soft_rank` prints how the strength parameter
sharpens soft ranks toward hard ranks:

```text
Scores: [5.0, 1.0, 2.0, 4.0, 3.0]

alpha=  0.1  ranks=[2.248, 1.752, 1.876, 2.124, 2.000]
alpha=  1.0  ranks=[3.546, 0.454, 1.167, 2.833, 2.000]
alpha= 10.0  ranks=[4.000, 0.000, 1.000, 3.000, 2.000]
```

## Crate topology

`rankit` builds on [`fynch`](https://crates.io/crates/fynch) (Fenchel-Young losses, differentiable sorting primitives). Related crates:

- [`rankfns`](https://crates.io/crates/rankfns): scoring functions (BM25, TF-IDF, language models)
- [`rankops`](https://crates.io/crates/rankops): rank fusion and reranking (RRF, CombMNZ, MMR, IR metrics)

## References

- Burges et al. "Learning to Rank using Gradient Descent" (ICML 2005): RankNet
- Burges. ["From RankNet to LambdaRank to LambdaMART: An Overview"](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) (Microsoft Research, MSR-TR-2010-82, 2010): LambdaRank
- Qin et al. "A General Approximation Framework for Direct Optimization of Information Retrieval Measures" (Information Retrieval, 2010): ApproxNDCG
- Cao et al. "Learning to Rank: From Pairwise Approach to Listwise Approach" (ICML 2007): ListNet
- Xia et al. "Listwise Approach to Learning to Rank" (ICML 2008): ListMLE
- Blondel et al. "Fast Differentiable Sorting and Ranking" (ICML 2020): soft ranking methods
- Grover et al. "Stochastic Optimization of Sorting Networks via Continuous Relaxations" (ICLR 2019): NeuralSort
- Prillo & Eisenschlos. "SoftSort: A Continuous Relaxation for the argsort Operator" (ICML 2020): SoftSort

## License

MIT OR Apache-2.0
