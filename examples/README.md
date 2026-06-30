# rankit examples

Each example answers one question and is runnable as-is. Examples that need a
dataset are **data-gated**: they exit 0 with a message naming the fetch script,
so they are safe to compile/run in CI. A few need feature flags, noted inline.

All outputs below are real, captured from a run.

## Metrics and evaluation

### `eval_metrics` — which IR metrics does rankit compute?

Precision/recall at k, MRR, MAP, and NDCG over a ranked list with known
relevance, for both binary and graded judgments.

```bash
cargo run --release --example eval_metrics
```
```text
=== Binary relevance ===
Ranked:   ["doc_1", "doc_7", "doc_3", "doc_9", "doc_2", ...]
Relevant: {"doc_5", "doc_1", "doc_8", "doc_3"}

  P@1  = 1.000    R@1  = 0.250
  P@3  = 0.667    R@3  = 0.500
  P@5  = 0.400    R@5  = 0.500
  P@10 = 0.400    R@10 = 1.000

  MRR   = 1.000
  MAP   = 0.685
  NDCG@5 = 0.586
```

### `ltr_significance` — is my ranker significantly better, or just luckier?

Pairs rankit's NDCG with [statskit](https://crates.io/crates/statskit): a
Wilcoxon signed-rank test plus a bootstrap confidence interval on the per-query
NDCG difference between two rankers.

```bash
cargo run --release --example ltr_significance
```
```text
LTR significance: rankit NDCG@10 + statskit
  strong ranker mean NDCG@10: 1.0000
  weak   ranker mean NDCG@10: 0.0956
  Wilcoxon: statistic=-0.0, p=1.82e-5, significant=true
  mean NDCG@10 difference: 0.9044 (95% CI [0.8588, 0.9411])
  [PASS] strong ranker significantly beats weak (p<0.05, CI excludes 0)
```

## Differentiable ranking

### `soft_rank` — how do you turn a hard ranking into a differentiable one?

Soft ranks at increasing sharpness `alpha` (recovering the hard permutation in
the limit), and a comparison of the Sigmoid, NeuralSort, and SmoothI methods.

```bash
cargo run --release --example soft_rank
```
```text
Scores: [5.0, 1.0, 2.0, 4.0, 3.0]

alpha=  0.1  ranks=[2.248, 1.752, 1.876, 2.124, 2.000]
alpha=  1.0  ranks=[3.546, 0.454, 1.167, 2.833, 2.000]
alpha= 10.0  ranks=[4.000, 0.000, 1.000, 3.000, 2.000]

Method comparison (alpha=5):
  Sigmoid      [3.993, 0.007, 1.000, 3.000, 2.000]
  NeuralSort   [2.484, 1.516, 1.756, 2.244, 2.000]
  SmoothI      [3.993, 0.007, 1.000, 3.000, 2.000]
```

### `lambdarank` — which documents does LambdaRank push up or down?

The per-document LambdaRank gradient `dC/ds_i`: its sign and magnitude say which
way and how hard each document should move to raise NDCG.

```bash
cargo run --release --example lambdarank
```
```text
Current ranking by score: [1, 4, 0, 2, 3]
NDCG (full list):         0.6538

LambdaRank gradients (default params):
  doc_0: score=0.5, rel=3, lambda=-0.122671 (push UP)
  doc_1: score=0.8, rel=1, lambda=+0.028672 (push DOWN)
  doc_2: score=0.3, rel=2, lambda=+0.014749 (push DOWN)
  ...
```

## Training

### `train_with_schedule` — how do I train a ranker end to end?

A RankNet training loop driven by [descend](https://crates.io/crates/descend)'s
warmup-cosine LR schedule with EMA weight averaging.

```bash
cargo run --release --example train_with_schedule
```
```text
RankNet loss: 0.6931 -> 0.1225 over 200 steps
EMA-averaged model loss: 0.1226
schedule LR: warmup-start=0.0000, peak=0.5000, end=0.0000
  [PASS] loss decreased and the warmup-cosine schedule shaped the LR
```

### `ltr_lightgbm_train_eval` — does training actually help on a real LETOR dataset?

Linear learning-to-rank with LambdaRank gradients on the LightGBM ranking
dataset (300 features), reporting NDCG@10 and MAP before vs after training.

Data-gated: needs the LightGBM ranking dataset (`data/` is gitignored). Fetch it:

```bash
./scripts/fetch_lightgbm_rank.sh
cargo run --release --example ltr_lightgbm_train_eval
```
```text
features: 300  train queries: 201  test queries: 50

baseline (random init)   NDCG@10 = 0.5821   MAP = 0.7116
epoch  1  train NDCG@10 = 0.8094
epoch 30  train NDCG@10 = 0.8286
epoch 60  train NDCG@10 = 0.8287

trained                  NDCG@10 = 0.7629   MAP = 0.8187
improvement              NDCG@10 +0.1807   MAP +0.1071
```

## Pipeline and multi-objective

### `retrieval_pipeline` — what does a full retrieve-and-rank pipeline look like?

Tokenize, BM25-index, score, rank, and evaluate over a small document set.
Needs the `pipeline` and `eval` features.

```bash
cargo run --release --features "pipeline eval" --example retrieval_pipeline
```
```text
=== BM25 (10 docs, avg_dl=9.3) ===

  query: "rust systems programming"
    #1 doc=7  score=   4.163  Rust and C++ compete for systems programming use cases
    #2 doc=0  score=   3.822  Rust is a systems programming language focused on safety ...
    #3 doc=4  score=   1.215  Information retrieval systems combine indexing with relev...
```

### `pareto_ltr` — how do I trade off NDCG against diversity and fairness?

Sweeps ranking configs and marks the Pareto frontier across three competing
objectives, so the trade-off surface is explicit instead of collapsed to one
score.

```bash
cargo run --release --example pareto_ltr
```
```text
=== Multi-Objective LTR via Pareto Frontier ===

  config(temp=0.1, reg=0.0) -> NDCG=0.909 div=0.091 fair=0.410 <- Pareto
  config(temp=1.0, reg=0.0) -> NDCG=0.500 div=0.500 fair=0.500 <- Pareto
  config(temp=1.0, reg=0.5) -> NDCG=0.425 div=0.600 fair=0.700 <- Pareto
  ...
```

## More

`soft_dtw_ranking` (Soft-DTW as a ranking sequence loss) and
`neural_surrogate_loss` (learn a differentiable surrogate for NDCG) round out the
differentiable-ranking set.

## Datasets

`data/` is not tracked. The data-gated examples no-op with a fetch message when
it is absent; `scripts/fetch_lightgbm_rank.sh` downloads the LightGBM ranking set.
