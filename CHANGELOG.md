# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Source history (unversioned)

These entries identify source changes without assigning a package release.

- Add the LambdaRank overview to the bibliography. This is a documentation-only change.
  ([86a55cc](https://github.com/arclabs561/rankit/commit/86a55cc2aaf857e96c93598cf88ff4043dabd9b7))

## [Unreleased]

### Changed

- Behaviour change: `pipeline::Scoring::JelinekMercerLm { lambda }` now uses
  `lambda` as the collection-model weight, as documented (Zhai and Lafferty
  2001): `p(t|d) = (1 - lambda) * tf / |d| + lambda * p(t|C)`. It was passed
  to `rankfns::lm_smoothed_p` as the document-model weight, so the code
  computed the mirror image of the doc. Scores change for any `lambda` other
  than 0.5; a tuned `lambda` should be replaced by `1 - lambda` to keep
  previous scores.
- `soft_rank`, `soft_rank_sigmoid`, `pairwise_logistic_rank`, and the unsorted
  path of `soft_rank_optimized` now compute the pairwise sum with
  `fynch::soft_rank` instead of a local loop. Results match the previous loop
  to within 1e-10 on finite, tied, and non-finite inputs.
- With an infinite regularization strength (or zero temperature), tied values
  now share their average rank instead of returning NaN.

### Fixed

- Behaviour change: `compute_ranking_svm_gradients` with `cost_sensitivity`
  now takes the rank in `tau = 1 / ln(rank + 2)` from the current scores
  (descending, ties by input order), as `compute_lambdarank_gradients` does.
  It used input positions, so permuting the documents changed their
  gradients. Gradients change whenever input order differs from score order.
- `pipeline::Pipeline` language-model scores now reflect only live documents.
  `add` counted a document's tokens into the collection statistics before a
  duplicate id was rejected, and `remove` never subtracted them, so a failed
  add or a removal changed every later Dirichlet or Jelinek-Mercer score.
  Collection term counts are now read from the postings index.

## [0.1.8] - 2026-10-06

### Changed

- Use `rankops` 0.2 with default features disabled: evaluation uses its metrics
  without pulling in the optional `innr` backend.
- Correct the declared minimum Rust version to 1.89, matching the existing
  dependency requirements, and check external consumer builds in CI.
- Raised the optional `gumbel` dependency minimum to `drawset` 0.1.2 and
  documented its existing Rust 1.75 requirement.

### Fixed

- Show the mean over relevance pairs in the README RankNet formula, matching
  the implemented loss.
- Align the Python binding source compiler requirement with the Rust crate.

## [0.1.6] - 2026-07-07

### Changed

- Switched the optional `gumbel` feature from `kuji` to `drawset`.

## [0.1.5] - 2026-06-11

### Changed

- Switch from fynch::metrics to rankops::metrics

### Fixed

- Fix formatting in neural_surrogate_loss example

## [0.1.4] - 2026-04-06

### Added

- Add neural surrogate loss example
- Add math markup for soft ranking and LTR loss formulas
- Add natural gradient for ranking losses
- Add spearman_loss (fynch integration)
- Add ranklab Python bindings
- Add pipeline feature composing textprep + postings + rankfns
- Add structops Soft-DTW and pare Pareto examples
- Add LICENSE files, docs.rs metadata, fix missing import

### Changed

- Escape underscores in ListMLE formula for GitHub compat
- Wire kuji dependency into gumbel feature, add gradient tests and topk CE
- Delegate Gumbel sampling to kuji, add HashMap import
- Expand ranklab API
- Ranklab API polish
- Ranklab bindings accept numpy arrays
- Exclude soft_dtw example from crate package
- Remove dead _dl parameter from lm_score
- Remove unused HashMap import in eval/export.rs
- Initial crate — differentiable ranking, LTR losses, IR eval

### Fixed

- Fix approx_ndcg position calculation
- Fix keyword length for crates.io
