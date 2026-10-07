# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
