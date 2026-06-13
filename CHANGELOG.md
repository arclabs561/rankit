# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
