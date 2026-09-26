# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-26

### Added

- Python 3.8 support. pip selects scikit-learn 1.3 on 3.8 and 1.6+ on 3.9+.
  Input checks go through a small adapter; `fit`, `predict`, and screening are
  unchanged.

## [0.1.0] - 2026-09-26

### Added

- SIS, FPSIS, FPSIS-BIC, PPIS and TPPIS estimators implementing Tanaka and Matsui (2023).
- Shared spectral kernel, BIC-type criterion, and joint grid search over `d`, `alpha` and `k`.
- Simulation generators for the paper's Examples 1 to 4, plus screening metrics.
- scikit-learn compatible `fit` / `transform` / `get_support` API.

