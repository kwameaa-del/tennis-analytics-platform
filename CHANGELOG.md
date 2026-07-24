# Changelog

## 0.2.0 — Production engineering refactor

- Added validated, typed configuration via a `Settings` dataclass.
- Added project-specific exceptions for configuration, data, and evaluation failures.
- Added rotating file logging with consistent structured messages.
- Added atomic yearly-cache writes and clearer download failure behavior.
- Added persisted scikit-learn model artifacts with preprocessing inside a pipeline.
- Added tests for configuration, downloads, feature generation, evaluation, logging, and model training.
- Added GitHub Actions workflows for Python 3.11/3.12 testing, Ruff linting, coverage, and Docker builds.
- Hardened the Docker image with a non-root user, persistent volumes, and a health check.
- Expanded documentation for architecture, local execution, CI, Docker, and modeling safeguards.

## 0.1.0 — Portfolio refactor

- Reframed the repository as a sports analytics and probabilistic forecasting platform.
- Removed bankroll, staking, payout, compounding, and cash-out functionality.
- Reorganized code into an installable `src/` package.
- Added configuration, structured CLI scripts, tests, and deployment assets.
- Renamed market-derived fields to external benchmark terminology.
- Added Brier score and accuracy alongside log loss.
- Preserved walk-forward validation and pre-match leakage controls.

## Historical baseline retained from the original research

ATP walk-forward average log loss:

- External closing-probability benchmark: 0.5886
- Tennis-only logistic model: 0.6181
- Tennis plus external benchmark: 0.5908

WTA walk-forward average log loss:

- External closing-probability benchmark: 0.5891
- Tennis-only logistic model: 0.6174
- Tennis plus external benchmark: 0.5912

These values remain historical reference points. New evaluations should be appended rather than silently replacing earlier reported results.
