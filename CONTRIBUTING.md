# Contributing

## Development workflow

1. Create a focused branch for one change.
2. Install developer dependencies with `pip install -e ".[dev]"`.
3. Run `ruff check .` and `pytest --cov=tennis_analytics` before opening a pull request.
4. Add tests for every bug fix or behavior change.
5. Keep generated datasets, logs, reports, and model files out of Git.

## Modeling safeguards

- Every feature must be knowable before match start.
- Use walk-forward validation only; do not substitute random splits.
- Test one modeling hypothesis at a time.
- Recheck P1/P2 target balance whenever assignment logic changes.
- Append new evaluation results to the changelog rather than rewriting prior baselines.
