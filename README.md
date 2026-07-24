# Tennis Analytics Platform

[![CI](https://github.com/kwameaa-del/tennis-analytics-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/kwameaa-del/tennis-analytics-platform/actions/workflows/ci.yml)
[![Docker Build](https://github.com/kwameaa-del/tennis-analytics-platform/actions/workflows/docker.yml/badge.svg)](https://github.com/kwameaa-del/tennis-analytics-platform/actions/workflows/docker.yml)

A production-oriented Python analytics platform for engineering, evaluating, and deploying probabilistic ATP and WTA tennis forecasting models using reproducible machine learning workflows.

---

# Why this project exists

Sports data is inherently time ordered. A random train/test split can unintentionally expose future information during model training, producing overly optimistic results. This project demonstrates how to build a reproducible sports analytics pipeline that avoids data leakage through strict walk-forward validation, pre-match feature engineering, and proper probabilistic evaluation.

The goal is not simply to generate predictions—it is to demonstrate sound software engineering and machine learning practices that produce trustworthy, reproducible results.

---

# Technologies

- Python 3.13
- pandas
- NumPy
- scikit-learn
- pytest
- Git
- GitHub Actions
- Docker
- Linux
- VS Code

**Planned**

- Microsoft Azure Virtual Machines
- Docker Compose Production Deployment

---

# Engineering Highlights

- Installable `src/` Python package with typed configuration objects
- Per-year data caching, retries, atomic writes, and explicit failure handling
- Pre-match overall Elo, surface Elo, recent form, and ranking-derived features
- Seeded P1/P2 framing to prevent winner/loser column leakage
- Walk-forward validation using log loss, Brier score, and accuracy
- Persisted scikit-learn model pipelines with preprocessing included
- Rotating application logs and append-only forecast audit records
- Automated tests, linting, coverage checks, and Docker builds in GitHub Actions
- Non-root Docker runtime
- Linux systemd deployment assets
- Modular project architecture designed for maintainability

---

# Software Engineering Practices

This repository follows modern software engineering principles:

- Modular package architecture
- Separation of concerns
- Configuration-driven execution
- Structured logging
- Comprehensive exception handling
- Automated testing with pytest
- Continuous Integration using GitHub Actions
- Containerized deployment with Docker
- Reproducible machine learning workflow
- Version-controlled development with Git

---

# Architecture

```text
External historical data
        │
        ▼
Per-year cache and validation
        │
        ▼
Pre-match feature engineering
        │
        ├── Overall Elo
        ├── Surface Elo
        ├── Recent Form
        └── Ranking Features
        │
        ▼
Walk-forward Evaluation
        │
        ├── Accuracy
        ├── Log Loss
        ├── Brier Score
        │
        ▼
Persisted Model Artifact
        │
        ▼
Forecast Generation
```

---

# Project Structure

```text
.github/workflows/         Continuous Integration pipelines
config/                    Runtime configuration
deployment/                Docker and systemd deployment assets
scripts/                   Command-line entry points
src/tennis_analytics/      Core application package
tests/                     Automated test suite
data/                      Generated datasets (excluded from Git)
models/                    Trained models (excluded from Git)
reports/                   Evaluation reports (excluded from Git)
logs/                      Rotating application logs (excluded from Git)
```

---

# Installation

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

---

# Running the Pipeline

Run the complete analytics pipeline

```bash
python scripts/run_daily_pipeline.py --tours atp wta
```

Use existing downloaded datasets

```bash
python scripts/run_daily_pipeline.py --tours atp --skip-download
```

Run individual stages

```bash
python scripts/download_data.py atp
python scripts/build_features.py atp
python scripts/evaluate.py atp
python scripts/train_model.py atp
```

---

# Testing

Run the automated test suite

```bash
pytest
```

Run with coverage

```bash
pytest --cov=tennis_analytics --cov-report=term-missing
```

Run linting

```bash
ruff check .
```

---

# Docker

Build

```bash
docker build -f deployment/Dockerfile -t tennis-analytics .
```

Run

```bash
docker run --rm \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/reports:/app/reports" \
-v "$(pwd)/logs:/app/logs" \
-v "$(pwd)/models:/app/models" \
tennis-analytics
```

Or

```bash
docker compose -f deployment/docker-compose.yml up --build
```

---

# Validation Rules

The project follows five core validation principles:

1. Features must be knowable before match start.
2. Training data must always precede testing data.
3. P1/P2 framing must remain statistically balanced.
4. Only one modeling hypothesis is changed at a time.
5. Historical evaluation results remain immutable.

---

# Current Limitations

- Historical data source availability depends on the upstream provider.
- Upcoming scheduled matches are not yet automatically ingested.
- Injury reports, recovery time, travel, and player fatigue are not yet modeled.
- Market probabilities are retained only as an external evaluation benchmark.

---

# Future Improvements

- Live scheduled match ingestion
- REST API
- Interactive web dashboard
- Microsoft Azure deployment
- Model monitoring
- Performance dashboards
- Calibration visualization
- Multiple model comparison framework

---

# Responsible Use

This repository is an educational sports analytics project.

Its purpose is to demonstrate reproducible machine learning workflows, software engineering practices, and probabilistic forecasting techniques.

Forecasts generated by this software represent statistical probability estimates only. They should not be interpreted as guarantees or financial advice.

---

# License

This project is released under the MIT License.