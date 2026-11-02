# Model Deployment with Monitoring — Report

## What was built
A deployment wrapper around a small, honest model: LogisticRegression on the
iris dataset (150 flowers, 4 measurements), test accuracy **0.9474**. The model
is trained by `src/train.py`; the artifact carries the classifier plus the
training feature means/stds that the drift monitor compares against.

## Service (`src/api.py`)
- `POST /predict` — Pydantic validation (each measurement 0–12 cm; wrong types
  or out-of-range values get a 422 before the model ever sees them).
- `GET /health` — liveness + model metadata.
- `GET /monitoring` — latency stats, drift status per feature, training baseline.
- A middleware logs every request as JSON lines
  (`logs/requests.jsonl`): timestamp, method, path, status, latency.

## Monitoring (`src/monitoring.py`)
- **Latency:** every prediction's server-side time is recorded; stats are
  count/mean/p50/p95/max.
- **Drift:** a rolling window of the last 200 inputs; a feature is flagged when
  its window mean deviates more than 3 training standard deviations from the
  training mean. Deliberately simple so the flag is explainable.

## Load test (`scripts/load_test.py`, real run 2026-11-02)
Two phases against the live service, 20 concurrent clients:

| Phase | Requests | Throughput | Mean | p95 | Errors | Drift after |
|---|---|---|---|---|---|---|
| 1 - in-distribution (iris test rows) | 500 | 178.3 r/s | 111.5 ms | 124.4 ms | 0 | none |
| 2 - drifted (sepal_length + 4 cm) | 200 | 116.7 r/s | 160.9 ms | 1220.4 ms | 0 | **sepal_length** |

The monitor's verdict after phase 2: drift = true, sepal_length recent mean
9.73 vs training 5.88 (deviation 3.85 > limit 2.55); the other three features
clean. Server-side prediction latency across the 700 requests: mean 0.54 ms.

Note: client-measured latency exceeds server-side latency because 20 threads
contend for 2 CPUs on the test machine; the endpoint itself is sub-millisecond.

## Docker
`Dockerfile` (python:3.12-slim) installs dependencies, trains the model during
the build for a deterministic artifact, and serves on port 8000. The sandbox
used for this build has no Docker daemon, so the image was not built here —
the service was verified live via the same `uvicorn` command the container runs.

## Reproduce
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.cli train
.venv/bin/python -m pytest tests/ -q        # 11/11
.venv/bin/python -m src.cli serve &         # then:
.venv/bin/python scripts/load_test.py
```
