# Model Deployment with Monitoring

An iris flower classifier (scikit-learn LogisticRegression, test accuracy 0.9474)
packaged as a Docker container behind a FastAPI service, with request logging,
prediction-latency tracking, and input-drift detection against the training data.

## Portal requirements checklist

| Requirement | Where |
|---|---|
| Trained model (own or from a previous task) | `src/train.py` trains it; `models/model.joblib` (baked into the image at build) |
| Docker container | `Dockerfile` (python:3.12-slim, trains at build, serves on port 8000) |
| FastAPI service with input validation | `src/api.py` → `POST /predict` (Pydantic: 4 measurements, 0–12 cm; 422 on bad input) |
| Health endpoint | `GET /health` |
| Request logging | middleware in `src/api.py` → `logs/requests.jsonl` (method, path, status, latency) |
| Monitoring: prediction latency | `src/monitoring.py` → `LatencyTracker`; `GET /monitoring` (count, mean, p50, p95, max) |
| Monitoring: input drift vs training data | `src/monitoring.py` → `DriftMonitor` (rolling 200-input window; flags a feature when its mean drifts > 3 training stds) |
| Load test with results | `scripts/load_test.py`; `results/load_test_table.md`, `results/load_test_results.json` |
| Monitoring output | `results/monitoring_output.md` (captured live after the load test) |
| Code, Dockerfile, monitoring output, README | this repo |

## Results

Load test (700 requests, 20 concurrent, measured 2026-11-02):

| Phase | Requests | Throughput (req/s) | Mean latency | p95 latency | Errors |
|---|---|---|---|---|---|
| 1 - in-distribution | 500 | 178.3 | 111.5 ms | 124.4 ms | 0 |
| 2 - drifted inputs | 200 | 116.7 | 160.9 ms | 1220.4 ms | 0 |

Monitoring after the test: 700 predictions, server-side mean latency **0.54 ms**;
**drift detected on `sepal_length`** (recent mean 9.73 vs training 5.88) after
phase 2 shifted it by +4 cm — the other three features stayed clean.

(Client-measured latency is higher than server-side because 20 threads share
2 CPUs; the endpoint itself is sub-millisecond.)

## Quickstart

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows cmd:        .venv\Scripts\activate
pip install -r requirements.txt
python -m src.cli train          # train + save models/model.joblib
python -m pytest tests/ -q       # 11/11 tests
python -m src.cli serve          # API → http://127.0.0.1:8000
python scripts/load_test.py      # needs the service running
```

Docker (needs a Docker host):

```bash
docker build -t iris-deployment .
docker run -p 8000:8000 iris-deployment
```

## API

- `GET /health` → `{"status": "ok", ...}`
- `POST /predict` → `{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}`
  returns the predicted species + probabilities + latency.
- `GET /monitoring` → latency stats, drift status per feature, training baseline.

## Layout

```
src/        train.py, api.py, monitoring.py, cli.py
scripts/    load_test.py
models/     model.joblib (trained artifact)
results/    load_test_results.json, load_test_table.md, monitoring_output.md
logs/       requests.jsonl (per-request log, git-ignored)
tests/      11 pytest tests
Dockerfile
```
