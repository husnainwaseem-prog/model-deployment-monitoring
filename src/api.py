"""FastAPI service: validated predictions, health, request logging, monitoring."""

import json
import os
import time
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from .monitoring import DriftMonitor, LatencyTracker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
LOG_DIR = os.path.join(BASE_DIR, "logs")
REQUEST_LOG = os.path.join(LOG_DIR, "requests.jsonl")

app = FastAPI(title="Iris Classifier Deployment")

ARTIFACT = joblib.load(ARTIFACT_PATH)
MODEL = ARTIFACT["model"]
FEATURES = ARTIFACT["feature_names"]
TARGETS = ARTIFACT["target_names"]

latency = LatencyTracker()
drift = DriftMonitor(ARTIFACT["train_mean"], ARTIFACT["train_std"])

os.makedirs(LOG_DIR, exist_ok=True)


class IrisInput(BaseModel):
    sepal_length: float = Field(ge=0.0, le=12.0)
    sepal_width: float = Field(ge=0.0, le=12.0)
    petal_length: float = Field(ge=0.0, le=12.0)
    petal_width: float = Field(ge=0.0, le=12.0)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
             "method": request.method, "path": request.url.path,
             "status_code": response.status_code,
             "latency_ms": round(latency_ms, 3)}
    with open(REQUEST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": "iris-logistic-regression",
            "test_accuracy": ARTIFACT["test_accuracy"]}


@app.post("/predict")
def predict(inp: IrisInput):
    start = time.perf_counter()
    vector = [getattr(inp, f) for f in FEATURES]
    proba = MODEL.predict_proba([vector])[0]
    pred_idx = int(proba.argmax())
    latency_ms = (time.perf_counter() - start) * 1000

    latency.record(latency_ms)
    drift.update(dict(zip(FEATURES, vector)))

    return {"predicted_class": TARGETS[pred_idx],
            "probabilities": {t: round(float(p), 4)
                              for t, p in zip(TARGETS, proba)},
            "latency_ms": round(latency_ms, 3)}


@app.get("/monitoring")
def monitoring():
    return {"latency": latency.stats(),
            "drift": drift.check(),
            "training_baseline": {"mean": ARTIFACT["train_mean"],
                                  "std": ARTIFACT["train_std"],
                                  "n_train": ARTIFACT["n_train"]}}
