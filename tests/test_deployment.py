"""Tests: training, validation, health, monitoring, drift, logging."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from src.monitoring import DriftMonitor, LatencyTracker
from src.train import train
from scripts.load_test import compute_stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TRAIN_MEAN = {"sepal_length": 5.8, "sepal_width": 3.05,
              "petal_length": 3.75, "petal_width": 1.2}
TRAIN_STD = {"sepal_length": 0.83, "sepal_width": 0.43,
             "petal_length": 1.76, "petal_width": 0.76}
GOOD = {"sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2}


def test_train_accuracy_and_artifact():
    artifact = train()
    assert artifact["test_accuracy"] >= 0.9
    assert os.path.exists(os.path.join(BASE_DIR, "models", "model.joblib"))
    assert set(artifact["train_mean"]) == set(FEATURES)


def _client():
    from src import api
    return TestClient(api.app)


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["test_accuracy"] >= 0.9


def test_predict_valid():
    r = _client().post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "setosa"
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert body["latency_ms"] >= 0


def test_predict_rejects_bad_input():
    c = _client()
    assert c.post("/predict", json={}).status_code == 422          # missing fields
    bad = dict(GOOD, sepal_length=-1.0)
    assert c.post("/predict", json=bad).status_code == 422         # out of range
    bad = dict(GOOD, petal_width="huge")
    assert c.post("/predict", json=bad).status_code == 422         # wrong type


def test_monitoring_endpoint_shape():
    body = _client().get("/monitoring").json()
    assert "latency" in body and "drift" in body and "training_baseline" in body
    assert body["latency"]["count"] >= 1  # the /predict calls above were tracked


def test_request_log_written():
    _client().get("/health")
    log = os.path.join(BASE_DIR, "logs", "requests.jsonl")
    assert os.path.exists(log)
    with open(log) as f:
        lines = f.readlines()
    assert lines, "request log is empty"
    entry = json.loads(lines[-1])
    assert {"timestamp", "method", "path", "status_code", "latency_ms"} <= set(entry)


def test_latency_stats_math():
    t = LatencyTracker()
    assert t.stats()["count"] == 0
    for v in [1.0, 2.0, 3.0, 4.0]:
        t.record(v)
    s = t.stats()
    assert s["count"] == 4 and s["mean_ms"] == 2.5 and s["max_ms"] == 4.0


def test_drift_monitor_warming_up():
    m = DriftMonitor(TRAIN_MEAN, TRAIN_STD)
    m.update(GOOD)
    assert m.check()["drift"] is False
    assert "warming up" in m.check()["reason"]


def test_drift_monitor_no_drift_on_normal_data():
    m = DriftMonitor(TRAIN_MEAN, TRAIN_STD)
    for _ in range(50):
        m.update(GOOD)
    assert m.check()["drift"] is False


def test_drift_monitor_flags_shifted_data():
    m = DriftMonitor(TRAIN_MEAN, TRAIN_STD)
    shifted = dict(GOOD, sepal_length=GOOD["sepal_length"] + 4.0)
    for _ in range(50):
        m.update(shifted)
    out = m.check()
    assert out["drift"] is True
    assert "sepal_length" in out["drifted_features"]


def test_load_test_stats_helper():
    s = compute_stats([10.0, 20.0, 30.0])
    assert s["count"] == 3 and s["mean_ms"] == 20.0 and s["p50_ms"] == 20.0
