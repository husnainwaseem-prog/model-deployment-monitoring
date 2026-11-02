"""Basic monitoring: prediction-latency stats + input-drift detection.

Drift check: keep a rolling window of recent input feature vectors. For each
feature, compare the window mean against the training mean; flag the feature
when it drifts more than `threshold` training standard deviations. Simple,
explainable, and enough to catch a shifted input stream.
"""

import threading
from collections import deque

import numpy as np


class LatencyTracker:
    def __init__(self):
        self._latencies = []
        self._lock = threading.Lock()

    def record(self, latency_ms):
        with self._lock:
            self._latencies.append(float(latency_ms))

    def stats(self):
        with self._lock:
            lat = self._latencies
            if not lat:
                return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0,
                        "p95_ms": 0.0, "max_ms": 0.0}
            arr = np.array(lat)
            return {"count": len(arr),
                    "mean_ms": round(float(arr.mean()), 3),
                    "p50_ms": round(float(np.percentile(arr, 50)), 3),
                    "p95_ms": round(float(np.percentile(arr, 95)), 3),
                    "max_ms": round(float(arr.max()), 3)}


class DriftMonitor:
    def __init__(self, train_mean, train_std, window=200, threshold=3.0,
                 min_samples=30):
        self.train_mean = dict(train_mean)
        self.train_std = dict(train_std)
        self.features = list(train_mean)
        self.window = window
        self.threshold = threshold
        self.min_samples = min_samples
        self._recent = deque(maxlen=window)
        self._lock = threading.Lock()

    def update(self, feature_vector):
        with self._lock:
            self._recent.append([float(feature_vector[f]) for f in self.features])

    def check(self):
        with self._lock:
            n = len(self._recent)
            if n < self.min_samples:
                return {"drift": False, "n_recent": n,
                        "reason": f"warming up ({n}/{self.min_samples} samples)"}
            recent_mean = np.array(self._recent).mean(axis=0)
            per_feature = {}
            for i, f in enumerate(self.features):
                deviation = abs(recent_mean[i] - self.train_mean[f])
                limit = self.threshold * self.train_std[f]
                per_feature[f] = {
                    "recent_mean": round(float(recent_mean[i]), 4),
                    "train_mean": self.train_mean[f],
                    "deviation": round(float(deviation), 4),
                    "limit": round(float(limit), 4),
                    "drifted": bool(deviation > limit),
                }
            drifted = [f for f, d in per_feature.items() if d["drifted"]]
            return {"drift": bool(drifted), "n_recent": n,
                    "drifted_features": drifted, "per_feature": per_feature}
