# Monitoring output (captured right after the load test)

## Prediction latency

- predictions served: 700
- mean: 0.537 ms, p50: 0.346 ms, p95: 0.6 ms, max: 105.809 ms

## Input drift

- drift detected: **True**
- drifted features: sepal_length
- recent window: 200 inputs

Phase 1 (in-distribution) stayed inside the training baseline; phase 2 shifted sepal_length by +4 cm and the monitor flagged it, as designed.
