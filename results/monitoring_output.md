# Monitoring output (captured right after the load test)

## Prediction latency

- predictions served: 700
- mean: 2.601 ms, p50: 1.041 ms, p95: 1.605 ms, max: 136.732 ms

## Input drift

- after phase 1 (in-distribution): drift detected **False** (window 200 inputs)
- after phase 2 (shifted): drift detected **True**
- drifted features: sepal_length
- recent window: 200 inputs

Phase 1 is in-distribution (expect no drift); phase 2 shifts sepal_length by +4 cm (expect the monitor to flag it). Both snapshots above are the live /monitoring responses.
