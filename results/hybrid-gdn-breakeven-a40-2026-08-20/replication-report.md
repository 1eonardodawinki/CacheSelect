# Hybrid GDN break-even replication analysis

Decision: **NO_RELIABLE_SPEEDUP**

All outputs exact: **yes**

Intervals are deterministic 95% percentile-bootstrap intervals for the pooled median. A speedup is called replicated only when at least two source runs contribute and both interval lower bounds exceed 1.0.

| Blocks | Trials | Runs | Wall median [95% interval] | TTFT median [95% interval] | Wall wins | Evidence |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | 3 | 1 | 0.992x [0.965, 1.012] | 0.991x [0.956, 1.022] | 1/3 | INCONCLUSIVE |
| 2 | 3 | 1 | 0.964x [0.962, 0.971] | 0.959x [0.956, 0.966] | 0/3 | OBSERVED_SLOWER |
| 4 | 3 | 1 | 0.973x [0.968, 0.981] | 0.970x [0.963, 0.978] | 0/3 | OBSERVED_SLOWER |
| 8 | 3 | 1 | 0.983x [0.979, 0.990] | 0.982x [0.977, 0.990] | 0/3 | OBSERVED_SLOWER |
| 16 | 8 | 2 | 0.997x [0.989, 1.015] | 0.997x [0.988, 1.014] | 3/8 | INCONCLUSIVE |
| 24 | 5 | 1 | 0.984x [0.974, 0.987] | 0.983x [0.972, 0.985] | 0/5 | OBSERVED_SLOWER |
| 32 | 5 | 1 | 1.013x [0.988, 1.036] | 1.012x [0.986, 1.036] | 3/5 | INCONCLUSIVE |
| 48 | 5 | 1 | 1.003x [0.944, 1.041] | 1.002x [0.943, 1.039] | 3/5 | INCONCLUSIVE |
