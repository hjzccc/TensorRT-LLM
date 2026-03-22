## [9] Iteration 3 - Related Work Comparison + Alternative Strategies

**Approach**: Reused the Iteration 2 streamed perplexity loop, Spike 1 quantizers, and the gate-aware activation metric, then added two follow-ups: a related-work simulation at the 25% average-FP8 budget and a set of unconstrained alternative assignment rules. Calibration remains a single 128-token WikiText-2 train pass, evaluation stays on the first 512 WikiText-2 test tokens, and all methods stream one layer at a time so the full model is never resident at once.

**Related work comparison**: The best related-work baseline in this run is `ours_two_level` at PPL 5.3428 and 27.612 GB, while our reused two-level hierarchy stays at PPL 5.3428 and 27.612 GB.

| Method | PPL | Memory (GB) |
|--------|-----|-------------|
| MxMoE-style per-block | 16.0565 | 27.612 |
| DynaExq per-expert | 5.3739 | 27.612 |
| FGMP per-16-block | 15.1703 | 27.657 |
| ScaleBITS submodular | 15.8179 | 27.612 |
| Random baseline (3-seed mean) | 15.0375 | 27.612 |
| Our two-level hierarchy | 5.3428 | 27.612 |

**Alternative strategies**: The best unconstrained strategy here is `sensitivity_gap` at PPL 14.9518 and 23.833 GB. Threshold rules expose how much the score distribution itself wants to spend, the k-means split tests adaptive per-expert cluster sizes, and the gap rule checks whether sharp elbows exist in the score spectra.

| Method | PPL | Memory (GB) |
|--------|-----|-------------|
| Threshold @ median | 15.9585 | 26.336 |
| Threshold @ p75 | 15.9585 | 26.336 |
| Threshold @ p90 | 16.8520 | 24.326 |
| K-means 2-cluster | 16.1715 | 24.104 |
| Sensitivity gap | 14.9518 | 23.833 |

**Verdict**: This iteration answers two practical questions: whether the gain of the two-level hierarchy survives comparison against prior mixed-precision assignment ideas, and whether a different decision rule on the same gate-aware signal can outperform simple top-k routing-aware allocation. The related-work rows isolate granularity and allocation policy effects, while the alternative rows show whether the score distribution prefers fixed-budget, thresholded, clustered, or gap-based splits.
