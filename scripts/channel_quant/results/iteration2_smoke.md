## [8] Iteration 2 - Projection/Budget/Hierarchy Sweep

**Approach**: Reused the streamed Spike 5 full-model forward together with Spike 1 quantizers and the new Spike 1 gate-aware activation metric, then ran three follow-up experiments: projection-specific FP8 splits, a 0-100% routing-aware budget sweep, and a hierarchy comparison between expert-only, channel-only, and two-level allocation. Calibration still uses a single 128-token WikiText-2 train pass and evaluation streams the full WikiText-2 test split in fixed-size blocks.

**Per-projection**: The best split is `w1_heavy_0_50` at PPL 29.6147 and approx 28.954 GB. This isolates whether W2 deserves more of the mixed-precision budget than W1 when channels are ranked by the gate-aware metric.

- `uniform_25_25`: PPL 30.3955, memory 27.612 GB.
- `w2_heavy_50_0`: PPL 32.8054, memory 26.270 GB.
- `w2_heavy_40_10`: PPL 29.8913, memory 26.806 GB.
- `w1_heavy_0_50`: PPL 29.6147, memory 28.954 GB.

**Budget sweep**: The best routing-aware per-channel point is 0% FP8 at PPL 30.2850 and 23.585 GB. The sweep traces the Pareto curve from all-FP4 through all-FP8 under the same per-projection policy for both W1 and W2.

**Hierarchy comparison**: The best hierarchy variant is `expert_only` at PPL 30.1547 and 27.612 GB. Comparing this against `expert_only`, `channel_only`, and `aggressive_two_level` shows how much of the gain comes from routing-aware expert budgeting versus channel ranking alone.

- `expert_only`: PPL 30.1547, memory 27.612 GB.
- `channel_only`: PPL 30.5417, memory 27.612 GB.
- `two_level`: PPL 30.3955, memory 27.612 GB.
- `aggressive_two_level`: PPL 31.4023, memory 27.628 GB.

**Verdict**: This iteration tests whether the remaining gap is mostly a projection split problem, a global budget problem, or a hierarchy problem. The resulting JSON records the exact perplexity/memory trade-off for each setting, and the per-projection rows also expose that nominal W1/W2 percentage splits are not perfectly iso-memory once the true tensor sizes are accounted for.
