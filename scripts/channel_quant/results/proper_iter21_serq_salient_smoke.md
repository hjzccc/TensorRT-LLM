## [24] Iteration 21 - SERQ Salient Correction
**Approach**: Reused the strongest existing bases as the quantized scaffold, formed the quantization residual `R = W - Q(W)` after the base FP8/FP4 assignment, and then kept exact residual add-back only on a sparse global top-k of non-FP8 W1 pairs and W2 channels. This is a SERQ-style simulation in the standard BF16 eval path: base weights still use quantize-dequantize FP4/FP8, while the selected salient rows/channels receive an exact add-back term before the MoE `F.linear` calls. All salient masks are ranked by the Iteration 10 router-affinity per-channel cache so the correction budget targets high-impact routed channels rather than broad residual coverage.
**Eval**: Full WikiText-2 test (297193 tokens, 145 chunks of 2048), BF16 `F.linear`, FP32 loss.
**Result**:
| Config | PPL | Delta vs best prior | Memory GB | FP8 frac | Exact corr frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| `serq_joint_salient_1pct` | 6.5798 | +0.0073 | 28.578 | 0.2700 | 0.0100 |
**Winner**: `serq_joint_salient_1pct` at PPL 6.5798, memory 28.578 GB, FP8 fraction 0.2700, exact correction fraction 0.0100.
**Insight**: This isolates the algorithmic value of a very sparse exact residual path. If it helps, the gain comes from spending high-precision budget only where router-weighted quantization error is most consequential, rather than promoting whole projections or carrying a low-precision residual everywhere.
**Next**: If one of the 1% or 3% SERQ runs is competitive, the next obvious sweep is to keep the same exact-correction mechanism but compare router-affinity saliency against router-gap saliency on the same union base so the correction ranking and the base routing heuristic can be separated cleanly.
