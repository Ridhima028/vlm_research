# C2 River vs Flood: Extended Evaluation Summary

Comparison of model responses across original satellite images, baseline camouflage fill,
deep LaMa inpainting, and area-matched C1 dry controls.

| Model | Condition | % Yes (Orig) | % Yes (After) | Persistence (95% CI) | McNemar p-value | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B | Baseline Camouflage | 23.3% | 26.7% | 57.1% [40.7, 73.1] | 0.4996 | Completed |
| Qwen2.5-VL-7B | LaMa Inpainting | — | — | Pending GPU inference | — | Pending |
| Qwen2.5-VL-7B | C1 Area-Matched Control | — | — | Pending GPU inference | — | Pending |
| LLaVA-1.5-7B | Baseline Camouflage | 40.7% | 30.7% | 65.6% [52.4, 78.0] | 0.0059 | Completed |
| LLaVA-1.5-7B | LaMa Inpainting | — | — | Pending GPU inference | — | Pending |
| LLaVA-1.5-7B | C1 Area-Matched Control | — | — | Pending GPU inference | — | Pending |

### Methodological Notes:
1. **Bootstrap Unit:** 95% Confidence Intervals are derived from 2,000 resamples over parent chips, preserving spatial clustering.
2. **Exact McNemar Test:** Evaluates paired contingency tables on paired identical tiles.
3. **C1 Validity:** Area-matched dry inpainting is restricted to tiles with verified in-bounds dry land.