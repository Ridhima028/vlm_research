"""Phase 7: Statistical scoring for C2 Extended (Baseline Camouflage vs LaMa Inpainting vs C1 Control).

Computes:
  - % Yes with 95% bootstrap CIs resampled by parent chip (2000 resamples, seed 42)
  - Paired persistence: Yes on orig -> still Yes after removal/control
  - Paired switch rate: No on orig -> Yes after removal/control
  - Exact McNemar test with b (Yes->No), c (No->Yes), and two-sided p-value
  - Comparison across:
      1. Original (unmodified)
      2. Baseline Camouflage (speckled dry pixel fill)
      3. LaMa Inpainting (deep background inpainting)
      4. C1 Control (area-matched inpainting of dry land)

Outputs:
  results/c2_lama/comparison_table.md
  results/c2_lama/metrics.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import config

MODELS = {"qwen": "Qwen2.5-VL-7B", "llava": "LLaVA-1.5-7B"}
N_BOOTSTRAP = 2000
C2_DIR = config.DATA_DIR / "c2"
C2_META = C2_DIR / "metadata.json"
LAMA_MANIFEST = C2_DIR / "lama" / "manifest.json"
RESULTS_BASELINE = config.RESULTS_DIR / "c2"
RESULTS_LAMA = config.RESULTS_DIR / "c2_lama"


def parse_answer(text):
    words = re.findall(r"[a-z]+", str(text).lower())
    if words and words[0] in ("yes", "no"):
        return words[0]
    return "other"


def parent_of(tile_id):
    return tile_id.split("_r")[0]


def load_answers(path):
    answers = defaultdict(dict)
    counts = Counter()
    if not path.exists():
        return answers, counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["condition"], r["prompt_id"])
            if key in answers[r["tile_id"]]:
                continue
            a = parse_answer(r.get("answer", ""))
            answers[r["tile_id"]][key] = a
            counts[a] += 1
    return answers, counts


def is_yes(ans, cond, prompt):
    return ans.get((cond, prompt)) == "yes"


def answered(ans, cond, prompt):
    return (cond, prompt) in ans


def exact_mcnemar(before_yes, after_yes):
    """b = 1->0 (Yes->No), c = 0->1 (No->Yes)."""
    b = int(((before_yes == 1) & (after_yes == 0)).sum())
    c = int(((before_yes == 0) & (after_yes == 1)).sum())
    p = float(binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0)
    return {"b_yes_to_no": b, "c_no_to_yes": c, "p_value": p}


def score_paired_condition(tiles, answers, group_parents, boot_idx, before_cond, after_cond, prompt="yn_flood"):
    """Scores paired transition between before_cond and after_cond for mixed tiles."""
    parents = group_parents["mixed"]
    p_index = {p: i for i, p in enumerate(parents)}

    # Collect paired values: [parent_idx, orig_yes, after_yes, valid_pair]
    paired_data = []
    for tid, t in tiles.items():
        if t["group"] != "mixed":
            continue
        ans = answers.get(tid, {})
        if answered(ans, before_cond, prompt) and answered(ans, after_cond, prompt):
            by = 1 if is_yes(ans, before_cond, prompt) else 0
            ay = 1 if is_yes(ans, after_cond, prompt) else 0
            paired_data.append((p_index[parent_of(tid)], by, ay))

    if not paired_data:
        return {"status": "pending", "n_pairs": 0}

    n_pairs = len(paired_data)
    by_arr = np.array([x[1] for x in paired_data], dtype=int)
    ay_arr = np.array([x[2] for x in paired_data], dtype=int)
    mcn = exact_mcnemar(by_arr, ay_arr)

    # Persistence: after == 1 among before == 1
    # Switch: after == 1 among before == 0
    # Parent-bootstrap for persistence
    counts = np.zeros((len(parents), 4))  # [pers_num, pers_den, switch_num, switch_den]
    for p_i, by, ay in paired_data:
        if by == 1:
            counts[p_i, 1] += 1
            if ay == 1:
                counts[p_i, 0] += 1
        else:
            counts[p_i, 3] += 1
            if ay == 1:
                counts[p_i, 2] += 1

    # Bootstrap CIs
    boot = counts[boot_idx["mixed"]].sum(axis=1)  # (N_BOOTSTRAP, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        pers_boot = np.where(boot[:, 1] > 0, boot[:, 0] / boot[:, 1], np.nan)
        switch_boot = np.where(boot[:, 3] > 0, boot[:, 2] / boot[:, 3], np.nan)

    pers_num, pers_den, switch_num, switch_den = counts.sum(axis=0)

    pers_ci = (
        [round(100 * x, 1) for x in np.nanpercentile(pers_boot, [2.5, 97.5])]
        if pers_den > 0
        else [None, None]
    )
    switch_ci = (
        [round(100 * x, 1) for x in np.nanpercentile(switch_boot, [2.5, 97.5])]
        if switch_den > 0
        else [None, None]
    )

    return {
        "status": "completed",
        "n_pairs": n_pairs,
        "before_yes_rate": round(100 * float(by_arr.mean()), 1),
        "after_yes_rate": round(100 * float(ay_arr.mean()), 1),
        "persistence": {
            "numerator": int(pers_num),
            "denominator": int(pers_den),
            "rate": round(100 * pers_num / pers_den, 1) if pers_den > 0 else None,
            "ci95": pers_ci,
        },
        "switch_to_yes": {
            "numerator": int(switch_num),
            "denominator": int(switch_den),
            "rate": round(100 * switch_num / switch_den, 1) if switch_den > 0 else None,
            "ci95": switch_ci,
        },
        "mcnemar": mcn,
    }


def main():
    print("=== Phase 7: Scoring C2 Extended Experiments ===")
    if not C2_META.exists():
        raise FileNotFoundError(f"Missing {C2_META}")

    with open(C2_META, encoding="utf-8") as f:
        meta_obj = json.load(f)
        tiles = meta_obj["tiles"]

    # Valid C1 tiles from manifest
    valid_c1_tiles = set()
    if LAMA_MANIFEST.exists():
        with open(LAMA_MANIFEST, encoding="utf-8") as f:
            for rec in json.load(f):
                if rec.get("control_status") == "ok":
                    valid_c1_tiles.add(rec["tile_id"])

    # Setup parent grouping & bootstrap indices
    group_parents = defaultdict(lambda: defaultdict(list))
    for tid, t in tiles.items():
        group_parents[t["group"]][parent_of(tid)].append(tid)

    group_p_lists = {g: sorted(group_parents[g].keys()) for g in group_parents}
    rng = np.random.default_rng(config.SEED)
    boot_idx = {
        g: rng.integers(0, len(ps), size=(N_BOOTSTRAP, len(ps)))
        for g, ps in group_p_lists.items()
    }

    all_metrics = {}
    for mkey, mname in MODELS.items():
        base_path = RESULTS_BASELINE / f"{mkey}.jsonl"
        lama_path = RESULTS_LAMA / f"{mkey}_lama.jsonl"

        base_ans, _ = load_answers(base_path)
        lama_ans, _ = load_answers(lama_path)

        # Merge answers: lama answers complement or override
        merged_ans = defaultdict(dict)
        for tid in set(list(base_ans.keys()) + list(lama_ans.keys())):
            merged_ans[tid].update(base_ans.get(tid, {}))
            merged_ans[tid].update(lama_ans.get(tid, {}))

        # 1. Baseline Camouflage (orig vs flood_only_removed)
        camo_res = score_paired_condition(
            tiles, merged_ans, group_p_lists, boot_idx, "orig", "flood_only_removed"
        )

        # 2. LaMa Inpainting (orig vs flood_only_removed_lama)
        lama_res = score_paired_condition(
            tiles, merged_ans, group_p_lists, boot_idx, "orig", "flood_only_removed_lama"
        )

        # 3. C1 Area-Matched Control (orig vs c1_area_matched_lama)
        # Filter to tiles with valid dry location
        c1_tiles = {tid: t for tid, t in tiles.items() if tid in valid_c1_tiles}
        c1_res = score_paired_condition(
            c1_tiles, merged_ans, group_p_lists, boot_idx, "orig", "c1_area_matched_lama"
        )

        all_metrics[mkey] = {
            "model_name": mname,
            "baseline_camouflage": camo_res,
            "lama_inpainting": lama_res,
            "c1_area_matched_control": c1_res,
        }

    # Format Markdown Table
    RESULTS_LAMA.mkdir(parents=True, exist_ok=True)
    out_table = RESULTS_LAMA / "comparison_table.md"
    out_json = RESULTS_LAMA / "metrics.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to: {out_json}")

    lines = [
        "# C2 River vs Flood: Extended Evaluation Summary",
        "",
        "Comparison of model responses across original satellite images, baseline camouflage fill,",
        "deep LaMa inpainting, and area-matched C1 dry controls.",
        "",
        "| Model | Condition | % Yes (Orig) | % Yes (After) | Persistence (95% CI) | McNemar p-value | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for mkey, mdata in all_metrics.items():
        mname = mdata["model_name"]
        for cond_name, cdata in [
            ("Baseline Camouflage", mdata["baseline_camouflage"]),
            ("LaMa Inpainting", mdata["lama_inpainting"]),
            ("C1 Area-Matched Control", mdata["c1_area_matched_control"]),
        ]:
            if cdata.get("status") == "completed":
                pers = cdata["persistence"]
                pers_str = f"{pers['rate']:.1f}% [{pers['ci95'][0]}, {pers['ci95'][1]}]"
                mcn_p = f"{cdata['mcnemar']['p_value']:.4f}"
                lines.append(
                    f"| {mname} | {cond_name} | {cdata['before_yes_rate']}% | {cdata['after_yes_rate']}% | {pers_str} | {mcn_p} | Completed |"
                )
            else:
                lines.append(
                    f"| {mname} | {cond_name} | — | — | Pending GPU inference | — | Pending |"
                )

    lines.append("")
    lines.append("### Methodological Notes:")
    lines.append("1. **Bootstrap Unit:** 95% Confidence Intervals are derived from 2,000 resamples over parent chips, preserving spatial clustering.")
    lines.append("2. **Exact McNemar Test:** Evaluates paired contingency tables on paired identical tiles.")
    lines.append("3. **C1 Validity:** Area-matched dry inpainting is restricted to tiles with verified in-bounds dry land.")

    with open(out_table, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary table saved to: {out_table}")


if __name__ == "__main__":
    main()
