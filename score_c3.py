#!/usr/bin/env python3
"""Score C3 dry-tile prompt runs for Qwen and LLaVA.

Reads results/c3/qwen_c3.jsonl and results/c3/llava_c3.jsonl.
Writes C3 metrics, Markdown tables, CSV, and figures.

For direct/evidence/conservative prompts, explicit Yes answers are reported as
an observed false-alarm proxy on dry tiles. The leading_damage prompt is free
text; this script does not claim to semantically score hallucinations for it.
That requires a human-validated LLM judge.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {"qwen": "Qwen2.5-VL-7B", "llava": "LLaVA-1.5-7B"}
BINARY_PROMPTS = ("direct", "evidence", "conservative")
PROMPT_ORDER = ("direct", "evidence", "conservative", "leading_damage")


def parse_answer(text):
    """Parse only explicit Yes/No at the start; free text remains 'other'."""
    words = re.findall(r"[a-z]+", str(text).lower())
    return words[0] if words and words[0] in ("yes", "no") else "other"


def load_jsonl(path, model):
    rows, seen = [], set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            for key in ("answer", "prompt_id", "tile_id"):
                if key not in record:
                    raise ValueError(f"{path}:{line_no}: missing required key {key!r}")

            job_id = record.get(
                "job_id", f'{record["tile_id"]}|{record["prompt_id"]}'
            )
            if job_id in seen:
                continue
            seen.add(job_id)

            answer = str(record["answer"]).strip()
            rows.append({
                "model": model,
                "job_id": job_id,
                "tile_id": str(record["tile_id"]),
                "parent_chip_id": record.get("parent_chip_id"),
                "prompt_id": str(record["prompt_id"]),
                "answer": answer,
                "answer_type": parse_answer(answer),
                "word_count": len(answer.split()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/c3"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir
    out_dir = args.out_dir or results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model in MODELS:
        path = results_dir / f"{model}_c3.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Expected qwen_c3.jsonl and llava_c3.jsonl."
            )
        model_rows = load_jsonl(path, model)
        if not model_rows:
            raise ValueError(f"No predictions found in {path}")
        print(f"Loaded {len(model_rows)} unique jobs for {MODELS[model]} from {path}")
        all_rows.extend(model_rows)

    raw = pd.DataFrame(all_rows)
    results = []
    for (model, prompt_id), group in raw.groupby(["model", "prompt_id"], sort=False):
        n = len(group)
        counts = Counter(group["answer_type"])
        yes, no, other = counts["yes"], counts["no"], counts["other"]
        results.append({
            "model": model,
            "model_name": MODELS[model],
            "prompt_id": prompt_id,
            "n": n,
            "yes_count": yes,
            "no_count": no,
            "other_count": other,
            "yes_rate_pct": round(100 * yes / n, 3) if n else None,
            "no_rate_pct": round(100 * no / n, 3) if n else None,
            "other_rate_pct": round(100 * other / n, 3) if n else None,
            "mean_response_words": round(float(group["word_count"].mean()), 2),
        })

    table = pd.DataFrame(results)
    order = {name: i for i, name in enumerate(PROMPT_ORDER)}
    table["_order"] = table["prompt_id"].map(order).fillna(99)
    table = table.sort_values(["_order", "model"]).drop(columns="_order")
    table.to_csv(out_dir / "c3_model_comparison.csv", index=False)

    metrics = {
        "experiment": "C3 prompt comparison on dry tiles",
        "interpretation": (
            "For binary prompts, yes_rate_pct is the observed share of explicit Yes "
            "responses on dry tiles (a false-alarm proxy). For leading_damage, the "
            "Yes/No/other parser is not a semantic hallucination classifier."
        ),
        "models": {},
        "prompt_results": table.to_dict(orient="records"),
    }
    for model in MODELS:
        subset = raw[raw["model"] == model]
        metrics["models"][model] = {
            "model_name": MODELS[model],
            "total_unique_jobs": int(len(subset)),
            "prompt_ids": sorted(subset["prompt_id"].unique().tolist()),
            "answer_type_counts": {
                str(k): int(v) for k, v in subset["answer_type"].value_counts().items()
            },
        }
    (out_dir / "c3_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table_md = table.drop(columns=["model_name"]).to_markdown(index=False)
    report = "\n".join([
        "# C3 Results: Qwen vs LLaVA",
        "",
        "C3 compares prompt behaviour on dry satellite tiles. For binary prompts,",
        "the observed percentage of explicit Yes answers is a false-alarm proxy.",
        "",
        table_md,
        "",
        "## Interpretation and limitations",
        "- `direct`, `evidence`, and `conservative`: Yes rate counts explicit Yes answers.",
        "- `leading_damage` uses free-text output; `other` means the response did not start with Yes/No.",
        "- The parser does not determine whether free-text responses semantically claim flooding.",
        "- Validate an LLM judge against human labels before reporting a semantic false-claim rate.",
        "- Dry tiles alone cannot estimate flood recall or overall accuracy.",
        "",
    ])
    (out_dir / "c3_table.md").write_text(report, encoding="utf-8")
    (out_dir / "comparison_table.md").write_text(report, encoding="utf-8")

    # Figure 1: observed false-alarm proxy for binary prompts.
    binary = table[table["prompt_id"].isin(BINARY_PROMPTS)]
    if not binary.empty:
        pivot = binary.pivot(index="prompt_id", columns="model", values="yes_rate_pct")
        pivot = pivot.reindex([p for p in BINARY_PROMPTS if p in pivot.index])
        ax = pivot.plot(kind="bar", figsize=(9, 5), rot=0)
        ax.set_title("C3: Observed False-Alarm Proxy on Dry Tiles")
        ax.set_xlabel("Prompt")
        ax.set_ylabel("Explicit Yes responses (%)")
        ax.set_ylim(bottom=0)
        ax.legend(title="Model")
        plt.tight_layout()
        plt.savefig(out_dir / "c3_false_alarm_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Figure 2: answer-type counts by model and prompt.
    chart = table.copy()
    chart["label"] = chart["model"].map(MODELS) + " / " + chart["prompt_id"]
    x = np.arange(len(chart))
    yes = chart["yes_count"].to_numpy()
    no = chart["no_count"].to_numpy()
    other = chart["other_count"].to_numpy()
    fig, ax = plt.subplots(figsize=(max(10, len(chart) * 0.8), 6))
    ax.bar(x, yes, label="Yes")
    ax.bar(x, no, bottom=yes, label="No")
    ax.bar(x, other, bottom=yes + no, label="Other / free text")
    ax.set_xticks(x)
    ax.set_xticklabels(chart["label"], rotation=40, ha="right")
    ax.set_title("C3: Answer-Type Counts by Model and Prompt")
    ax.set_ylabel("Number of responses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "c3_answer_type_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nC3 comparison:")
    print(table.drop(columns=["model_name"]).to_string(index=False))
    print(f"\nSaved outputs in: {out_dir.resolve()}")
    for name in (
        "c3_metrics.json", "c3_table.md", "comparison_table.md",
        "c3_model_comparison.csv", "c3_false_alarm_comparison.png",
        "c3_answer_type_counts.png",
    ):
        if (out_dir / name).exists():
            print(f" - {name}")


if __name__ == "__main__":
    main()
