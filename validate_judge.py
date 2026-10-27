"""Phase 6: Validate LLM Judge on 100 hand-labelled VLM free-text responses.

Computes:
  - Cohen's kappa (inter-rater agreement between human expert and LLM judge)
  - Confusion matrix across ["supported", "unsupported", "uncertain"]
  - Precision, Recall, F1 classification report
  - Outputs results/judge_validation_metrics.json
"""

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

CSV_PATH = Path("results/judge_validation_100.csv")
OUT_JSON = Path("results/judge_validation_metrics.json")


def validate_judge(csv_path=CSV_PATH):
    print("=== Phase 6: Evaluating LLM Judge Validation (100 Labels) ===")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Please ensure validation CSV is created.")

    df = pd.read_csv(csv_path)
    required = {"human_label", "judge_label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}, got {list(df.columns)}")

    df = df.dropna(subset=["human_label", "judge_label"])
    if len(df) != 100:
        raise ValueError(f"Expected 100 labelled examples, got {len(df)}")

    labels = ["supported", "unsupported", "uncertain"]
    kappa = cohen_kappa_score(df["human_label"], df["judge_label"], labels=labels)
    cm = confusion_matrix(df["human_label"], df["judge_label"], labels=labels)
    report = classification_report(
        df["human_label"], df["judge_label"], labels=labels, zero_division=0, output_dict=True
    )
    report_text = classification_report(
        df["human_label"], df["judge_label"], labels=labels, zero_division=0
    )

    print(f"Total labelled examples: {len(df)}")
    print(f"Cohen's kappa: {kappa:.4f}")
    print("\nConfusion Matrix (Rows: Human, Columns: Judge):")
    cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df)

    print("\nClassification Report:")
    print(report_text)

    # Class distributions
    human_dist = df["human_label"].value_counts().to_dict()
    judge_dist = df["judge_label"].value_counts().to_dict()

    metrics = {
        "n_samples": len(df),
        "cohens_kappa": round(float(kappa), 4),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "class_distributions": {
            "human": human_dist,
            "judge": judge_dist,
        },
        "classification_report": report,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved validation metrics to: {OUT_JSON}")
    return metrics


if __name__ == "__main__":
    validate_judge()
