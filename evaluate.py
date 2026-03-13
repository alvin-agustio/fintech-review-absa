"""
Experiment Evaluation & Comparison
====================================
Compare: Baseline vs LoRA vs Retrained models.
Generates comprehensive evaluation summary.

Usage:
    python evaluate.py
"""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import MODELS_DIR, DATA_PROCESSED


LABEL_ORDER = ["Negative", "Neutral", "Positive"]
EPOCH_DIR_PATTERN = re.compile(r"^epoch_(\d+)$")


def has_eval_artifacts(exp_dir: Path) -> bool:
    if not exp_dir.exists():
        return False
    return any(
        (exp_dir / name).exists()
        for name in [
            "metrics.json",
            "baseline_metrics.json",
            "filtered_metrics.json",
            "test_predictions.csv",
            "baseline_test_predictions.csv",
            "filtered_test_predictions.csv",
        ]
    )


def find_candidate_roots(model_name: str) -> list[Path]:
    roots: list[Path] = []

    active = MODELS_DIR / model_name
    if active.exists():
        roots.append(active)

    archive_root = MODELS_DIR / "archive"
    if archive_root.exists():
        snapshots = sorted(
            [p for p in archive_root.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            candidate = snapshot / model_name
            if candidate.exists():
                roots.append(candidate)

    return roots


def resolve_experiment_dir(name: str) -> Path:
    active = MODELS_DIR / name
    if has_eval_artifacts(active):
        return active

    for candidate_root in find_candidate_roots(name):
        if candidate_root == active:
            continue
        if has_eval_artifacts(candidate_root):
            print(f"  [INFO] Using archived artifacts for {name}: {candidate_root}")
            return candidate_root

    return active


def load_json_from_candidates(paths: list[Path]) -> dict | None:
    for path in paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    print(f"  [SKIP] None of these files found: {[str(path) for path in paths]}")
    return None


def resolve_metrics(exp_dir: Path) -> dict | None:
    return load_json_from_candidates([
        exp_dir / "metrics.json",
        exp_dir / "baseline_metrics.json",
        exp_dir / "filtered_metrics.json",
    ])


def resolve_predictions(exp_dir: Path) -> Path | None:
    for path in [
        exp_dir / "test_predictions.csv",
        exp_dir / "baseline_test_predictions.csv",
        exp_dir / "filtered_test_predictions.csv",
    ]:
        if path.exists():
            return path
    return None


def numeric_or_none(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def display_or_na(value, float_digits: int | None = None) -> str:
    if value is None:
        return "N/A"
    if float_digits is not None and isinstance(value, (int, float)):
        return f"{value:.{float_digits}f}"
    return str(value)


def round_float_dict(values: dict) -> dict:
    rounded = {}
    for key, value in values.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return rounded


def compute_ece(df: pd.DataFrame, n_bins: int = 10) -> float | None:
    prob_cols = ["prob_negative", "prob_neutral", "prob_positive"]
    if not all(col in df.columns for col in prob_cols):
        return None

    probs = df[prob_cols].copy()
    probs.columns = LABEL_ORDER
    confidences = probs.max(axis=1)
    predicted_labels = probs.idxmax(axis=1)
    correctness = (predicted_labels == df["label"]).astype(float)

    bins = pd.cut(confidences, bins=n_bins, labels=False, include_lowest=True)
    ece = 0.0
    total = len(df)

    for bin_idx in range(n_bins):
        mask = bins == bin_idx
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correctness[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_prediction_diagnostics(df: pd.DataFrame) -> dict:
    y_true = df["label"]
    y_pred = df["pred_label"]

    overall_report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    overall_report = {key: round_float_dict(value) if isinstance(value, dict) else round(value, 4)
                      for key, value in overall_report.items()}

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    confusion = {
        actual: {pred: int(cm[i][j]) for j, pred in enumerate(LABEL_ORDER)}
        for i, actual in enumerate(LABEL_ORDER)
    }

    per_aspect = {}
    if "aspect" in df.columns:
        for aspect, aspect_df in df.groupby("aspect"):
            aspect_true = aspect_df["label"]
            aspect_pred = aspect_df["pred_label"]
            per_aspect[aspect] = {
                "n_samples": int(len(aspect_df)),
                "accuracy": round(float(accuracy_score(aspect_true, aspect_pred)), 4),
                "f1_macro": round(float(f1_score(aspect_true, aspect_pred, average="macro", labels=LABEL_ORDER, zero_division=0)), 4),
                "f1_weighted": round(float(f1_score(aspect_true, aspect_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)), 4),
                "label_distribution": {str(k): int(v) for k, v in aspect_df["label"].value_counts().to_dict().items()},
                "prediction_distribution": {str(k): int(v) for k, v in aspect_df["pred_label"].value_counts().to_dict().items()},
            }

    error_df = df[df["label"] != df["pred_label"]].copy()
    error_transitions = []
    if not error_df.empty:
        trans = (
            error_df.groupby(["label", "pred_label"]).size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        error_transitions = trans.head(10).to_dict("records")
        for row in error_transitions:
            row["count"] = int(row["count"])

    ece = compute_ece(df)

    diagnostics = {
        "n_predictions": int(len(df)),
        "n_errors": int((df["label"] != df["pred_label"]).sum()),
        "error_rate": round(float((df["label"] != df["pred_label"]).mean()), 4),
        "class_report": overall_report,
        "confusion_matrix": confusion,
        "per_aspect": per_aspect,
        "top_error_transitions": error_transitions,
        "ece": round(ece, 6) if ece is not None else None,
        "ece_available": ece is not None,
    }
    return diagnostics


def collect_epoch_dirs(model_name: str) -> list[tuple[str, Path, int]]:
    collected: list[tuple[str, Path, int]] = []
    seen_paths: set[Path] = set()

    for root in find_candidate_roots(model_name):
        source = "active" if root.parent == MODELS_DIR else root.parent.name
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = EPOCH_DIR_PATTERN.match(child.name)
            if not match:
                continue
            if child in seen_paths:
                continue
            seen_paths.add(child)
            collected.append((source, child, int(match.group(1))))

    return sorted(collected, key=lambda item: (item[2], item[0], str(item[1])))


def collect_epoch_results() -> pd.DataFrame:
    rows = []

    for model_name in ["baseline", "lora"]:
        for source, epoch_dir, epochs in collect_epoch_dirs(model_name):
            metrics = resolve_metrics(epoch_dir)
            if metrics is None:
                continue
            rows.append({
                "model": model_name,
                "epochs": epochs,
                "source": source,
                "output_dir": str(epoch_dir),
                "accuracy": metrics.get("test_accuracy"),
                "f1_macro": metrics.get("test_f1_macro"),
                "f1_weighted": metrics.get("test_f1_weighted"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "trainable_params": metrics.get("trainable_params"),
                "trainable_pct": metrics.get("trainable_pct"),
                "n_train": metrics.get("n_train"),
                "n_test": metrics.get("n_test"),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["epochs", "model", "source"]).reset_index(drop=True)
    for col in ["accuracy", "f1_macro", "f1_weighted", "training_time_seconds", "trainable_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_epoch_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    subset = df[[
        "model",
        "epochs",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "training_time_seconds",
        "trainable_params",
    ]].copy()

    wide = subset.pivot(index="epochs", columns="model")
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index().sort_values("epochs")
    return wide


def print_epoch_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("\n[EPOCH] No epoch sweep artifacts found.")
        return

    printable = df.copy()
    for col in ["accuracy", "f1_macro", "f1_weighted"]:
        printable[col] = printable[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    printable["training_time_seconds"] = printable["training_time_seconds"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    printable["trainable_params"] = printable["trainable_params"].map(
        lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
    )

    print("\n[EPOCH] Baseline vs LoRA per epoch:")
    print(
        printable[[
            "model",
            "epochs",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "training_time_seconds",
            "trainable_params",
            "source",
        ]].to_string(index=False)
    )

    best_idx = df["f1_macro"].astype(float).idxmax()
    best_row = df.loc[best_idx]
    print(
        "\n[EPOCH] Best F1-macro run: "
        f"{best_row['model']} epoch={int(best_row['epochs'])} "
        f"f1_macro={best_row['f1_macro']:.4f} "
        f"accuracy={best_row['accuracy']:.4f}"
    )


def main():
    output_dir = DATA_PROCESSED / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "baseline": resolve_experiment_dir("baseline"),
        "lora": resolve_experiment_dir("lora"),
        "retrained": resolve_experiment_dir("retrained"),
    }

    results = {}
    detailed_results = {}

    for name, exp_dir in experiments.items():
        metrics = resolve_metrics(exp_dir)
        if metrics is None:
            continue

        predictions_path = resolve_predictions(exp_dir)
        diagnostics = None
        if predictions_path is not None:
            diagnostics = compute_prediction_diagnostics(pd.read_csv(predictions_path))

        results[name] = {
            "accuracy": metrics["test_accuracy"],
            "f1_macro": metrics["test_f1_macro"],
            "f1_weighted": metrics["test_f1_weighted"],
            "n_train": metrics.get("n_train"),
            "n_test": metrics.get("n_test"),
            "total_params": metrics.get("total_params"),
            "trainable_params": metrics.get("trainable_params"),
            "trainable_pct": metrics.get("trainable_pct"),
            "training_time_seconds": metrics.get("training_time_seconds"),
        }

        detailed_results[name] = {
            **results[name],
            "predictions_path": str(predictions_path) if predictions_path is not None else None,
            "diagnostics": diagnostics,
        }

    # Compute deltas
    if "baseline" in results and "lora" in results:
        baseline_trainable = results["baseline"].get("trainable_params")
        lora_trainable = results["lora"].get("trainable_params")
        baseline_time = numeric_or_none(results["baseline"].get("training_time_seconds"))
        lora_time = numeric_or_none(results["lora"].get("training_time_seconds"))
        results["delta_lora_vs_baseline"] = {
            "accuracy": results["lora"]["accuracy"] - results["baseline"]["accuracy"],
            "f1_macro": results["lora"]["f1_macro"] - results["baseline"]["f1_macro"],
            "f1_weighted": results["lora"]["f1_weighted"] - results["baseline"]["f1_weighted"],
        }
        if baseline_time is not None and lora_time is not None:
            results["delta_lora_vs_baseline"]["training_time_diff"] = lora_time - baseline_time
        if baseline_trainable and lora_trainable is not None:
            results["delta_lora_vs_baseline"]["param_reduction_pct"] = round(
                (1 - lora_trainable / max(baseline_trainable, 1)) * 100,
                2,
            )

    if "baseline" in results and "retrained" in results:
        results["delta_retrained_vs_baseline"] = {
            "accuracy": results["retrained"]["accuracy"] - results["baseline"]["accuracy"],
            "f1_macro": results["retrained"]["f1_macro"] - results["baseline"]["f1_macro"],
            "f1_weighted": results["retrained"]["f1_weighted"] - results["baseline"]["f1_weighted"],
        }

    for key in ["delta_lora_vs_baseline", "delta_retrained_vs_baseline"]:
        if key in results:
            detailed_results[key] = results[key]

    # Noise stats
    noise_summary = load_json_from_candidates([
        DATA_PROCESSED / "noise" / "noise_summary.json",
    ])
    if noise_summary is not None:
        results["noise_detection"] = noise_summary
        detailed_results["noise_detection"] = noise_summary

    # Uncertainty stats
    mc_summary = load_json_from_candidates([
        DATA_PROCESSED / "uncertainty" / "mc_summary.json",
    ])
    if mc_summary is not None:
        results["uncertainty"] = mc_summary
        detailed_results["uncertainty"] = mc_summary

    # Save
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_dir / "evaluation_detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    epoch_df = collect_epoch_results()
    epoch_wide_df = build_epoch_wide_table(epoch_df)
    if not epoch_df.empty:
        epoch_df.to_csv(output_dir / "epoch_comparison_summary.csv", index=False, encoding="utf-8")
        epoch_df.to_json(
            output_dir / "epoch_comparison_summary.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )
        epoch_wide_df.to_csv(output_dir / "epoch_comparison_wide.csv", index=False, encoding="utf-8")

    print("[EVAL] Evaluation Summary:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    for name in ["baseline", "lora", "retrained"]:
        if name not in detailed_results:
            continue
        diagnostics = detailed_results[name].get("diagnostics")
        if not diagnostics:
            continue
        print(f"\n[{name.upper()}] Per-aspect summary:")
        for aspect, aspect_metrics in diagnostics.get("per_aspect", {}).items():
            print(
                f"  {aspect:<8} n={aspect_metrics['n_samples']:<5} "
                f"acc={aspect_metrics['accuracy']:.4f} "
                f"f1_macro={aspect_metrics['f1_macro']:.4f}"
            )
        if diagnostics.get("ece_available"):
            print(f"  ECE: {diagnostics['ece']:.6f}")
        else:
            print("  ECE: N/A (probability columns not available in prediction file)")

    # Print comparison table
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print(f"{'Experiment':<15} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weight':>10} {'Time (s)':>10} {'Params':>15}")
        print("-" * 70)
        for name in ["baseline", "lora", "retrained"]:
            if name in results:
                r = results[name]
                time_display = display_or_na(r.get("training_time_seconds"), 2)
                params_display = display_or_na(r.get("trainable_params"))
                print(
                    f"{name:<15} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
                    f"{r['f1_weighted']:>10.4f} {time_display:>10} "
                    f"{params_display:>15}"
                )
        print("=" * 70)

    print_epoch_summary(epoch_df)


if __name__ == "__main__":
    main()
