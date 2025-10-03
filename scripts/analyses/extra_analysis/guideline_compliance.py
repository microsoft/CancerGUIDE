import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def compute_confusion_metrics(entry):
    """Compute TP, FP, FN, TN and derived metrics from a result entry."""
    TP = entry["guideline_compliance_correct"]
    FN = entry["guideline_compliance_false_negative"]
    FP = entry["guideline_compliance_false_positive"]
    TN = entry["total_patients_matched"] - (TP + FN + FP)  # corrected formula

    # Derived metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def plot_confusion_matrix(metrics, model_name, benchmark_name, save_dir):
    """Plot and save a confusion matrix heatmap."""
    cm = np.array([[metrics["TP"], metrics["FP"]],
                   [metrics["FN"], metrics["TN"]]])

    labels = np.array([["TP", "FP"], ["FN", "TN"]])
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=labels + "\n" + cm.astype(str),
                fmt="", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix\n{model_name} on {benchmark_name}")

    out_path = Path(save_dir) / f"{model_name}_{benchmark_name}_confusion.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_bar_chart(all_metrics, save_dir):
    """Plot a bar chart comparing accuracy, precision, recall, and F1 across models."""
    metrics_names = ["accuracy", "precision", "recall", "f1"]
    models = [m["model"] for m in all_metrics]
    data = {name: [m[name] for m in all_metrics] for name in metrics_names}

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8,5))
    for i, metric in enumerate(metrics_names):
        ax.bar(x + i*width, data[metric], width, label=metric.capitalize())

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(models)
    ax.set_ylim(0,1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()

    out_path = Path(save_dir) / "model_metrics_bar.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main(json_file, save_dir="plots"):
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    with open(json_file, "r") as f:
        entries = json.load(f)

    all_metrics = []
    for entry in entries:
        metrics = compute_confusion_metrics(entry)
        metrics["model"] = entry["model"]
        metrics["benchmark"] = entry["benchmark"]
        all_metrics.append(metrics)

        # Plot confusion matrix for each model + benchmark
        plot_confusion_matrix(metrics, entry["model"], entry["benchmark"], save_dir)

    # Plot bar chart across models
    plot_bar_chart(all_metrics, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate confusion matrix + bar plots from JSON results.")
    parser.add_argument("--json_file", type=str, help="Path to JSON file with results.")
    parser.add_argument("--save_dir", type=str, default="plots", help="Directory to save plots.")
    args = parser.parse_args()

    main(args.json_file, args.save_dir)
