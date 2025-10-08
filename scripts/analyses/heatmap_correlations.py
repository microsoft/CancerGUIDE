import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import argparse
import os
import math

# === Argument parser ===
parser = argparse.ArgumentParser(description="Evaluate model predictions against benchmark data.")
parser.add_argument("--result_json", required=True, help="Path to the results JSON file")
parser.add_argument("--output_dir", default="./results/figures/", help="Directory to save output figures")
parser.add_argument("--one_plot", action="store_true", help="If set, saves one PNG with all heatmaps stacked")
args = parser.parse_args()

# === Load results ===
with open(args.result_json, "r") as f:
    results = json.load(f)

# === Config: benchmark order & labels ===
BENCHMARKS = [
    ("human_new_prompt", "Human"),  # reference
("unstructured", "Synthetic\nUnstructured"),
("structured", "Synthetic\nStructured"),
("path_filter", "Self-Consistency\n(Path Overlap Threshold)"),
("treatment_filter", "Self-Consistency\n(Treatment Match Threshold)"),
("path_aggregation", "Cross-Model Consistency\n(Path Overlap Threshold)"),
("treatment_aggregation", "Cross-Model Consistency\n(Treatment Match Threshold)")
]
METRICS = ["Path Overlap", "Treatment Match"]
TOTAL_PATIENTS = 121

# === Helper functions ===
def mean_squared_error(x, y):
    x, y = np.asarray(x), np.asarray(y)
    print(f"RMSE calculation with {len(x)} points.")
    print(f"x: {x}")
    print(f"y: {y}")
    return math.sqrt(np.mean((x - y) ** 2)), 0  # dummy second value for compatibility

# === Methods to compute ===
METHODS = [
    # (pearsonr, "Pearson", "pearson"),
    (mean_squared_error, "RMSE", "rmse"),
    (spearmanr, "Spearman Correlation", "spearman")
]

def prepare_comparison_values(human, comp, key, model):
    """Adjust filtering benchmarks and return human and comp values"""
    # TOTAL_PATIENTS = 121
    # if model == "gpt-5-high":
    #     TOTAL_PATIENTS = 111
    
    # Make copies
    human_val = human.copy()
    comp_val = comp.copy()

    if key in ["path_filter", "treatment_filter"]:
        count = comp_val["total_patients_matched"]
        comp_val["average_path_match"] *= count / TOTAL_PATIENTS
        comp_val["average_treatment_match"] *= count / TOTAL_PATIENTS
    
    return (
        (human_val["average_path_match"], comp_val["average_path_match"]),
        (human_val["average_treatment_match"], comp_val["average_treatment_match"])
    )

# === Organize results by model and benchmark ===
by_model = {}
for entry in results:
    model, benchmark = entry["model"], entry.get("benchmark")
    if "gpt-5_" in benchmark:
        benchmark = benchmark.replace("gpt-5_", "")
    by_model.setdefault(model, {})[benchmark] = entry

# === Create output directory ===
os.makedirs(args.output_dir, exist_ok=True)

# === Generate heatmaps ===
heatmaps = []

for method_func, method_name, method_filename in METHODS:
    print(f"Generating heatmap for {method_name}...")

    method_labels = [label for _, label in BENCHMARKS[1:]]  # skip human
    matrix = np.zeros((len(method_labels), len(METRICS)))

    for i, (bench_key, _) in enumerate(BENCHMARKS[1:]):
        path_vals, treat_vals = [], []
        for model, bench_dict in by_model.items():
            # if model == "llama":
            #     continue
            if "human_new_prompt" not in bench_dict or bench_key not in bench_dict:
                continue
            human, comp = bench_dict["human_new_prompt"], bench_dict[bench_key]
            path_pair, treat_pair = prepare_comparison_values(human, comp, bench_key, model)
            path_vals.append(path_pair)
            treat_vals.append(treat_pair)
        print(f"Benchmark: {bench_key}, Path pairs: {len(path_vals)}, Treat pairs: {len(treat_vals)}")
        if path_vals:
            human_vals, comp_vals = zip(*path_vals)
            matrix[i, 0] = method_func(human_vals, comp_vals)[0]
        if treat_vals:
            human_vals, comp_vals = zip(*treat_vals)
            matrix[i, 1] = method_func(human_vals, comp_vals)[0]

    cmap = "Reds_r" if method_filename == "rmse" else "Blues"
    heatmaps.append((matrix, method_name, method_filename, method_labels, cmap))

    # Save individual heatmap
    if not args.one_plot:
        plt.figure(figsize=(8, 5))
        sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=METRICS, yticklabels=method_labels,
                    cmap=cmap, cbar_kws={'label': 'RMSE' if method_filename == "rmse" else 'Correlation'})
        plt.title(f"{method_name} of Proxy Benchmarks vs Ground Truth")
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"heatmap-{method_filename}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

# === Combined plot ===
if args.one_plot:
    fig, axes = plt.subplots(1, len(heatmaps), figsize=(6 * len(heatmaps), 6))
    if len(heatmaps) == 1: axes = [axes]

    for ax, (matrix, method_name, _, method_labels, cmap) in zip(axes, heatmaps):
        sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=METRICS, yticklabels=method_labels,
                    cmap=cmap, ax=ax, annot_kws={"size": 14}, cbar_kws={'label': 'RMSE' if ax == axes[0] else 'Correlation'})
        ax.set_title(f"Proxy Benchmarks vs Ground Truth:\n{method_name}", fontsize=16)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    combined_path = os.path.join(args.output_dir, "heatmaps_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmap: {combined_path}")

print("All heatmaps generated successfully!")
