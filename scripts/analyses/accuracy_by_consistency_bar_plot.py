import json
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.stats import pearsonr
import json5

# -----------------------------
# JSON loader
# -----------------------------
def load_json(file_path: Path):
    """Extract and parse JSON from text."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    after_think = None
    if "</think>" in text:
        after_think = text.split("</think>")[-1]
    search_region = after_think if after_think is not None else text
    match = re.search(r"(?:json)?\s*({.*?})\s*", search_region, re.DOTALL)
    if not match:
        match = re.search(r"({.*?})", search_region, re.DOTALL)
    if not match:
        return None
    try:
        return json5.loads(match.group(1))
    except Exception:
        return None

# -----------------------------
# Helper to filter benign endings
# -----------------------------
def filter_path_17_node(lst):
    if lst:
        if lst[-1] == "NSCL-17-10" and len(lst) > 1:
            lst = lst[:-1]
    if lst:
        if lst[-1] == "NSCL-17-1" and len(lst) > 1:
            lst = lst[:-1]
    return lst

# -----------------------------
# Analyzer
# -----------------------------
class AccuracyAnalyzer:
    def __init__(self, results_dir: Path, annotations_path: Path, patient_id_pool: int = 360):
        self.results_dir = Path(results_dir)
        self.annotations_path = Path(annotations_path)
        self.patient_id_pool = patient_id_pool
        self.ground_truth = self.load_ground_truth()

    def load_ground_truth(self):
        ground_truth = {}
        for patient_id in range(self.patient_id_pool + 1):
            file_path = self.annotations_path / f"patient_{patient_id}.json"
            if not file_path.exists():
                continue
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                if "label" in data:
                    ground_truth[str(patient_id)] = data["label"]
            except Exception as e:
                print(f"Error loading GT for patient {patient_id}: {e}")
        return ground_truth

    def calculate_treatment_match(self, human_prediction_path_list, model_prediction_list):
        filtered_human = filter_path_17_node(human_prediction_path_list)
        filtered_model = filter_path_17_node(model_prediction_list)
        return filtered_human[-1] == filtered_model[-1]

    def compare_lists(self,lists):
        # Clean up special terminal cases
        cleaned_lists = []
        for lst in lists:
            lst_filtered = filter_path_17_node(lst)
            cleaned_lists.append(lst_filtered)

        # --- Path match fraction using Jaccard similarity ---
        sets = [set(lst) for lst in cleaned_lists if lst]
        if sets:
            intersection = set.intersection(*sets) if len(sets) > 1 else sets[0]
            union = set.union(*sets)
            path_match_fraction = len(intersection) / len(union) if union else 1.0
        else:
            print("Error in comparing paths, no paths found")
            return None

        final_elements = [lst[-1] for lst in cleaned_lists if lst]
        if not final_elements:
            treatment_match_fraction = 1.0
            treatment_mode = None
        elif len(final_elements) == 1:
            treatment_match_fraction = 1.0
            treatment_mode = final_elements[0]
        else:
            final_counter = Counter(final_elements)
            treatment_mode, most_common_count = final_counter.most_common(1)[0]
            treatment_match_fraction = most_common_count / len(final_elements)

        list_tuples = [tuple(lst) for lst in cleaned_lists]
        list_counter = Counter(list_tuples)
        path_mode = list(list_counter.most_common(1)[0][0]) if list_counter else None
        return path_match_fraction, int(treatment_match_fraction)

    def load_matched_outputs(self, patient_dir: Path, pid: str, num_iterations: int):
        fpath = patient_dir / f"matched_outputs_{pid}_k_{num_iterations}.json"
        if not fpath.exists():
            return None
        with open(fpath, "r") as f:
            return json.load(f)

    def load_patient_paths(self, patient_dir: Path, pid: str, num_iterations: int = 10):
        paths = []
        for i in range(num_iterations):
            fpath = patient_dir / f"patient_{pid}_iteration_{i}.json"
            if not fpath.exists():
                continue
            data = load_json(fpath)
            if not data:
                continue
            if "final_path" in data:
                paths.append(data["final_path"])
        return paths

    def collect_data(self, num_iterations: int = 10):
        xs1, ys1, xs2, ys2, xs3, ys3, xs4, ys4 = [], [], [], [], [], [], [], []
        for patient_dir in self.results_dir.glob("patient_*"):
            pid = patient_dir.name.split("_")[1]
            if pid not in self.ground_truth:
                continue
            gt_path = self.ground_truth[pid]
            ground_truth = filter_path_17_node(gt_path)

            matched_data = self.load_matched_outputs(patient_dir, pid, num_iterations)
            if matched_data:
                model_prediction = filter_path_17_node(matched_data["final_path_mode"])
                xs1.append(matched_data["final_path_match"])
                path_score, _ = self.compare_lists([model_prediction, ground_truth])
                ys1.append(path_score)
                xs2.append(matched_data["final_treatment_score"])
                _, treat_match = self.compare_lists([matched_data["final_treatment_mode"], ground_truth])
                ys2.append(treat_match)

            paths = self.load_patient_paths(patient_dir, pid, num_iterations=num_iterations)
            if paths:
                paths = [filter_path_17_node(p) for p in paths]
                counter = Counter(paths)
                mode_path, mode_count = counter.most_common(1)[0]
                path_score, treat_match = self.compare_lists([mode_path, ground_truth])
                xs3.append(mode_count/len(paths))
                ys3.append(path_score)
                xs4.append(mode_count/len(paths))
                ys4.append(treat_match)

        return xs1, ys1, xs2, ys2, xs3, ys3, xs4, ys4

# -----------------------------
# Grouped Bar Plotting (with SEM)
label_map = {"o3": "o3", "gpt-5": "GPT-5-Minimal", "gpt-5-med": "GPT-5-Medium", "gpt-4.1": "GPT-4.1", "o4-mini": "o4-mini", "deepseek": "DeepSeek-R1", "llama": "LLaMA-3.3-70B-Instr.", "gpt-5-high": "GPT-5-High"}
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def grouped_bar_plot(all_data, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(12, 6))  # wider plot
    models = list(all_data.keys())
    
    # define bins
    all_x = np.concatenate([all_data[m]["x"] for m in models if len(all_data[m]["x"]) > 0])
    bins = np.linspace(max(all_x)/5, max(all_x), num=5)
    width = 0.8 / len(models)
    colors = plt.cm.tab10.colors

    # CSV data collection
    csv_rows = []

    for i, model in enumerate(models):
        x_vals, y_vals = all_data[model]["x"], all_data[model]["y"]
        means, sems = [], []
        # breakpoint()
        for b in bins:
            # find all y-values in this bin
            vals = [y for (x, y) in zip(x_vals, y_vals) 
                    if b == bins[np.argmin(np.abs(np.array(bins) - x))]]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                means.append(np.nan)
                sems.append(np.nan)

        # Only use bins that are not empty
        valid_bins = [b for b, m in zip(bins, means) if not np.isnan(m)]
        valid_means = [m for m in means if not np.isnan(m)]

        if len(valid_bins) > 1:
            r, _ = pearsonr(valid_bins, valid_means)
        else:
            r = np.nan

        # Save CSV row for smallest and largest bin
        model_label = label_map.get(model, model)
        smallest_bin_val = means[0]
        largest_bin_val = means[-1]
        # safe multiplier calculation
        multiplier = largest_bin_val / smallest_bin_val if smallest_bin_val != 0 else np.nan

        csv_rows.append({
            "model": model_label,
            "score_smallest_bin": smallest_bin_val,
            "score_largest_bin": largest_bin_val,
            "multiplier": multiplier,
            "r_score": r
        })

        # Plot bars
        plt.bar([b + i * width for b in range(len(bins))],
                means, width=width, label=model_label, color=colors[i % len(colors)],
                yerr=sems, capsize=4, alpha=0.9)

    plt.xticks([r + width*(len(models)/2) for r in range(len(bins))], 
            [round(b, 2) for b in bins], fontsize=14)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=22)

    # Move legend outside the figure on the right
    # plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    plt.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')  # Ensure legend is included
    plt.close()


    # Save CSV
    df = pd.DataFrame(csv_rows)
    csv_outpath = outpath.with_suffix('.csv')
    df.to_csv(csv_outpath, index=False)
    print(f"Results CSV saved to {csv_outpath}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze accuracy by consistency across models.")
    parser.add_argument("--base_path", type=Path, required=True, help="Base path for results and data.")
    parser.add_argument("--outdir", type=Path, required=True, help="Path to output directory.")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    BASE_PATH = args.base_path
    ANNOTATIONS_PATH = BASE_PATH / "data" / "benchmarks" / "human_annotations"

    MODELS = ["gpt-4.1", "gpt-5", "o3", "o4-mini", "gpt-5-med", "deepseek", "llama", "gpt-5-high"]

    all_plot_data = {1: {}, 2: {}, 3: {}, 4: {}}
    for model in MODELS:
        RESULTS_PATH = BASE_PATH / "results" / "rollout_results" / f"rollout_experiment_{model}"
        analyzer = AccuracyAnalyzer(RESULTS_PATH, ANNOTATIONS_PATH, patient_id_pool=360)
        num_iterations=10
        xs1, ys1, xs2, ys2, xs3, ys3, xs4, ys4 = analyzer.collect_data(
            num_iterations=num_iterations
        )
        all_plot_data[1][model] = {"x": xs1, "y": ys1}
        all_plot_data[2][model] = {"x": xs2, "y": ys2}
        all_plot_data[3][model] = {"x": xs3, "y": ys3}
        all_plot_data[4][model] = {"x": xs4, "y": ys4}

    grouped_bar_plot(all_plot_data[1], "Path Overlap Score (model)", "Path Overlap Score (vs GT)",
                     "Path Consistency vs Accuracy", args.outdir / "plot1_grouped.png")
    grouped_bar_plot(all_plot_data[2], "Treatment Consistency Score (model)", "Treatment Match (vs GT)",
                     "Treatment Consistency vs Accuracy", args.outdir / "plot2_grouped.png")
    grouped_bar_plot(all_plot_data[3], "Fraction of Iterations Matching: Total Path", "Path Overlap Score (vs GT)",
                     "Iteration Consistency vs Path Overlap", args.outdir / "plot3_grouped.png")
    grouped_bar_plot(all_plot_data[4], "Fraction of Iterations Matching: Total Path", "Treatment Match (vs GT)",
                     " Iteration Consistency vs Treatment Prediction", args.outdir / "plot4_grouped.png")
