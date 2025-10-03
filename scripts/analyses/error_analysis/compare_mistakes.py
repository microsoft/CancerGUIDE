#!/usr/bin/env python3
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from itertools import combinations
import json5
import re
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

# ------------------------------------------------
# HELPERS
# ------------------------------------------------
def extract_json_from_text(text: str):
    """Extract and parse the first valid JSON object from raw text."""
    match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
    json_str = match.group(1) if match else None

    if not json_str:
        match = re.search(r"({.*?})", text, re.DOTALL)
        json_str = match.group(1) if match else None

    if not json_str:
        return None

    try:
        parsed = json5.loads(json_str)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None

def collect_mistakes(data_dir: Path):
    """Parse evaluation JSON files in a directory and return mistake structures."""
    edges = []
    pair_counter = Counter()

    for file in data_dir.glob("*_evaluation.json"):
        with open(file, "r") as f:
            data_text = f.read()
            data = extract_json_from_text(data_text)

        if not data:
            continue

        node_mistake = data.get("node_mistake")
        if not node_mistake or "->" not in node_mistake:
            continue

        source, targets_str = node_mistake.split("->")
        source = source.strip()
        targets = targets_str.strip().strip("[]").split(",")
        targets = [t.strip() for t in targets if t.strip()]

        processed_targets = []
        for t in targets:
            if "NSCL" not in t:
                if "-" in source:
                    source_prefix = "-".join(source.split("-")[:-1])
                    full_target = f"{source}-{t}"
                else:
                    full_target = t
                processed_targets.append(full_target)
            else:
                processed_targets.append(t)

        for t in processed_targets:
            edges.append((source, t))

        if len(processed_targets) >= 2:
            for combo in combinations(sorted(processed_targets), 2):
                pair_counter[combo] += 1

    if edges:
        df_edges = pd.DataFrame(edges, columns=["source", "target"])
        freq_matrix = pd.crosstab(df_edges["source"], df_edges["target"])
    else:
        freq_matrix = pd.DataFrame()

    return {
        "edges": edges,
        "pair_counter": pair_counter,
        "freq_matrix": freq_matrix,
    }

def prepare_top_pairs(pair_counter, gamma, top_n=15):
    filtered = [(p1, p2, c) for (p1, p2), c in pair_counter.items() if c > gamma]
    df = pd.DataFrame(filtered, columns=["t1", "t2", "count"]).sort_values("count", ascending=False)
    if not df.empty:
        df["pair"] = df["t1"] + " + " + df["t2"]
    return df.head(top_n)

def compare_pair_frequencies(model_counter, human_counter):
    all_pairs = set(model_counter.keys()) | set(human_counter.keys())
    model_freqs = [model_counter.get(p, 0) for p in all_pairs]
    human_freqs = [human_counter.get(p, 0) for p in all_pairs]

    if len(all_pairs) == 0:
        return {"pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None,
                "r2": None}

    pearson = pearsonr(model_freqs, human_freqs)
    spearman = spearmanr(model_freqs, human_freqs)
    r2 = r2_score(human_freqs, model_freqs)

    return {
        "pearson_r": pearson[0],
        "pearson_p": pearson[1],
        "spearman_r": spearman.correlation,
        "spearman_p": spearman.pvalue,
        "r2": r2
    }

def get_pair_df(pair_counter, label, gamma):
    filtered = [(p1, p2, c) for (p1, p2), c in pair_counter.items() if c > gamma]
    df = pd.DataFrame(filtered, columns=["t1", "t2", "count"])
    if not df.empty:
        df["pair"] = df["t1"] + " + " + df["t2"]
        df["who"] = label
    return df


def filter_pairs(counter):
    """Remove keys where one element contains 'end' and the other 'NSCL-17-1'."""
    return {
        k: v
        for k, v in counter.items()
        if not (
            isinstance(k, tuple)
            and len(k) == 2
            and (
                ("end" in k[0] and "NSCL-17-1" in k[1])
                or ("end" in k[1] and "NSCL-17-1" in k[0])
            )
        )
    }
# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_mistakes = collect_mistakes(Path(args.pred_dir))
    human_mistakes = collect_mistakes(Path(args.human_dir))

    print("✅ Model mistakes:", len(model_mistakes["edges"]), "edges")
    print("✅ Human mistakes:", len(human_mistakes["edges"]), "edges")

    # ------------------------------------------------
    # GROUPED BAR CHART
    # ------------------------------------------------
    top_k = getattr(args, "top_k", 5)  # default to 5
    model_mistakes["pair_counter"] = filter_pairs(model_mistakes["pair_counter"])
    human_mistakes["pair_counter"] = filter_pairs(human_mistakes["pair_counter"])

    model_df = get_pair_df(model_mistakes["pair_counter"], "Model", args.gamma)
    human_df = get_pair_df(human_mistakes["pair_counter"], "Human", args.gamma)

    if not model_df.empty or not human_df.empty:
        # Get top K from each
        top_model_pairs = model_df.sort_values("count", ascending=False).head(top_k)["pair"].tolist()
        model_df = model_df[model_df["pair"].isin(top_model_pairs)].sort_values("count", ascending=False)
        top_human_pairs = human_df.sort_values("count", ascending=False).head(top_k)["pair"].tolist()
        human_df = human_df[human_df["pair"].isin(top_human_pairs)].sort_values("count", ascending=False)

        # ✅ Only keep pairs that were actually in one of the top-k lists
        top_pairs = list(dict.fromkeys(top_model_pairs + top_human_pairs))  # preserves order, dedup

        # Build pivot manually
        total_model_errors = sum(model_mistakes["pair_counter"].values())
        total_human_errors = sum(human_mistakes["pair_counter"].values())

        data = []
        for pair in top_pairs:
            model_count = model_df.loc[model_df["pair"] == pair, "count"].sum()
            human_count = human_df.loc[human_df["pair"] == pair, "count"].sum()
            data.append({
                "pair": pair,
                "Model_prop": model_count / total_model_errors if total_model_errors > 0 else 0,
                "Human_prop": human_count / total_human_errors if total_human_errors > 0 else 0
            })
        pivot = pd.DataFrame(data)
        pivot = pivot.sort_values("Human_prop", ascending=False).reset_index(drop=True)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        x = np.arange(len(pivot))

        bars1 = ax.bar(x - bar_width/2, pivot["Model_prop"], bar_width, label="Model", alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, pivot["Human_prop"], bar_width, label="Human", alpha=0.8)

        # Annotate
        for bars, col in zip([bars1, bars2], ["Model_prop", "Human_prop"]):
            for bar, val in zip(bars, pivot[col]):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., val + 0.005,
                            f"{val:.1%}", ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot["pair"], rotation=45, ha="right")

        label_map = {
            "o3": "o3", "gpt-5": "GPT-5-Minimal", "gpt-5-med": "GPT-5-Medium",
            "gpt-4.1": "GPT-4.1", "o4-mini": "o4-mini", "deepseek": "DeepSeek-R1",
            "llama": "LLaMA-3.3-70B-Instr.", "gpt-5-high": "GPT-5-High"
        }
        model_title = label_map.get(args.model, args.model)

        ax.set_ylabel("Proportion of All Errors")
        ax.set_title(f"Top {top_k} {model_title} & Human Co-occurrence Mistakes")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f"{args.model}_ea.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model vs human mistakes")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--pred_dir", type=str, required=True, help="Prediction JSON directory")
    parser.add_argument("--human_dir", type=str, required=True, help="Human JSON directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gamma", type=int, default=4, help="Gamma frequency threshold")
    args = parser.parse_args()
    main(args)
