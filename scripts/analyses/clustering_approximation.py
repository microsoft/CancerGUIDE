import json
import json5
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from itertools import combinations
from sklearn.decomposition import PCA
# -------------------------------
# Utility functions
# -------------------------------
def load_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    after_think = None
    if "</think>" in text:
        after_think = text.split("</think>")[-1]
    search_region = after_think if after_think is not None else text
    match = re.search(r"```(?:json)?\s*({.*?})\s*```", search_region, re.DOTALL)
    if not match:
        match = re.search(r"({.*?})", search_region, re.DOTALL)
    if not match:
        return None
    try:
        return json5.loads(match.group(1))
    except Exception:
        return None

def normalize_path_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [s.strip() for s in x]
    return [s.strip() for s in re.split(r"\s*->\s*|\s*>\s*|\s*→\s*", str(x)) if s.strip()]


def filter_path_17_node(lst):
    if not lst:
        return lst
    out = list(lst)
    if out and out[-1] == "NSCL-17-10" and len(out) > 1:
        out = out[:-1]
    if out and out[-1] == "NSCL-17-1" and len(out) > 1:
        out = out[:-1]
    return out

def compare_lists(paths_filtered):
    n = len(paths_filtered)
    if n < 2:
        return 1.0, 1.0, None, None
    path_frac = np.mean([paths_filtered[i] == paths_filtered[j] 
                         for i in range(n) for j in range(i+1, n)])
    treatment_frac = path_frac
    return path_frac, treatment_frac, None, None

# -------------------------------
# Clustering Analysis
# -------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

def clustering_entropy(labels):
    """Compute normalized entropy of cluster assignments"""
    counts = np.bincount(labels)
    probs = counts / counts.sum()
    return entropy(probs, base=2) / np.log2(len(probs))  # normalized [0,1]

def cluster_accuracy_and_f1(y_true, y_pred):
    """
    Map cluster labels to true labels using Hungarian algorithm.
    Returns accuracy and F1 (macro).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        cost_matrix[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    y_pred_mapped = np.vectorize(mapping.get)(y_pred)
    
    acc = accuracy_score(y_true, y_pred_mapped)
    f1 = f1_score(y_true, y_pred_mapped, average='macro')
    return acc, f1
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
def cluster_accuracy_and_f1(y_true, y_pred):
    """Best-match cluster labels to true labels, then compute accuracy and F1 (macro)."""
    from sklearn.metrics import accuracy_score, f1_score

    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))
    y_pred_aligned = np.array([mapping[label] for label in y_pred])

    acc = accuracy_score(y_true, y_pred_aligned)
    f1 = f1_score(y_true, y_pred_aligned, average="macro")
    return acc, f1, y_pred_aligned

def run_clustering_analysis(X, y_treat=None, n_clusters=2):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    print(f"Cluster centroids:\n{kmeans.cluster_centers_}")

    metrics = {}
    cm = None
    aligned_labels = cluster_labels.copy()

    if y_treat is not None:
        ari = adjusted_rand_score(y_treat, cluster_labels)
        nmi = normalized_mutual_info_score(y_treat, cluster_labels)
        acc, f1, aligned_labels = cluster_accuracy_and_f1(y_treat, cluster_labels)
        cm = confusion_matrix(y_treat, aligned_labels)

        metrics = {
            "ARI": ari,
            "NMI": nmi,
            "Accuracy": acc,
            "F1 (macro)": f1
        }
        print(f"Adjusted Rand Index: {ari:.3f}, Normalized Mutual Info: {nmi:.3f}, Accuracy: {acc:.3f}, F1 (macro): {f1:.3f}")

    # PCA projection to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # --- Plot 1: Subplots of clusters colored by true vs predicted ---
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    
    if y_treat is not None:
        sns.scatterplot(ax=axes[0], x=X_pca[:,0], y=X_pca[:,1], hue=y_treat, palette="Set2", s=80)
        axes[0].set_title("Clusters Colored by True Labels")
    else:
        sns.scatterplot(ax=axes[0], x=X_pca[:,0], y=X_pca[:,1], hue=cluster_labels, palette="Set2", s=80)
        axes[0].set_title("Clusters Colored by Predicted Labels")
    
    sns.scatterplot(ax=axes[1], x=X_pca[:,0], y=X_pca[:,1], hue=cluster_labels, palette="Set2", s=80)
    axes[1].set_title("Clusters Colored by Predicted Labels")
    
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend_.remove()  # remove legend to avoid clutter

    plt.tight_layout()
    plt.show()
    plt.savefig("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/results/0827/figures/clustering_analysis.png")

    # --- Plot 2: Confusion Matrix ---
    if cm is not None:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (aligned clusters → true labels)")
        plt.xlabel("Predicted Cluster")
        plt.ylabel("True Label")
        plt.show()
        plt.savefig("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/results/0827/figures/clustering_confusion_matrix.png")

    return cluster_labels, cm, metrics


def feature_selection_unsupervised_stability(X, feature_names, max_features=4, n_clusters=2, n_repeats=10):
    """
    Feature selection by clustering stability and cluster balance.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    import numpy as np

    best_score = -np.inf
    best_subset = []

    n_features = X.shape[1]
    max_features = min(max_features, n_features)
    scaler = StandardScaler()

    for k in range(4, max_features + 1):
        for subset in combinations(range(n_features), k):
            X_sub = X[:, subset]
            X_scaled = scaler.fit_transform(X_sub)
            clusterings = []

            # Repeat KMeans multiple times
            entropies = []
            for _ in range(n_repeats):
                labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=None).fit_predict(X_scaled)
                clusterings.append(labels)
                entropies.append(clustering_entropy(labels))

            # Skip subsets that produce too unbalanced clusters
            if np.mean(entropies) < 0.5:
                continue

            # Compute average pairwise ARI
            ari_sum = 0
            count = 0
            for i in range(len(clusterings)):
                for j in range(i + 1, len(clusterings)):
                    ari_sum += adjusted_rand_score(clusterings[i], clusterings[j])
                    count += 1
            avg_ari = ari_sum / count

            if avg_ari > best_score:
                best_score = avg_ari
                best_subset = list(subset)
                print(f"New best subset: {[feature_names[i] for i in best_subset]} -> Stability ARI: {best_score:.3f}, Avg entropy: {np.mean(entropies):.3f}")
                if avg_ari ==1.0:
                    print("MAXimum ARI, breaking")
                    break
        if best_score == 1.0:
            break

    print(f"\nSelected feature subset (stability+balance): {[feature_names[i] for i in best_subset]}")
    print(f"Best stability ARI achieved: {best_score:.3f}")
    return best_subset, best_score

def feature_selection_max_ari(X, y_treat, feature_names, max_features=2, n_clusters=2):
    if max_features is None:
        max_features = X.shape[1]

    best_ari = -1.0
    best_subset = []

    n_features = X.shape[1]
    max_features = min(max_features, n_features)
    print("Unique rows:", np.unique(X, axis=0).shape)
    print("Min/Max per feature:", X.min(axis=0), X.max(axis=0))


    # Exhaustive search over all subset sizes
    for k in range(1, max_features + 1):
        for subset in combinations(range(n_features), k):
            X_sub = X[:, subset]
            X_scaled = StandardScaler().fit_transform(X_sub)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            cluster_labels = kmeans.fit_predict(X_scaled)
            ari = adjusted_rand_score(y_treat, cluster_labels)

            if ari > best_ari:
                best_ari = ari
                best_subset = list(subset)
                print(f"New best subset: {[feature_names[i] for i in best_subset]} -> ARI: {best_ari:.3f}")

    selected_features = best_subset
    print(f"\nSelected feature subset: {selected_features}")
    print(f"Max ARI achieved: {best_ari:.3f}")

    return best_subset, best_ari

import json
from pathlib import Path
from typing import Dict

def load_trend_data() -> Dict[str, Dict[str, float]]:
    """
    Load benchmark results and extract both average_treatment_match 
    and average_path_match metrics for each model/benchmark.

    Returns:
        Dict[str, Dict[str, float]]:
            model -> { "benchmark_metric": value, ... }
    """
    reformatted: Dict[str, Dict[str, float]] = {}
    data_path = Path("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/results/0906/heatmap_results_4.json")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        model = entry["model"]
        benchmark = entry["benchmark"]

        # Skip unwanted benchmarks
        if benchmark in {"human_new_prompt", "treatment_aggregation", "path_aggregation"}:
            continue

        if model not in reformatted:
            reformatted[model] = {}

        for metric in ["average_treatment_match", "average_path_match"]:
            value = entry.get(metric)
            if value is None:
                continue  # skip if metric not present

            # Apply penalty for filter benchmarks
            if benchmark in {"path_filter", "treatment_filter"}:
                answered = entry.get("total_patients_matched", 0)
                penalty = answered / 121 if answered else 0
                value *= penalty

            # Unique key: benchmark + metric
            if "gpt-5_" in benchmark:
                benchmark = benchmark.split("gpt-5_")[1]
            key = f"{benchmark}_{metric}"
            reformatted[model][key] = float(value)

    return reformatted


# -------------------------------
# Analyzer Class
# -------------------------------
class AccuracyAnalyzer:
    def __init__(self, annotations_path: Path, patient_id_pool: int = 360):
        self.annotations_path = Path(annotations_path)
        self.patient_id_pool = patient_id_pool
        self.ground_truth = self.load_ground_truth()

    def load_ground_truth(self) -> Dict[str, List[str]]:
        ground_truth = {}
        for patient_id in range(self.patient_id_pool + 1):
            file_path = self.annotations_path / f"patient_{patient_id}.json"
            if not file_path.exists():
                continue
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                if "label" in data:
                    ground_truth[str(patient_id)] = normalize_path_list(data["label"])
            except Exception as e:
                print(f"Error loading GT for patient {patient_id}: {e}")
        return ground_truth

    @staticmethod
    # def iterations_for_model(model_name: str) -> int:
    #     return 3 if model_name.lower() == "deepseek" else 10

    @staticmethod
    def results_dir_for_model(base_path: Path, model_name: str) -> Path:
        return base_path / "results" / "rollout_results_0815_benchmark" / f"rollout_experiment_{model_name}"

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
                paths.append(normalize_path_list(data["final_path"]))
        return paths

    def collect_rows_for_model(self, results_dir: Path, model_name: str, num_iterations: int,
                               all_model_paths: Dict[str, Dict[str,List[str]]], model_trend_features: List[dict]):
        if not results_dir.exists():
            print(f"[WARN] Missing results_dir for {model_name}: {results_dir}")
            return

        for patient_dir in results_dir.glob("patient_*"):
            pid = patient_dir.name.split("_")[1]
            if pid not in self.ground_truth:
                continue
            gt_path = filter_path_17_node(self.ground_truth[pid])

            paths = self.load_patient_paths(patient_dir, pid, num_iterations=num_iterations)
            if not paths:
                continue

            paths_filtered = [tuple(filter_path_17_node(p)) for p in paths if p]
            if not paths_filtered:
                continue
            counter = Counter(paths_filtered)
            mode_path, mode_count = counter.most_common(1)[0]
            mode_path = list(mode_path)

            path_match_fraction_prediction, treatment_match_fraction_prediction, _, _ = compare_lists(paths_filtered)
            mode_frac = mode_count / max(1, num_iterations)

            # Aggregated feature
            match_count = 0
            total_models = 0
            for other_model, patient_paths in all_model_paths.items():
                if pid in patient_paths:
                    other_mode_path = patient_paths[pid]
                    total_models += 1
                    if other_mode_path[-1] == mode_path[-1]:
                        match_count += 1
            aggregated_score = match_count / max(1, total_models) if total_models > 0 else 0.0

            feats = {
                "path_match_fraction": path_match_fraction_prediction,
                "treatment_match_fraction": treatment_match_fraction_prediction,
                "aggregated": aggregated_score,
                **{benchmark: value for benchmark, value in model_trend_features.get(model_name).items()}
            }

            treat_match_target = int(gt_path and (gt_path[-1] == mode_path[-1]))
            path_overlap_target = compare_lists([mode_path, gt_path])[0]
            yield feats, (path_match_fraction_prediction, treatment_match_fraction_prediction), pid, model_name, treat_match_target, path_overlap_target

    def build_dataset(self, base_path: Path, model_names: List[str], model_trend_features: dict[str, List[dict]]):
        # Build all_model_paths for aggregation
        all_model_paths = {}
        for mname in model_names:
            rdir = self.results_dir_for_model(base_path, mname)
            patient_paths = {}
            n_iter = 10
            for patient_dir in rdir.glob("patient_*"):
                pid = patient_dir.name.split("_")[1]
                paths = self.load_patient_paths(patient_dir, pid, num_iterations=n_iter)
                if not paths:
                    continue
                paths_filtered = [tuple(filter_path_17_node(p)) for p in paths if p]
                if not paths_filtered:
                    continue
                counter = Counter(paths_filtered)
                mode_path, _ = counter.most_common(1)[0]
                patient_paths[pid] = list(mode_path)
            all_model_paths[mname] = patient_paths

        # Collect dataset
        rows = []
        for model_name in model_names:
            n_iter = 10
            rdir = self.results_dir_for_model(base_path, model_name)
            for row in self.collect_rows_for_model(rdir, model_name, n_iter, all_model_paths, model_trend_features):
                rows.append(row)

        if not rows:
            print("No data gathered from the selected models.")
            return None

        ordered_models = list(dict.fromkeys([m for *_rest, m in [(None, None, None, r[3]) for r in rows]]))
        X_list, y_path, y_treat, groups, model_labels = [], [], [], [], []
        for feats, (ps, tm), pid, mname, treat_match, path_match in rows:
            vec = [val for key, val in feats.items()]
            X_list.append(vec)
            y_path.append(path_match)
            y_treat.append(treat_match)
            groups.append(pid)
            model_labels.append(mname)

        X = np.asarray(X_list, dtype=float)
        y_path = np.asarray(y_path, dtype=float)
        y_treat = np.asarray(y_treat, dtype=int)
        groups = np.asarray(groups)

        return X, y_path, y_treat, groups, ordered_models, model_labels

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    BASE_PATH = Path("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean")
    ANNOTATIONS_PATH = BASE_PATH / "data" / "benchmarks" / "human_annotations"
    model_names = ["gpt-5", "gpt-5-med", "gpt-4.1", "o4-mini", "o3", "deepseek", "llama", "gpt-5-high"]

    # model_names=model_names[5:6]  # for quick testing
    # Trend features
    model_trend_features= load_trend_data()
    # Load dataset
    analyzer = AccuracyAnalyzer(annotations_path=ANNOTATIONS_PATH)
    X, y_path, y_treat, groups, ordered_models, model_labels = analyzer.build_dataset(BASE_PATH, model_names, model_trend_features)
    BASE_FEATURES = ["path_match_fraction", "treatment_match_fraction", "aggregated"]
    feature_names = BASE_FEATURES + list(model_trend_features['gpt-4.1'].keys()) # + model_onehot_names

    def split_unique_validation_test(X, y=None, N_unique=100, random_state=42):
        """
        Split X into:
        - validation set with exactly N_unique unique rows
        - test set with the remaining rows

        If y is provided, it will be split accordingly.
        """
        rng = np.random.default_rng(random_state)

        # Find unique rows and their first occurrence indices
        X_unique, unique_indices = np.unique(X, axis=0, return_index=True)

        if N_unique > len(X_unique):
            raise ValueError(f"Requested {N_unique} unique points, but only {len(X_unique)} exist.")

        # Randomly choose N_unique unique rows for validation
        selected_unique_idx = rng.choice(unique_indices, size=N_unique, replace=False)
        mask = np.zeros(X.shape[0], dtype=bool)
        mask[selected_unique_idx] = True

        X_val = X[mask]
        X_test = X[~mask]

        if y is not None:
            y_val = y[mask]
            y_test = y[~mask]
            return X_val, X_test, y_val, y_test
        else:
            return X_val, X_test

    # Greedy feature selection to maximize ARI
    # X_val, X_test, y_val, y_test = split_unique_validation_test(X, y_treat, N_unique=350)
    # selected_idx, max_ari = feature_selection_max_ari(X_val, y_val, feature_names, max_features=2, n_clusters=2)
    # breakpoint()
    # selected_idx, best_score = feature_selection_unsupervised_stability(X, feature_names, max_features=len(feature_names), n_clusters=2)
    selected_idx=[0,1,2]  # path_match_fraction, treatment_match_fraction, aggregated, gpt-4.1_average_treatment_match
    X_selected = X[:, selected_idx]
    total = np.concatenate([X_selected, y_treat[:, None]], axis=1)
    run_clustering_analysis(X_selected, y_treat=y_treat, n_clusters=2)
    # for k in range(3, len(feature_names)-1):
    #     selected_idx.append(k)
    #     X_selected = X[:, selected_idx]
    #     run_clustering_analysis(X_selected, y_treat=y_treat, n_clusters=2)
    #     print(f"Clustering with features: {[feature_names[i] for i in selected_idx]}")
    #     selected_idx.pop()  # remove last added feature to try next one

