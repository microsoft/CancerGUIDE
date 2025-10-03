# Standard library imports
import json
import json5
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional
import argparse
import itertools

# Numerical and plotting libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: preprocessing, models, metrics, and model selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
    f1_score
)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# Statistics
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

def get_model_configs():
    """Return regression and classification model configurations"""
    regressors = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', random_state=42, n_jobs=-1
        )}
    classifiers= {
    # Linear / regularized
    "Logistic Regression (L2, balanced)": LogisticRegression(
        penalty="l2", C=0.1, solver="lbfgs", max_iter=10000, class_weight="balanced"
    )}
    return regressors, classifiers

# --------------------------
# Utilities / parsing helpers
# --------------------------

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


def compare_lists(lists, return_final_consistency=True):
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

    if return_final_consistency:
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

        return path_match_fraction, treatment_match_fraction, path_mode, treatment_mode

    return path_match_fraction

def load_trend_data(heatmap_results:str) -> Dict[str, Dict[str, float]]:
    """
    Load benchmark results and extract both average_treatment_match 
    and average_path_match metrics for each model/benchmark.

    Returns:
        Dict[str, Dict[str, float]]:
            model -> { "benchmark_metric": value, ... }
    """
    reformatted: Dict[str, Dict[str, float]] = {}
    data_path = Path(heatmap_results)

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
            key = f"{benchmark}_{metric}"
            reformatted[model][key] = float(value)
    return reformatted


def print_coeffs(model, name):
    # after model.fit(X, y)
    model_trend_features = load_trend_data("average_treatment_match").get("gpt-4.1")
    if not model_trend_features:
        return
    if hasattr(model, "named_steps"):  # pipeline
        if "reg" in model.named_steps:
            est = model.named_steps["reg"]
        elif "clf" in model.named_steps:
            est = model.named_steps["clf"]
        else:
            est = None
    else:
        est = model

    if est is not None and hasattr(est, "coef_"):
        coef = est.coef_
        intercept = getattr(est, "intercept_", None)

        # Build feature names
        base_feats = [
            "path_match_fraction",
            "treatment_match_fraction",
            "mode_frac",
            "aggregated",
        ]
        feature_names = base_feats + list(model_trend_features.keys())

        if coef.ndim > 1:  # multiclass or multioutput
            for class_idx, row in enumerate(coef):
                print(f"\n--- Coefficients for {name}, class {class_idx} ---")
                for fname, c in zip(feature_names, row):
                    print(f"{fname:25s} {float(c):.4f}")
            if intercept is not None:
                print(f"Intercepts: {intercept}")
        else:
            print(f"\n--- Coefficients for {name} ---")
            for fname, c in zip(feature_names, coef.ravel()):
                print(f"{fname:25s} {float(c):.4f}")
            if intercept is not None:
                print(f"Intercept: {float(intercept):.4f}")


class AccuracyAnalyzer:
    def __init__(self, annotations_path: Path, heatmap_results:str, patient_id_pool: int = 360):
        self.annotations_path = Path(annotations_path)
        self.patient_id_pool = patient_id_pool
        self.ground_truth = self.load_ground_truth()
        self.model_trend_features = load_trend_data(heatmap_results)

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
    def iterations_for_model(model_name: str) -> int:
        return 10

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

    def collect_rows_for_model(self, results_dir: Path, model_name: str, num_iterations: int, all_model_paths: Dict[str, List[List[str]]] = None):
        """
        all_model_paths: Dict mapping model_name -> list of mode_path per patient_id
        """
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
            path_score_target, treatment_match_target, _, _ = compare_lists([mode_path, gt_path])

            mode_frac = mode_count / max(1, num_iterations)

            # ---- NEW AGGREGATED FEATURE ----
            aggregated_score = None
            if all_model_paths is not None:
                match_count = 0
                total_models = 0
                for other_model, patient_paths in all_model_paths.items():
                    if pid in patient_paths:
                        other_mode_path = patient_paths[pid]
                        total_models += 1
                        if other_mode_path[-1] == mode_path[-1]:
                            match_count += 1
                aggregated_score = match_count / max(1, total_models) if total_models > 0 else 0.0
            # --------------------------------
            feats = {
                "path_match_fraction": path_match_fraction_prediction,
                "treatment_match_fraction": treatment_match_fraction_prediction,
                "mode_frac": mode_frac,
                "aggregated": aggregated_score,
                "num_iterations": num_iterations,
                "model_name": model_name,
                "pid": pid,
                **self.model_trend_features.get(model_name, {})
            }

            yield feats, (path_score_target, int(treatment_match_target)), pid, model_name

    def build_dataset(self, base_path: Path, all_model_names: List[str], target_model_names: List[str], feature_set: str):
        all_model_paths = {}
        for mname in all_model_names:
            rdir = self.results_dir_for_model(base_path, mname)
            patient_paths = {}
            n_iter = self.iterations_for_model(mname)
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
        
        rows = []
        for model_name in target_model_names:
            print("target model:", model_name)
            n_iter = self.iterations_for_model(model_name)
            rdir = self.results_dir_for_model(base_path, model_name)
            for row in self.collect_rows_for_model(rdir, model_name, n_iter, all_model_paths):
                rows.append(row)

        if not rows:
            print("No data gathered from the selected models.")
            return None

        ordered_models = list(dict.fromkeys([m for *_rest, m in [(None, None, None, r[3]) for r in rows]]))
        X_list, y_path, y_treat, groups, model_labels = [], [], [], [], []
        for feats, (ps, tm), pid, mname in rows:
            # Select features based on feature_set parameter
            if feature_set == "base":
                vec = [
                    feats["path_match_fraction"],
                    feats["treatment_match_fraction"],
                ]
            elif feature_set == "base_aggregated":
                vec = [
                    feats["path_match_fraction"],
                    feats["treatment_match_fraction"],
                    feats["aggregated"] if feats["aggregated"] is not None else 0.0,
                ]
            elif feature_set == "trend_only":
                vec = list(self.model_trend_features.get(mname, {}).values())
            elif feature_set == "aggregated_only":
                vec = [
                    feats["aggregated"] if feats["aggregated"] is not None else 0.0,
                ]
            elif feature_set=="internal":
                internal_extra_features = ["unstructured_average_treatment_match", "unstructured_average_path_match","structured_average_treatment_match",
                                            "structured_average_path_match", "path_filter_average_treatment_match", "path_filter_average_path_match", "treatment_filter_average_treatment_match", 
                                            "treatment_filter_average_path_match"]
                vec = [
                    feats["path_match_fraction"],
                    feats["treatment_match_fraction"], 
                ]
                
                # Ensure consistent ordering and handle missing features
                model_feats = self.model_trend_features.get(mname, {})
                for feature_name in internal_extra_features:
                    vec.append(model_feats.get(feature_name, 0.0))  # Use 0.0 as default
            elif feature_set == "all":
                base_feats = [
                    feats["path_match_fraction"],
                    feats["treatment_match_fraction"],
                    feats["aggregated"] if feats["aggregated"] is not None else 0.0,
                ]
                vec = base_feats + list(self.model_trend_features.get(mname, {}).values())
            else:
                raise ValueError(f"Unknown feature_set: {feature_set}")

            X_list.append(vec)
            y_path.append(ps)
            y_treat.append(float(tm))
            groups.append(pid)
            model_labels.append(mname)

        X = np.asarray(X_list, dtype=float)
        y_path = np.asarray(y_path, dtype=float)
        y_treat = np.asarray(y_treat, dtype=int)  # classification
        groups = np.asarray(groups)
        return X, y_path, y_treat, groups, ordered_models, model_labels

    def build_train_test_split(self, base_path: Path, train_models: List[str], test_model: str, 
                              train_split: float, feature_set: str = "all"):
        """
        Build train/test datasets where train comes from train_models and 
        test comes from a single test_model. Uses train_split for patient IDs.
        """
        all_models = train_models #+ [test_model]
        ds_train = self.build_dataset(base_path, all_models, train_models, feature_set)
        print("TRAINING DATASTE BUILT")
        ds_test = self.build_dataset(base_path, all_models, [test_model], feature_set)
        print("TESTING DS BUILT")

        if ds_train is None or ds_test is None:
            return None

        X_train, y_path_train, y_treat_train, groups_train, ordered_models_train, model_labels_train = ds_train
        X_test, y_path_test, y_treat_test, groups_test, ordered_models_test, model_labels_test = ds_test

        # Combine all patient IDs to get unique patients
        all_groups = np.concatenate([groups_train, groups_test])
        unique_patients = np.unique(all_groups)

        # Shuffle and assign
        np.random.seed(42)
        shuffled_patients = np.random.permutation(unique_patients)

        # Use train_split for train patients, rest for test
        n_train_patients = int(len(shuffled_patients) * train_split)
        train_patients = set(shuffled_patients[:n_train_patients])
        test_patients = set(shuffled_patients[n_train_patients:])

        # Remove any overlapping patients
        train_patients = train_patients - test_patients
        test_patients = test_patients - train_patients

        # Create masks
        train_mask = np.isin(groups_train, list(train_patients))
        test_mask = np.isin(groups_test, list(test_patients))

        # Subset the data
        X_train_split = X_train[train_mask]
        y_path_train_split = y_path_train[train_mask]
        y_treat_train_split = y_treat_train[train_mask]
        groups_train_split = groups_train[train_mask]

        X_test_split = X_test[test_mask]
        y_path_test_split = y_path_test[test_mask]
        y_treat_test_split = y_treat_test[test_mask]
        groups_test_split = groups_test[test_mask]

        return (X_train_split, y_path_train_split, y_treat_train_split, groups_train_split), (X_test_split, y_path_test_split, y_treat_test_split, groups_test_split)


class GridSearchAnalyzer(AccuracyAnalyzer):
    """Extended analyzer for comprehensive grid search of hyperparameters"""
    
    def __init__(self, annotations_path: Path, heatmap_results: str, patient_id_pool: int = 360, test_model: str = "deepseek"):
        super().__init__(annotations_path, heatmap_results, patient_id_pool)
        self.test_model = test_model
        
    
    def evaluate_single_config(self, X, y, groups, method: str, test_model: str, train_size: float = 1.0, 
                            cv_splits: int = 5, feature_set: str = "base") -> Dict:
        """Evaluate a single configuration and return metrics including ROC curve data"""
        
        _, classifiers = get_model_configs()
        clf = list(classifiers.values())[0]  # Use first classifier
        
        if method == "cv":
            # Cross-validation
            n_groups = len(np.unique(groups))
            cv_splits = min(cv_splits, n_groups)
            if cv_splits < 2:
                return {
                    "auroc": np.nan, "accuracy": np.nan, "f1": np.nan, "n_samples": len(y),
                    "fpr": None, "tpr": None, "roc_thresholds": None
                }
            
            gkf = GroupKFold(n_splits=cv_splits)
            
            try:
                # Single cross-validation call to ensure consistency
                if hasattr(clf, "predict_proba"):
                    y_scores = cross_val_predict(clf, X, y, groups=groups, cv=gkf, method="predict_proba")[:, 1]
                    # Convert probabilities to predictions using 0.5 threshold
                    y_pred = (y_scores > 0.5).astype(int)
                elif hasattr(clf, "decision_function"):
                    y_scores = cross_val_predict(clf, X, y, groups=groups, cv=gkf, method="decision_function")
                    # Convert decision function scores to predictions
                    y_pred = (y_scores > 0).astype(int)
                else:
                    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=gkf)
                    y_scores = y_pred.astype(float)  # Use predictions as scores
                
                # Compute metrics
                if len(np.unique(y)) > 1 and y_scores is not None:
                    fpr, tpr, thresholds = roc_curve(y, y_scores)
                    auroc = auc(fpr, tpr)
                else:
                    fpr, tpr, thresholds = None, None, None
                    auroc = np.nan
                
                accuracy = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, average="binary")
                n_samples = len(y)
                
            except Exception as e:
                print(f"CV evaluation failed: {e}")
                return {
                    "auroc": np.nan, "accuracy": np.nan, "f1": np.nan, "n_samples": len(y),
                    "fpr": None, "tpr": None, "roc_thresholds": None
                }
                
        else:  # method == "holdout"
            base_path = BASE_PATH
            ordered_models = ["gpt-4.1", "gpt-5", "gpt-5-med", "o4-mini", "o3", 'deepseek', 'gpt-5-high', 'llama']

            # Identify train/test models
            train_models = [m for m in ordered_models if m != test_model]

            # Build proper train/test split using your function
            train_split = train_size  # fraction of patients for training
            split_data = self.build_train_test_split(base_path, train_models, test_model, train_split, feature_set=feature_set)
            if split_data is None:
                return {
                    "auroc": np.nan, "accuracy": np.nan, "f1": np.nan, "n_samples": 0,
                    "fpr": None, "tpr": None, "roc_thresholds": None
                }

            (X_train, y_train, y_treat_train, groups_train), \
            (X_test, y_test, y_treat_test, groups_test) = split_data

            if len(y_treat_test) == 0 or len(np.unique(y_treat_test)) < 2:
                return {
                    "auroc": np.nan, "accuracy": np.nan, "f1": np.nan, "n_samples": len(y_treat_test),
                    "fpr": None, "tpr": None, "roc_thresholds": None
                }

            try:
                clf.fit(X_train, y_treat_train)

                # Get predictions and scores
                y_pred = clf.predict(X_test)
                
                if hasattr(clf, "predict_proba"):
                    y_scores = clf.predict_proba(X_test)[:, 1]
                elif hasattr(clf, "decision_function"):
                    y_scores = clf.decision_function(X_test)
                else:
                    y_scores = y_pred.astype(float)

                # Compute metrics
                if len(np.unique(y_treat_test)) > 1:
                    fpr, tpr, thresholds = roc_curve(y_treat_test, y_scores)
                    auroc = auc(fpr, tpr)
                else:
                    fpr, tpr, thresholds = None, None, None
                    auroc = np.nan
                    
                accuracy = accuracy_score(y_treat_test, y_pred)
                f1 = f1_score(y_treat_test, y_pred, average="binary")
                n_samples = len(y_treat_test)

            except Exception as e:
                print(f"Holdout evaluation failed: {e}")
                return {
                    "auroc": np.nan, "accuracy": np.nan, "f1": np.nan, "n_samples": len(y_treat_test),
                    "fpr": None, "tpr": None, "roc_thresholds": None
                }
                
        return {
            "auroc": auroc, "accuracy": accuracy, "f1": f1, "n_samples": n_samples,
            "fpr": fpr, "tpr": tpr, "roc_thresholds": thresholds
        }
    
    def run_grid_search(self, base_path: Path, model_names: List[str], 
                       train_sizes: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                       feature_sets: List[str] = ["base", "base_aggregated", "trend_only", "all"],
                       methods: List[str] = ["cv", "holdout"],
                       outdir: Path = Path("./grid_search_results"), 
                       test_model: str = "deepseek"):
        """Run comprehensive grid search"""
        
        print("="*60)
        print("RUNNING COMPREHENSIVE GRID SEARCH")
        print("="*60)
        
        results = []
        total_configs = len(train_sizes) * len(feature_sets) * len(methods)
        config_count = 0
        
        for train_size in train_sizes:
            for feature_set in feature_sets:
                for method in methods:
                    config_count += 1
                    print(f"\nConfig {config_count}/{total_configs}: "
                          f"train_size={train_size}, features={feature_set}, method={method}")
                    
                    try:
                        
                        # Evaluate treatment accuracy (classification)
                        if method == "cv":# Build dataset with specified feature set
                            ds = self.build_dataset(base_path, model_names, model_names, feature_set)
                            if ds is None:
                                print("  -> No data, skipping")
                                continue
                            
                            X, y_path, y_treat, groups, ordered_models, model_labels = ds
                        else:
                            X, y_treat, groups = None, None, None

                        metrics = self.evaluate_single_config(
                            X, y_treat, groups, method, test_model, float(train_size), feature_set=feature_set
                        )
                        
                        result = { 
                            "train_size": train_size,
                            "feature_set": feature_set,
                            "method": method,
                            "n_features": X.shape[1] if X is not None else 0,
                            **metrics
                        }
                        results.append(result)
                        
                        print(f"  -> AUROC: {metrics['auroc']:.3f}, "
                              f"Acc: {metrics['accuracy']:.3f}, "
                              f"F1: {metrics['f1']:.3f}, "
                              f"N: {metrics['n_samples']}")
                        
                    except Exception as e:
                        print(f"  -> Error: {e}")
                        continue
        
        # Save results
        outdir.mkdir(parents=True, exist_ok=True)
        results_df, roc_data = self._results_to_dataframe(results)
        data_serializable = {
            int_key: {str_key: array.tolist() if hasattr(array, 'tolist') else list(array) 
                    for str_key, array in inner_dict.items()}
            for int_key, inner_dict in roc_data.items()
        }

        # Save
        with open(f"{outdir}/roc_{test_model}.json", 'w') as f:
            json.dump(data_serializable, f)
        results_df.to_csv(outdir / "grid_search_results.csv", index=False)
        
        # Create comprehensive plots
        self._plot_grid_results(results_df, outdir)
        
        return results_df
    
    def _results_to_dataframe(self, results):
        """Convert results list to pandas DataFrame, handling ROC curve arrays"""
        
        # Separate the ROC curve data from scalar metrics
        scalar_results = []
        roc_data = {}
        
        for i, result in enumerate(results):
            # Extract scalar metrics
            scalar_result = {k: v for k, v in result.items() 
                           if k not in ['fpr', 'tpr', 'roc_thresholds']}
            scalar_result['config_id'] = i  # Add unique identifier
            scalar_results.append(scalar_result)
            
            # Store ROC curve data separately with config_id as key
            roc_data[i] = {
                'fpr': result.get('fpr'),
                'tpr': result.get('tpr'),
                'roc_thresholds': result.get('roc_thresholds')
            }
        
        df = pd.DataFrame(scalar_results)
        
        # Store ROC data as an attribute of the dataframe for later access
        df.roc_data = roc_data
        return df, roc_data
    
    def _plot_grid_results(self, df, outdir: Path):
        """Create comprehensive visualization of grid search results"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. AUROC vs Training Size by Feature Set and Method
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("AUROC Performance Across Different Configurations", fontsize=16)
        
        # Plot 1: AUROC vs Training Size (separate lines for feature sets)
        ax = axes[0, 0]
        for method in df['method'].unique():
            for feature_set in df['feature_set'].unique():
                subset = df[(df['method'] == method) & (df['feature_set'] == feature_set)]
                if not subset.empty:
                    ax.plot(subset['train_size'], subset['auroc'], 
                           marker='o', label=f"{method}-{feature_set}", linewidth=2)
        
        ax.set_xlabel("Training Size")
        ax.set_ylabel("AUROC")
        ax.set_title("AUROC vs Training Size")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: AUROC vs Number of Features
        ax = axes[0, 1]
        for method in df['method'].unique():
            subset = df[df['method'] == method]
            if not subset.empty:
                ax.scatter(subset['n_features'], subset['auroc'], 
                          alpha=0.6, s=50, label=method)
        
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("AUROC")
        ax.set_title("AUROC vs Number of Features")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Method Comparison (boxplot)
        ax = axes[1, 0]
        methods = df['method'].unique()
        auroc_by_method = [df[df['method'] == method]['auroc'].dropna() for method in methods]
        ax.boxplot(auroc_by_method, labels=methods)
        ax.set_ylabel("AUROC")
        ax.set_title("AUROC Distribution by Method")
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Feature Set Comparison (boxplot)
        ax = axes[1, 1]
        feature_sets = df['feature_set'].unique()
        auroc_by_features = [df[df['feature_set'] == fs]['auroc'].dropna() for fs in feature_sets]
        ax.boxplot(auroc_by_features, labels=feature_sets)
        ax.set_ylabel("AUROC")
        ax.set_title("AUROC Distribution by Feature Set")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(outdir / "grid_search_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Heatmaps
        self._create_heatmaps(df, outdir)
        
        # 3. Individual ROC curves for best configurations
        self._plot_best_config_rocs(df, outdir)
    
    def _create_heatmaps(self, df, outdir: Path):
        """Create heatmaps for different metric combinations"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Heatmap 1: AUROC by Training Size vs Feature Set (CV method)
        cv_data = df[df['method'] == 'cv']
        if not cv_data.empty:
            pivot1 = cv_data.pivot_table(values='auroc', index='feature_set', 
                                        columns='train_size', aggfunc='mean')
            sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
            axes[0].set_title('AUROC Heatmap: CV Method\n(Feature Set vs Training Size)')
        
        # Heatmap 2: AUROC by Training Size vs Feature Set (Holdout method)
        holdout_data = df[df['method'] == 'holdout']
        if not holdout_data.empty:
            pivot2 = holdout_data.pivot_table(values='auroc', index='feature_set', 
                                             columns='train_size', aggfunc='mean')
            sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='viridis', ax=axes[1])
            axes[1].set_title('AUROC Heatmap: Holdout Method\n(Feature Set vs Training Size)')
        
        # Heatmap 3: Method vs Feature Set (averaged over training sizes)
        pivot3 = df.pivot_table(values='auroc', index='feature_set', 
                               columns='method', aggfunc='mean')
        sns.heatmap(pivot3, annot=True, fmt='.3f', cmap='viridis', ax=axes[2])
        axes[2].set_title('AUROC Heatmap\n(Feature Set vs Method)')
        
        plt.tight_layout()
        plt.savefig(outdir / "auroc_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_best_config_rocs(self, df, outdir: Path):
        """Plot actual ROC curves for the best performing configurations using stored data"""
        
        # Find top 5 configurations by AUROC
        top_configs = df #.nlargest(5, 'auroc')
        
        print(f"Plotting stored ROC curves for top {len(top_configs)} configurations...")
        
        plt.figure(figsize=(6, 6))
        
        colors = plt.cm.get_cmap("tab20", len(top_configs)).colors
        
        valid_curves = 0
        
        for idx, (_, config) in enumerate(top_configs.iterrows()):
            config_id = config['config_id']
            
            # Get stored ROC data
            if hasattr(df, 'roc_data') and config_id in df.roc_data:
                roc_info = df.roc_data[config_id]
                fpr = roc_info['fpr']
                tpr = roc_info['tpr']
                
                if fpr is not None and tpr is not None:
                    # Plot the stored curve
                    label = (f"Feature Set: {config['feature_set']}, AUROC={config['auroc']:.3f}")
                    if config["method"]=="holdout":
                        label= (f"Feature Set: {config['feature_set']}\n"
                            f"training set size={config['train_size']:.1f}, AUROC={config['auroc']:.3f}")
                    plt.plot(fpr, tpr, color=colors[idx], linewidth=2.5, 
                            marker='o', markersize=3, alpha=0.8, label=label)
                    
                    valid_curves += 1
                    print(f"  Plotted: {config['method']}-{config['feature_set']}, "
                          f"AUROC={config['auroc']:.3f}")
                else:
                    print(f"  Skipped: {config['method']}-{config['feature_set']} "
                          f"(no valid ROC data)")
            else:
                print(f"  Skipped: {config['method']}-{config['feature_set']} "
                      f"(ROC data not found)")
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
        
        # Formatting
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Treatment Prediction ROC Curves by Feature Set', fontsize=14, fontweight='bold')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.legend(loc='lower right', fontsize=6)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        
        plt.tight_layout()
        plt.savefig(outdir / "top_configs_stored_roc_vcv.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"ROC plot saved with {valid_curves} valid curves")
        return valid_curves
    
    def plot_detailed_comparisons(self, df, outdir: Path):
        """Create detailed comparison plots"""
        
        # 1. Training Size Impact Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Detailed Training Size Impact Analysis", fontsize=16)
        
        # Plot training size impact for each method separately
        for i, method in enumerate(['cv', 'holdout']):
            method_data = df[df['method'] == method]
            
            if method_data.empty:
                continue
                
            ax = axes[0, i]
            for feature_set in method_data['feature_set'].unique():
                subset = method_data[method_data['feature_set'] == feature_set]
                ax.plot(subset['train_size'], subset['auroc'], 
                       marker='o', linewidth=2, markersize=6, label=feature_set)
            
            ax.set_xlabel('Training Size')
            ax.set_ylabel('AUROC')
            ax.set_title(f'AUROC vs Training Size - {method.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.5, 1.0)
        
        # Plot feature set comparison
        ax = axes[1, 0]
        feature_means = df.groupby('feature_set')['auroc'].mean().sort_values(ascending=True)
        feature_stds = df.groupby('feature_set')['auroc'].std()
        
        bars = ax.barh(range(len(feature_means)), feature_means.values)
        ax.errorbar(feature_means.values, range(len(feature_means)), 
                   xerr=feature_stds.values, fmt='none', color='black', capsize=5)
        ax.set_yticks(range(len(feature_means)))
        ax.set_yticklabels(feature_means.index)
        ax.set_xlabel('Mean AUROC')
        ax.set_title('Feature Set Performance Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, feature_means.values)):
            ax.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center')
        
        # Plot method comparison
        ax = axes[1, 1]
        method_means = df.groupby('method')['auroc'].mean()
        method_stds = df.groupby('method')['auroc'].std()
        
        bars = ax.bar(method_means.index, method_means.values, 
                     yerr=method_stds.values, capsize=10, alpha=0.7)
        ax.set_ylabel('Mean AUROC')
        ax.set_title('Method Performance Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, method_means.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(outdir / "detailed_comparisons.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, df, outdir: Path):
        """Generate a comprehensive summary report"""
        
        report = []
        report.append("="*60)
        report.append("GRID SEARCH ANALYSIS SUMMARY REPORT")
        report.append("="*60)
        report.append("")
        
        # Overall statistics
        report.append(f"Total configurations tested: {len(df)}")
        report.append(f"Training sizes tested: {sorted(df['train_size'].unique())}")
        report.append(f"Feature sets tested: {list(df['feature_set'].unique())}")
        report.append(f"Methods tested: {list(df['method'].unique())}")
        report.append("")
        
        # Best performing configurations
        report.append("TOP 5 CONFIGURATIONS BY AUROC:")
        report.append("-" * 40)
        top_5 = df.nlargest(5, 'auroc')
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            report.append(f"{idx}. Method: {row['method']}, Features: {row['feature_set']}, "
                         f"Train Size: {row['train_size']:.1f}")
            report.append(f"   AUROC: {row['auroc']:.3f}, Accuracy: {row['accuracy']:.3f}, "
                         f"F1: {row['f1']:.3f}, N: {row['n_samples']}")
            report.append("")
        
        # Method comparison
        report.append("METHOD PERFORMANCE SUMMARY:")
        report.append("-" * 30)
        method_stats = df.groupby('method')['auroc'].agg(['mean', 'std', 'count'])
        for method, stats in method_stats.iterrows():
            report.append(f"{method.upper()}: Mean AUROC = {stats['mean']:.3f} ± {stats['std']:.3f} "
                         f"(n={stats['count']})")
        report.append("")
        
        # Feature set comparison
        report.append("FEATURE SET PERFORMANCE SUMMARY:")
        report.append("-" * 35)
        feature_stats = df.groupby('feature_set')['auroc'].agg(['mean', 'std', 'count'])
        for feature_set, stats in feature_stats.iterrows():
            report.append(f"{feature_set}: Mean AUROC = {stats['mean']:.3f} ± {stats['std']:.3f} "
                         f"(n={stats['count']})")
        report.append("")
        
        # Training size analysis
        report.append("TRAINING SIZE IMPACT:")
        report.append("-" * 20)
        size_stats = df.groupby('train_size')['auroc'].agg(['mean', 'std'])
        for size, stats in size_stats.iterrows():
            report.append(f"Size {size:.1f}: Mean AUROC = {stats['mean']:.3f} ± {stats['std']:.3f}")
        report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS:")
        report.append("-" * 12)
        
        # Best method
        best_method = method_stats.idxmax()['mean']
        report.append(f"• Best performing method: {best_method.upper()}")
        
        # Best feature set
        best_features = feature_stats.idxmax()['mean']
        report.append(f"• Best performing feature set: {best_features}")
        
        # Training size effect
        size_corr = df[['train_size', 'auroc']].corr().iloc[0, 1]
        if size_corr > 0.1:
            report.append(f"• Training size shows positive correlation with AUROC (r={size_corr:.3f})")
        elif size_corr < -0.1:
            report.append(f"• Training size shows negative correlation with AUROC (r={size_corr:.3f})")
        else:
            report.append(f"• Training size shows minimal correlation with AUROC (r={size_corr:.3f})")
        
        # Save report
        with open(outdir / "summary_report.txt", "w") as f:
            f.write("\n".join(report))
        
        # Print to console
        print("\n".join(report))


# --------------------------
# Extended CLI
# --------------------------

if __name__ == "__main__":
    BASE_PATH = Path("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean")
    ANNOTATIONS_PATH = BASE_PATH / "data" / "benchmarks" / "human_annotations"

    parser = argparse.ArgumentParser(description="Extended accuracy prediction with comprehensive grid search.")
    parser.add_argument("--mode", choices=["cv", "split", "grid_search"], required=True,
                        help="Evaluation mode: 'cv', 'split', or 'grid_search'")
    
    # For CV mode
    parser.add_argument("--models", nargs="+",
                        help='List of model names for CV mode')
    parser.add_argument("--cv-splits", type=int, default=5,
                        help="Number of GroupKFold splits for CV mode.")
    
    # For split mode
    parser.add_argument("--train-models", nargs="+",
                        help='List of model names for training in split mode')
    parser.add_argument("--testmodel", type=str,
                        help='Single model name for testing in split mode')
    
    # For grid search mode
    parser.add_argument("--train-sizes", nargs="+", type=float, 
                        default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help="Training sizes to test in grid search")
    parser.add_argument("--feature-sets", nargs="+", 
                        default=["base", "base_aggregated", "trend_only", "all", "aggregated_only"],
                        help="Feature sets to test in grid search")
    parser.add_argument("--methods", nargs="+", 
                        choices=["cv", "holdout"],
                        default=["cv", "holdout"],
                        help="Evaluation methods to test in grid search")
    parser.add_argument("--heatmap_results", type=str, default="/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/results/0904/heatmap_results_2.json")
    
    # Common arguments
    parser.add_argument("--outdir", type=Path, default=Path("./pred_eval_outputs"),
                        help="Output directory for plots and results.")
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--feature-set", choices=["base", "base_aggregated", "trend_only", "all"],
                        default="all", help="Feature set to use for cv/split modes")
    
    args = parser.parse_args()

    if args.mode == "grid_search":
        if not args.models:
            parser.error("--models is required for grid search mode")
        
        analyzer = GridSearchAnalyzer(annotations_path=ANNOTATIONS_PATH, heatmap_results = args.heatmap_results, patient_id_pool=360)
        
        print(f"Running grid search with:")
        print(f"  Models: {args.models}")
        print(f"  Training sizes: {args.train_sizes}")
        print(f"  Feature sets: {args.feature_sets}")
        print(f"  Methods: {args.methods}")
        
        # Run comprehensive grid search
        results_df = analyzer.run_grid_search(
            base_path=BASE_PATH,
            model_names=args.models,
            train_sizes=args.train_sizes,
            feature_sets=args.feature_sets,
            methods=args.methods,
            outdir=args.outdir,
            test_model=args.testmodel if args.testmodel else "deepseek"
        )
        
        # Generate additional detailed plots
        analyzer.plot_detailed_comparisons(results_df, args.outdir)
        
        # Generate summary report
        analyzer.generate_summary_report(results_df, args.outdir)
    