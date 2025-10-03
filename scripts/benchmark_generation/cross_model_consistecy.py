from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple
import json
import argparse
import re
import json5
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_PATH = Path("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean")
HUMAN_LABELS_DIR = BASE_PATH / "data/benchmarks/human_annotations"

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def load_json(output_path):
    """
    Extract and parse the first valid JSON object found in the input text.
    Handles markdown ```json ... ``` blocks and loose JSON (e.g., trailing commas).
    Returns a dict if successful, else None.
    """
    # Find JSON inside markdown code blocks first
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            text=f.read()
        if "</think>" in text:
            text = text.split("</think>")[-1]
        match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
        json_str = match.group(1) if match else None

        # If no markdown block, try to find first {...} block
        if not json_str:
            match = re.search(r"({.*?})", text, re.DOTALL)
            json_str = match.group(1) if match else None

        if not json_str:
            return None

        parsed = json5.loads(json_str)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            breakpoint()
            return None

    return None

def split_path(path_str) -> List[str]:
    """Split a path string by '->' into a list of nodes."""
    if isinstance(path_str, list):
        return path_str
    return [node.strip() for node in path_str.split("->") if node.strip()]

def filter_path_17_node(lst):
    if lst:
        if lst[-1]=="NSCL-17-10" and len(lst) > 1:
            lst=lst[:-1]
    if lst:
        if lst[-1]=="NSCL-17-1" and len(lst) > 1:
            lst=lst[:-1]
    return lst

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

def process_patient_single(patient_id: str, pred_file: Path) -> Tuple[float, bool]:
    """Process a single patient in normal mode."""
    gt_file = HUMAN_LABELS_DIR / f"{patient_id}.json"
    if not pred_file.exists() or not gt_file.exists():
        raise FileNotFoundError(f"Missing files for patient {patient_id}")

    pred_data = load_json(pred_file)
    gt_data = load_json(gt_file)
    print("FOUND DATA ON BOTH HUMANS AND MODELS")

    pred_path_nodes = split_path(pred_data["label"])
    gt_path_nodes = split_path(gt_data["label"])
    pred_path_nodes = filter_path_17_node(pred_path_nodes)
    gt_path_nodes = filter_path_17_node(gt_path_nodes)

    path_score = compare_lists([gt_path_nodes, pred_path_nodes], return_final_consistency=False)
    treatment_match = gt_path_nodes[-1]==pred_path_nodes[-1]
    # if gt_path_nodes == ["NOT_GUIDELINE_COMPLIANT"]:
    #     return None, None
    return path_score, treatment_match


def find_duplicate_list(lists: List[List[str]]) -> Optional[List[str]]:
    lists_filtered=[]
    for lst in lists:
        lst = filter_path_17_node(lst)
        lists_filtered.append(lst)
    seen = {}
    for lst in lists_filtered:
        t = tuple(lst)  # Convert to tuple so it can be used as a dict key
        seen[t] = seen.get(t, 0) + 1
    keys_with_2 = [key for key, value in seen.items() if value == 2]
    if keys_with_2:  # Found a duplicate
        if len(keys_with_2) > 1:
            print("WARNING: Multiple duplicates found, taking the first one")
            curr_score=0
            best=None
            remainder= [key for key in seen.keys() if key not in keys_with_2]
            if len(remainder)==0 or remainder[0][0]=='':
                index = max(enumerate(keys_with_2), key=lambda x: len(x[1]))[0]
                return keys_with_2[index]
            for curr_path in keys_with_2:
                total=0
                for rem in remainder:
                    score = compare_lists([curr_path, rem], return_final_consistency=False)
                    total+=score
                if total>curr_score:
                    curr_score=total
                    best=curr_path
            if not(best):
                breakpoint()
            return list(best)
        return list(keys_with_2[0])
    return None
# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against benchmark data.")
    parser.add_argument("--results_json", default="/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/results/evaluation_results.json", type=Path, help="Path to output results.json")
    parser.add_argument("--filter_type", default="path")
    parser.add_argument("--consistency_bench_aggregated", action="store_true", help="Set to True for aggregated consistency benchmark")
    parser.add_argument("--aggregation_mode", action="store_true", help="Set to True for aggregated processing")
    parser.add_argument("--model_name", default="o4-mini", help="Model used for path prediction in single case")
    parser.add_argument("--experiment_name", default="unstructured", help="Name of the experiment")
    parser.add_argument("--model_list", default="gpt-4.1,gpt-5,o3,o4-mini", help="Comma-separated list of models")

    args = parser.parse_args()
    MODELS = args.model_list.split(",")
    if args.aggregation_mode:
        PATIENT_DATA_DIR = BASE_PATH / f"data/benchmarks/consistency_bench"
    else:
        PATIENT_DATA_DIR = BASE_PATH / f"data/benchmarks/consistency_bench/{args.model_name}/{args.filter_type}_filter"

    path_scores = []
    treatment_matches = 0
    total_patients = 0
    consensus_count = 0
    no_consensus_patients = []

    if args.aggregation_mode:
        consensus_tracking = {}
        ground_truth = None

        # Step 1: gather all patient IDs across models
        all_patient_ids = set()
        for model in MODELS:
            model_dir = PATIENT_DATA_DIR / f"{model}/{args.filter_type}_filter"
            for pred_file in model_dir.glob("*.json"):
                all_patient_ids.add(pred_file.stem)
        print(f"Found {len(all_patient_ids)} patients across models")
        # Step 2: iterate over patients, collect predictions from all models
        model_scoring_consistency_bench = {model: {"path_score": [], "treatment_match": 0, "total_patients": 0} for model in MODELS}
        for patient_id in sorted(all_patient_ids):
            model_predictions_consistency_bench = {model: [] for model in MODELS}
            prediction_list = []

            # load ground truth once per patient
            gt_file = HUMAN_LABELS_DIR / f"{patient_id}.json"
            if gt_file.exists():
                gt_data = load_json(gt_file)
                gt_path_nodes = split_path(gt_data["label"])
                ground_truth = gt_path_nodes
                # if "NOT_GUIDELINE_COMPLIANT" in ground_truth:
                #     print(f"GROUND TRUTH IS NOT COMPLIANT, skipping {patient_id}")
                #     continue
            else:
                print(f"Skipping {patient_id}: ground truth not found")
                continue

            # collect predictions from all models
            for model in MODELS:
                target_file = PATIENT_DATA_DIR / f"{model}/{args.filter_type}_filter/{patient_id}.json"
                if not target_file.exists():
                    id_number = patient_id.split("_")[1]
                    target_file = BASE_PATH / f"results/rollout_results_0815_benchmark/rollout_experiment_{model}/{patient_id}/matched_outputs_{id_number}_k_10.json"
                    if not target_file.exists():
                        print(f"Skipping {patient_id}: prediction not found for model {model}")
                        continue
                    final_path = load_json(target_file)["final_path_mode"]
                    model_predictions_consistency_bench[model] = final_path
                    continue
                prediction = load_json(target_file).get("label")
                model_predictions_consistency_bench[model] = prediction
                prediction_list.append(prediction)

            # skip patients where fewer than 2 models had predictions
            if len(prediction_list) < 2:
                print(f"Not enough predictions for patient {patient_id}")
                continue

            # Step 3: consensus + scoring
            
            consensus = find_duplicate_list(prediction_list)
            if not consensus:
                print(f"No consensus found for patient {patient_id}")
                print(prediction_list)
                
            if consensus and not args.consistency_bench_aggregated:
                path_score, treatment_match, path_mode, treatment_mode = compare_lists([gt_path_nodes, consensus], return_final_consistency=True)
                treatment_match=int(treatment_match)
                path_scores.append(path_score)
                if treatment_match:
                    treatment_matches += 1
                total_patients += 1
            if args.consistency_bench_aggregated and consensus:
                for model, predictions in model_predictions_consistency_bench.items():
                    path_score, treatment_match, path_mode, treatment_mode = compare_lists([predictions, consensus], return_final_consistency=True)
                    treatment_match=int(treatment_match)
                    path_scores.append(path_score)
                    if treatment_match:
                        treatment_matches += 1
                    total_patients += 1
                    model_scoring_consistency_bench[model]["path_score"].append(path_score)
                    model_scoring_consistency_bench[model]["treatment_match"] += treatment_match
                    model_scoring_consistency_bench[model]["total_patients"] += 1
        if not args.consistency_bench_aggregated:
            avg_path_score = sum(path_scores) / len(path_scores) if path_scores else 0
            p=treatment_matches/total_patients
            se = np.sqrt(p*(1-p)/total_patients)  
            new_entry = {
                "model": args.model_name,
                "average_path_match": avg_path_score,
                "se_path_match": np.std(path_scores) / np.sqrt(len(path_scores)) if path_scores else 0,
                "se_treatment_match": se if total_patients > 0 else 0,
                "average_treatment_match": treatment_matches / total_patients if total_patients > 0 else 0,
                "total_patients_matched": total_patients,
                "benchmark": args.experiment_name
            }

            # Load existing list if file exists & is valid JSON
            if os.path.exists(args.results_json) and os.path.getsize(args.results_json) > 0:
                with open(args.results_json, "r") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]  # in case it was a single dict
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            # Append new result
            data.append(new_entry)

            # Write back full list
            with open(args.results_json, "w") as f:
                json.dump(data, f, indent=2)
    else:
        for pred_file in PATIENT_DATA_DIR.glob("*.json"):
            patient_id = pred_file.stem
            try:
                score, match = process_patient_single(patient_id, pred_file)
                path_scores.append(score)
                if match:
                    treatment_matches += 1
                total_patients += 1
            except FileNotFoundError as e:
                print(f"Skipping {patient_id}: {e}")
        avg_path_score = sum(path_scores) / len(path_scores) if path_scores else 0
        p=treatment_matches/total_patients
        se = np.sqrt(p*(1-p)/total_patients)  
        new_entry = {
                "model": args.model_name,
                "average_path_match": avg_path_score,
                "se_path_match": np.std(path_scores) / np.sqrt(len(path_scores)) if path_scores else 0,
                "se_treatment_match": se if total_patients > 0 else 0,
                "average_treatment_match": treatment_matches / total_patients if total_patients > 0 else 0,
                "total_patients_matched": total_patients,
                "benchmark": args.experiment_name
            }

        # Load existing list if file exists & is valid JSON
        if os.path.exists(args.results_json) and os.path.getsize(args.results_json) > 0:
            with open(args.results_json, "r") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]  # in case it was a single dict
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append new result
        data.append(new_entry)

        # Write back full list
        with open(args.results_json, "w") as f:
            json.dump(data, f, indent=2)

    avg_path_score = sum(path_scores) / len(path_scores) if path_scores else 0
    print(f"Average path overlap score: {avg_path_score:.2f}")
    print(f"Treatment matches: {treatment_matches}/{total_patients}")
    print(f"Standard Error Path score: {np.std(path_scores) / np.sqrt(len(path_scores)) if path_scores else 0}")
    p=treatment_matches/total_patients
    se = np.sqrt(p*(1-p)/total_patients)  
    print(f"Standard Error Treatment Match: {se if total_patients > 0 else 0}") 
    if args.consistency_bench_aggregated:
        for model, scores in model_scoring_consistency_bench.items():
            avg_path_score = sum(scores["path_score"]) / len(scores["path_score"]) if scores["path_score"] else 0
            print(f"Model {model} - Average path score: {avg_path_score:.2f}, Treatment matches: {scores['treatment_match']}/{scores['total_patients']}")
            print(f"Standard Error Path score: {np.std(scores['path_score']) / np.sqrt(len(scores['path_score'])) if scores['path_score'] else 0}")
            print(f"Standard Error Treatment Match: {se if scores['treatment_match'] else 0}")
            new_entry = {
                "model": model,
                "average_path_match": avg_path_score,
                "se_path_match": np.std(scores['path_score']) / np.sqrt(len(scores['path_score'])) if scores['path_score'] else 0,
                "se_treatment_match": se if scores['treatment_match'] else 0,
                "average_treatment_match": scores['treatment_match'] / scores['total_patients'] if scores['total_patients'] > 0 else 0,
                "total_patients_matched": scores['total_patients'],
                "benchmark": args.experiment_name
            }

            # Load existing list if file exists & is valid JSON
            if os.path.exists(args.results_json) and os.path.getsize(args.results_json) > 0:
                with open(args.results_json, "r") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]  # in case it was a single dict
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            # Append new result
            data.append(new_entry)

            # Write back full list
            with open(args.results_json, "w") as f:
                json.dump(data, f, indent=2)