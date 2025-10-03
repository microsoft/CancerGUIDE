from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import json
import re
import os
import json5
import argparse

#-------------------------
# UTILITY FUNCTIONS
# -----------------------------

def parse_path(path_string: str) -> List[str]:
    """Parse a path string into individual nodes."""
    try:
        if not path_string or path_string.strip() == "":
            return []
        return [node.strip() for node in path_string.split("->")]
    except:
        return path_string

from typing import List, Optional, Tuple

def filter_parsed_path(lst):
    if lst:
        if lst[-1]=="NSCL-17-10":
            lst=lst[:-1]
    if lst:
        if lst[-1]=="NSCL-17-1":
            lst=lst[:-1]
    return lst

def find_divergence_point(paths: List[str]) -> Tuple[Optional[str], List[str]]:
    """
    Find where paths diverge and return the source node and target nodes.
    Returns (source_node, target_nodes) where source_node is the last common node
    and target_nodes are the different next nodes (or 'end' if path ends).
    
    Benign divergences are ignored:
      - One ends at 17-1 while another ends at 17-1,17-10
      - One ends at ...,X while another ends at ...,X,17-1
    """
    parsed_paths = [parse_path(path) for path in paths if path]
    parsed_paths = [filter_parsed_path(path) for path in parsed_paths]
    if len(parsed_paths) < 2:
        return None, []

    # # --- Special handling: benign extension rules ---
    # def is_benign_extension(p1, p2) -> bool:
    #     """Return True if p1 and p2 differ only by trailing 17-1 or 17-10."""
    #     if p1 == p2:
    #         return True
    #     # Case A: one ends at 17-1, other ends at 17-1,17-10
    #     if len(p1) + 1 == len(p2) and p1 == p2[:-1] and p2[-1] == "NSCL-17-10" and p1[-1] == "NSCL-17-1":
    #         return True
    #     if len(p2) + 1 == len(p1) and p2 == p1[:-1] and p1[-1] == "NSCL-17-10" and p2[-1] == "NSCL-17-1":
    #         return True
    #     # Case B: one ends at ...,X and the other at ...,X,17-1
    #     if len(p1) + 1 == len(p2) and p1 == p2[:-1] and p2[-1] == "NSCL-17-1":
    #         return True
    #     if len(p2) + 1 == len(p1) and p2 == p1[:-1] and p1[-1] == "NSCL-17-1":
    #         return True
    #     if len(p1) + 2 == len(p2) and p1 == p2[:-2] and p2[-1] == "NSCL-17-10" and p2[-2] == "NSCL-17-1":
    #         return True
    #     if len(p2) + 2 == len(p1) and p2 == p1[:-2] and p1[-1] == "NSCL-17-10" and p1[-2] == "NSCL-17-1":
    #         return True
    #     return False

    # # If *all pairs* are equivalent up to benign extensions, no divergence
    # all_equivalent = True
    # for i in range(len(parsed_paths)):
    #     for j in range(i + 1, len(parsed_paths)):
    #         if not is_benign_extension(parsed_paths[i], parsed_paths[j]):
    #             all_equivalent = False
    #             break
    #     if not all_equivalent:
    #         break
    # if all_equivalent:
    #     return None, []

    # --- Normal divergence detection ---
    min_length = min(len(path) for path in parsed_paths)
    last_common_idx = -1

    for i in range(min_length):
        nodes_at_position = [path[i] for path in parsed_paths]
        if len(set(nodes_at_position)) == 1:  # all nodes match
            last_common_idx = i
        else:
            break

    if last_common_idx == -1:
        # Diverge from the start
        source_node = "START"
        target_nodes = list(set(path[0] if path else "end" for path in parsed_paths))
    else:
        source_node = parsed_paths[0][last_common_idx]
        target_nodes = []
        for path in parsed_paths:
            if len(path) > last_common_idx + 1:
                target_nodes.append(path[last_common_idx + 1])
            else:
                target_nodes.append(f"{path[last_common_idx]}-end")
        target_nodes = list(set(target_nodes))

    # Clean redundant target_nodes like "17-1" vs "17-1,17-10"
    for target in target_nodes.copy():
        for target_2 in target_nodes:
            if target == target_2:
                continue
            if target.startswith(target_2):
                target_nodes.remove(target)
                break
    if "NSCL-17-1" in target_nodes:
        print(parsed_paths)
        print(source_node, target_nodes)
        # breakpoint()
    return source_node, target_nodes



def load_patient_iterations_data(patient_id: str) -> Optional[Dict]:
    """Load the patient iteration data from the JSON file."""
    patient_dir = DATA_PATH / f"patient_{patient_id}"

    path_options= []
    path_match=True
    path=patient_dir / f"matched_outputs_{patient_id}_k_3.json"
    if path.exists():
        with open(patient_dir / f"matched_outputs_{patient_id}_k_3.json", "r", encoding="utf-8") as f:
            data_text = f.read()
            data = extract_json_from_text(data_text)["final_path_match"]
            path_match = data >= 0.9

    # Look for files that might contain the iteration data
    for file_path in patient_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data_text = f.read()
                data = extract_json_from_text(data_text)["final_path"]
                if extract_json_from_text(data_text).get("reason"):
                    if "NOT_GUIDELINE_COMPLIANT" in extract_json_from_text(data_text)["reason"]:
                        data="NOT_GUIDELINE_COMPLIANT"
                path_options.append(data)

        except (json.JSONDecodeError, Exception):
            continue

    return path_match, path_options


def save_result_to_file(result, file_path: str):
    """Save a result to file, creating directories as needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        if hasattr(result, 'parsed_output'):
            f.write(result.parsed_output)
        else:
            f.write(str(result))

def extract_json_from_text(text: str):
    """
    Extract and parse the first valid JSON object found in the input text.
    Handles markdown ```json ... ``` blocks and loose JSON (e.g., trailing commas).
    Returns a dict if successful, else None.
    """
    # Find JSON inside markdown code blocks first
    match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
    json_str = match.group(1) if match else None

    # If no markdown block, try to find first {...} block
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
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            return json.loads(json_str)

    return None

# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Analyze human error in RLFollow paths.")
    argparser.add_argument("--model", required=True, help="Model to evaluate")
    argparser.add_argument("--data_path", required=True, help="Path to the data directory")
    argparser.add_argument("--output_dir", required=True, help="Path to the output directory")
    args = argparser.parse_args()

    MODEL = args.model
    BASE_PATH = Path("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean")
    DATA_PATH = Path(args.data_path) #BASE_PATH / f"results/benchmark_results/{MODEL}/human_new_prompt"
    OUTPUT_DIR = Path(args.output_dir) #BASE_PATH / f"results/error_analysis_0819/human_{MODEL}/human_{MODEL}"
    
    # Process each patient directory
    for entry in DATA_PATH.iterdir():
        if not entry.is_dir():
            continue
            
        patient_id = entry.name.split("_")[1]
        
        # Load iteration data
        path_match, paths = load_patient_iterations_data(patient_id)

        if not paths:
            print(f"No iteration data found for patient {patient_id}")
            continue

        
        # Find divergence point
        source_node, target_nodes = find_divergence_point(paths)
        
        if not source_node:
            print(f"Patient {patient_id}: Had no divergence!")
            continue
        
        print(f"Patient {patient_id}: Divergence at {source_node} -> {target_nodes}")
        
        # Create output path
        output_path = OUTPUT_DIR / f"{patient_id}_evaluation.json"
        
        
        try:
            result_data={}
            result_data["node_mistake"]= source_node + " ->[" + ", ".join(target_nodes)+"]"
            result_data['computed_source_node'] = source_node
            result_data['computed_target_nodes'] = target_nodes
            result_data['original_paths'] = paths
            result_data['patient_id'] = patient_id
                            
            # Save enhanced result
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2)
        except Exception as e:
            print(f"Error processing result for patient {patient_id}: {e}")