from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import json
import re
import os
import json5
import argparse

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def parse_path(path_string: str) -> List[str]:
    """Parse a path string into individual nodes."""
    try:
        if not path_string or path_string.strip() == "":
            return []
        return [node.strip() for node in re.split(r'-{1,2}>', path_string)]
    except:
        return path_string

def filter_parsed_path(lst):
    if lst:
        if lst[-1]=="NSCL-17-10":
            lst=lst[:-1]
    if lst:
        if lst[-1]=="NSCL-17-1":
            lst=lst[:-1]
    return lst

def find_divergence_point(patient_id) -> Tuple[Optional[str], List[str], Optional[bool]]:
    """
    Find where paths diverge and return the source node and target nodes.
    Returns (source_node, target_nodes, path_match) where:
      - source_node is the last common node
      - target_nodes are the different next nodes (or 'end' if path ends)
      - path_match is True if paths are equivalent under relaxed rules
    """
    path_match = None
    predicted_path_file = DATA_PATH / f"patient_{patient_id}/path_prediction_{patient_id}.json"
    with open(predicted_path_file, "r", encoding="utf-8") as f:
        path_text = f.read()
        path_text_extracted = extract_json_from_text(path_text)
        if path_text_extracted is None:
            return None, [], None
        if "final_path" not in path_text_extracted:
            return None, [], None
        predicted_path = path_text_extracted["final_path"]

    ground_truth_path_file = (
        f"/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/"
        f"RLFollow_clean/data/benchmarks/human_annotations/patient_{patient_id}.json"
    )
    with open(ground_truth_path_file, "r", encoding="utf-8") as f:
        ground_truth_path = json.load(f)["label"]

    parsed_paths = [parse_path(predicted_path), parse_path(ground_truth_path)]
    parsed_paths = [filter_parsed_path(path) for path in parsed_paths]

    if len(parsed_paths) < 2:
        return None, [], None
    extract_json_from_text(path_text)
    # # --- Special handling: ignore benign divergences ---
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
    
    # if is_benign_extension(parsed_paths[0], parsed_paths[1]):
    #     return parsed_paths[0][-1] if parsed_paths[0] else "START", [], True

    # --- Normal divergence detection ---
    min_length = min(len(path) for path in parsed_paths)
    last_common_idx = -1
    
    for i in range(min_length):
        nodes_at_position = [path[i] for path in parsed_paths]
        if len(set(nodes_at_position)) == 1:
            last_common_idx = i
        else:
            break
    
    if last_common_idx == -1:
        # Paths diverge from the start
        source_node = "START"
        target_nodes = list(set(path[0] for path in parsed_paths if path))
    else:
        source_node = parsed_paths[0][last_common_idx]
        target_nodes = []
        
        for path in parsed_paths:
            if len(path) > last_common_idx + 1:
                target_nodes.append(path[last_common_idx + 1])
            else:
                 target_nodes.append(f"{path[last_common_idx]}-end")
        
        target_nodes = list(set(target_nodes))

    if parsed_paths[0] == parsed_paths[1]:
        path_match = True
    # if "NSCL-17-1" in target_nodes:
    #     breakpoint()
    return source_node, target_nodes, path_match



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
        return None

    return None

def load_patient_match_data(patient_id):
    pathway = DATA_PATH / f"patient_{patient_id}/path_prediction_{patient_id}.json"
    with open(pathway, "r", encoding="utf-8") as f:
        data_text = f.read()
        path_match = extract_json_from_text(data_text)["path_match"]
        if path_match=="partial":
            path_match="true"
    if path_match=="true":
        return True
    return False

# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    argparser = argparse.ArgumentParser(description="Analyze human error in RLFollow paths.")
    argparser.add_argument("--model", required=True, help="Model to evaluate")
    argparser.add_argument("--data_path", required=True, help="Path to the data directory")
    argparser.add_argument("--output_dir", required=True, help="Path to the output directory")
    args = argparser.parse_args()

    MODEL = args.model
    BASE_PATH = Path("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean")
    DATA_PATH = Path(args.data_path) #BASE_PATH / f"results/benchmark_results/{MODEL}/human_new_prompt"
    OUTPUT_DIR = Path(args.output_dir) #BASE_PATH / f"results/error_analysis_0819/human_{MODEL}/human_{MODEL}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    

    # Process each patient directory
    for entry in DATA_PATH.iterdir():
        if not entry.is_dir():
            continue
            
        patient_id = entry.name.split("_")[1]
        # Find divergence point
        source_node, target_nodes, path_match = find_divergence_point(patient_id)
        if path_match:
            print(f"Patient {patient_id}: Paths match, skipping analysis")
            continue
        
        if not source_node:
            print(f"Patient {patient_id}: Could not determine divergence point")
            continue
        

        # Create output path
        output_path = OUTPUT_DIR / f"{patient_id}_evaluation.json"
        
        # if output_path.exists():
        #     print(f"Skipping existing file: {output_path}")
        #     continue
        
        # Create analysis prompt
        if path_match:
            print(f"Patient {patient_id}: Paths match, skipping analysis")
            continue

        # Process with LLM
        try:
            result_data={}
            result_data["node_mistake"]= source_node + " ->[" + ", ".join(target_nodes)+"]"
            result_data['computed_source_node'] = source_node
            result_data['computed_target_nodes'] = target_nodes
            result_data['patient_id'] = patient_id
                            
            # Save enhanced result
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2)
        except Exception as e:
            print(f"Error processing result for patient {patient_id}: {e}")