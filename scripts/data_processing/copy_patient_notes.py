import os
import json
import re
import pandas as pd
from typing import Dict, Optional

# Base directory containing all patient subfolders
BASE_DIR = "/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/results/patient_rollouts_o4_mini"
RESULT_DIR = "/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/data/consistency_bench_o4-mini"
PATIENT_DATA = "/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/data/patient_notes_raw.csv"
TRUE_DIR = os.path.join(RESULT_DIR, "consistent")
FALSE_DIR = os.path.join(RESULT_DIR, "inconsistent")

# Ensure output directories exist
os.makedirs(TRUE_DIR, exist_ok=True)
os.makedirs(FALSE_DIR, exist_ok=True)

consistent_not_compliant = 0
inconsistent_not_compliant = 0
no_common_path = 0

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON data from text, handling markdown code blocks."""
    # Try to extract JSON from markdown code block
    match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
    if not match:
        # Fallback: try to extract standalone JSON object
        match = re.search(r"({.*?})", text, re.DOTALL)
    
    if not match:
        return None
    
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

# Iterate through subdirectories
consistent_count = 0
inconsistent_count = 0
for patient_folder in os.listdir(BASE_DIR):
    patient_path = os.path.join(BASE_DIR, patient_folder)

    if not os.path.isdir(patient_path) or not patient_folder.startswith("patient_"):
        continue

    try:
        patient_id = patient_folder.split("_")[1]
        outputs_file = os.path.join(patient_path, f"matched_outputs_{patient_id}.json")
        patient_instance = pd.read_csv(PATIENT_DATA).loc[int(patient_id)]
        note_text = patient_instance["DEID_NOTE_TXT_DD48_NOTES"]
        # Load matched outputs
        with open(outputs_file, "r") as f:
            matched_outputs = extract_json_from_text(f.read())

        if matched_outputs is None:
            print(f"Could not extract JSON from: {outputs_file}")
            continue

        # Determine match status
        is_match = matched_outputs.get("final_path_match", False)
        if is_match:
            consistent_count += 1
        else:
            inconsistent_count += 1
        label = matched_outputs.get("final_path_mode", None)
        is_compliant = matched_outputs

        if label is None:
            print(f"No final_path_mode in: {outputs_file}")
            continue  # Skip if label is missing

        # Create output data
        output_data = {
            "patient_note": note_text,
            "label": label
        }
        if is_match and label == "NOT_GUIDELINE_COMPLIANT":
            consistent_not_compliant += 1
        elif not is_match and label == "NOT_GUIDELINE_COMPLIANT":
            inconsistent_not_compliant += 1
        elif label == "No common path":
            no_common_path += 1
        # Write to correct directory
        output_dir = TRUE_DIR if is_match else FALSE_DIR
        output_file = os.path.join(output_dir, f"patient_{patient_id}.json")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    except Exception as e:
        print(f"Failed to process {patient_folder}: {e}")
    
# Summary of processed files
print(f"Processed {consistent_count} consistent and {inconsistent_count} inconsistent patient notes.")
print(f"Consistent but not compliant: {consistent_not_compliant}")
print(f"Inconsistent and not compliant: {inconsistent_not_compliant}")
print(f"No common path found: {no_common_path}")
