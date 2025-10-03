from sam_mtb_utils.common import configure_azure_llm
from typing import List
import os
import json
import pandas as pd
import re
from prompts import *
import json5

from sam_mtb_utils.factuality import (
    MultithreadedExecutor,
)

# Configuration constants
API_BASE = "https://rwe-aoai.openai.azure.com"
API_VERSION = "2024-12-01-preview"
BASE_PATH = "/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/CancerGUIDE_Internal"  # Change this path as needed

# Central configuration for all file paths
class PathConfig:
    def __init__(self, base_path: str = BASE_PATH, method: str = "structured"):
        self.base_path = base_path
        self.data_dir = os.path.join(base_path, "data/benchmarks")
        self.synthetic_patients_dir = os.path.join(base_path, f"gpt-5_synthetic_patients_total_{method}")
        self.filtered_patients_dir = os.path.join(base_path, f"data/benchmarks/synthetic_bench/gpt-5_synthetic_final_{method}")

        # Data files
        self.guideline_path = os.path.join(self.data_dir, "nsclsc_guideline.json")
        self.onco_column_path = os.path.join(self.data_dir, "oncological_structured_data.csv")
        self.unstructured_data_path = os.path.join(self.data_dir, "unstructured_oncological_decision_data.csv")
    
    def get_patient_dir(self, patient_id: int) -> str:
        return os.path.join(self.synthetic_patients_dir, f"patient_{patient_id}")
    
    def get_filtered_patient_dir(self, patient_id: int) -> str:
        return os.path.join(self.filtered_patients_dir, f"patient_{patient_id}")
    
    def get_patient_file(self, patient_id: int, filename: str) -> str:
        return os.path.join(self.get_patient_dir(patient_id), filename)
    
    def get_filtered_patient_file(self, patient_id: int, filename: str) -> str:
        return os.path.join(self.get_filtered_patient_dir(patient_id), filename)


# LLM configuration
llm_full = configure_azure_llm(
    deployment_name="gpt-5",
    api_version=API_VERSION,
    api_base=API_BASE,
    reasoning_effort="minimal"
)

def load_gpt_output(file_path: str):
        """
        Extract and parse the first valid JSON object found in the input text.
        Handles markdown ```json ... ``` blocks and loose JSON (e.g., trailing commas).
        Returns a dict if successful, else None.
        """
        # Find JSON inside markdown code blocks first
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
        json_str = match.group(1) if match else None

        # If no markdown block, try to find first {...} block
        if not json_str:
            match = re.search(r"({.*?})", text, re.DOTALL)
            json_str = match.group(1) if match else None

        if not json_str:
            breakpoint()
            return None

        try:
            parsed = json5.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        breakpoint()
        return None

def save_result_to_file(result, file_path: str):
    """Save a result to file, creating directories as needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result.parsed_output)

def process_prompts_with_executor(prompts: List, timeout: int = 300, max_retries: int = 2, n_threads: int = 30):
    """Process prompts using MultithreadedExecutor."""
    return MultithreadedExecutor(
        llm_full, List, timeout=timeout, max_retries=max_retries
    ).process_prompts(prompts, n_threads=n_threads)

def generate_path(number_of_patients: int = 50):
    """Generate treatment paths for patients."""
    for patient_id in range(number_of_patients):
        output_path = paths.get_patient_file(patient_id, f"path_{patient_id}.txt")
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping patient {patient_id}.")
            continue
        
        prompt = prompt_path(
            guideline_path=paths.guideline_path,
            target_node=patient_id % 38
        )
        eval_prompts = [prompt]

        eval_results = process_prompts_with_executor(eval_prompts)
        
        for result in eval_results:
            save_result_to_file(result, output_path)

def generate_structured_data(number_of_patients: int = 50):
    """Generate structured patient data."""
    onco_column_list = pd.read_csv(paths.onco_column_path).columns.tolist()
    onco_columns = "\n".join(onco_column_list)

    for patient_id in range(number_of_patients):
        output_path = paths.get_patient_file(patient_id, f"structured_patient_{patient_id}.json")
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping patient {patient_id}.")
            continue
        
        path_file = paths.get_patient_file(patient_id, f"path_{patient_id}.txt")
        target_path_json = load_gpt_output(path_file)
        if target_path_json is None:
            print(f"Error loading target path for patient {patient_id}. Skipping...")
            continue
        
        target_path = target_path_json["final_path"]
        prompt = prompt_path_structured(
            guideline_path=paths.guideline_path,
            columns=onco_columns,
            target_path=target_path
        )
        eval_prompts = [prompt]

        eval_results = process_prompts_with_executor(eval_prompts)
        
        for result in eval_results:
            save_result_to_file(result, output_path)

def generate_treatment_from_structured(number_of_patients: int = 50):
    """Generate treatment predictions from structured patient data."""
    for patient_id in range(number_of_patients):
        print(f"Processing patient {patient_id}...")
        
        matched_output_path = paths.get_patient_file(patient_id, f"structured_matched_outputs_{patient_id}.json")
        if os.path.exists(matched_output_path):
            print(f"Output file {matched_output_path} already exists. Skipping patient {patient_id}.")
            continue
        
        structured_file = paths.get_patient_file(patient_id, f"structured_patient_{patient_id}.json")
        structured_result = load_gpt_output(structured_file)
        if structured_result is None:
            # breakpoint()
            print(f"Error loading structured patient data for patient {patient_id}. Skipping...")
            continue
        
        gold_path = structured_result["final_path"]
        del structured_result["final_path"]

        prompt = prompt_structured_path(
            guideline_path=paths.guideline_path,
            patient_information=structured_result,
        )
        eval_prompts = [prompt]
        
        eval_results = process_prompts_with_executor(eval_prompts, timeout=100)
        
        predicted_output_path = paths.get_patient_file(patient_id, f"structured_predicted_output_patient_{patient_id}.json")
        for result in eval_results:
            save_result_to_file(result, predicted_output_path)
        
        predicted_output = load_gpt_output(predicted_output_path)
        predicted_path = predicted_output["final_path"]

        prompt_comparison = prompt_compare_paths(
            final_path_original=gold_path,
            final_path_generated=predicted_path
        )
        eval_prompts_comparison = [prompt_comparison]
        eval_results_comparison = process_prompts_with_executor(eval_prompts_comparison, timeout=100)
        
        for result in eval_results_comparison:
            save_result_to_file(result, matched_output_path)

def generate_unstructured(number_of_patients: int = 50, method: str = "structured"):
    """Generate unstructured clinical notes."""
    sorted_notes = pd.read_csv(paths.unstructured_data_path)
    sorted_notes = sorted_notes.sort_values(by='DEID_NOTE_TXT_DD48_NOTES', key=lambda x: x.str.len(), ascending=True)
    sorted_notes = sorted_notes[['DEID_NOTE_TXT_DD48_NOTES', 'PTID']].reset_index(drop=True)
        
    for patient_id in range(number_of_patients):
        note_template_text = patient_id + 50
        note_example_instance = sorted_notes.iloc[note_template_text]
        note_example = note_example_instance['DEID_NOTE_TXT_DD48_NOTES']
        
        output_path = paths.get_patient_file(patient_id, f"synthetic_unstructured_{patient_id}.txt")
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping patient {patient_id}.")
            continue
        
        if method == "structured":
            matched_outputs_file = paths.get_patient_file(patient_id, f"structured_matched_outputs_{patient_id}.json")
            if not os.path.exists(matched_outputs_file):
                print(f"FILTERED: Skipping patient {patient_id} due to missing structured matched outputs.")
                continue
            
            filtered_status = load_gpt_output(matched_outputs_file)
            if filtered_status is None:
                print(f"Error loading structured matched outputs for patient {patient_id}. Skipping...")
                continue
            
            print(f"Processing patient {patient_id} with path match: {filtered_status['path_match']}")
            if filtered_status["path_match"] == "false":
                print(f"Skipping patient {patient_id} due to path or treatment mismatch.")
                continue
            
            structured_file = paths.get_patient_file(patient_id, f"structured_patient_{patient_id}.json")
            patient_data_og = load_gpt_output(structured_file)
            if patient_data_og is None:
                print(f"Error loading structured patient data for patient {patient_id}. Skipping...")
                continue
            
            patient_data = {k: v for k, v in patient_data_og.items() if "final" not in k}
            
            prompt = prompt_structured_to_unstructured(
                patient_data=patient_data,
                clinical_note_example=note_example,
                guideline_path=paths.guideline_path,
                target_path=patient_data_og["final_path"]
            )
        elif method == "unstructured":
            path_file = paths.get_patient_file(patient_id, f"path_{patient_id}.txt")
            patient_data_og = load_gpt_output(path_file)
            try:
                prompt = prompt_generate_unstructured_patient(
                guideline_path=paths.guideline_path,
                clinical_note=note_example,
                target_path=patient_data_og["final_path"]
            )
            except:
                print(f"Error generating unstructured patient data for patient {patient_id}. Skipping...")
                continue
        
        eval_prompts = [prompt]
        eval_results = process_prompts_with_executor(eval_prompts)
        
        for result in eval_results:
            save_result_to_file(result, output_path)

def generate_prediction_from_unstructured(number_of_patients: int = 50, method: str = "structured"):
    """Generate predictions from unstructured clinical notes."""
    for patient_id in range(number_of_patients):
        output_path = paths.get_patient_file(patient_id, f"unstructured_predicted_output_patient_{patient_id}.json")
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping patient {patient_id}.")
            continue
        
        if method == "structured":
            matched_outputs_file = paths.get_patient_file(patient_id, f"structured_matched_outputs_{patient_id}.json")
            if not os.path.exists(matched_outputs_file):
                print(f"FILTERED: Skipping patient {patient_id} due to missing structured matched outputs.")
                continue
            
            filtered_status = load_gpt_output(matched_outputs_file)
            print(f"Processing patient {patient_id} with path match: {filtered_status['path_match']}")
            if filtered_status["path_match"] == "false":
                print(f"Skipping patient {patient_id} due to path or treatment mismatch.")
                continue
        
        clinical_note_file = paths.get_patient_file(patient_id, f"synthetic_unstructured_{patient_id}.txt")
        with open(clinical_note_file, "r", encoding="utf-8") as f:
            clinical_note = f.read()
        
        prompt = prompt_unstructured_to_path(
            guideline_path=paths.guideline_path,
            clinical_note=clinical_note
        )
        eval_prompts = [prompt]

        eval_results = process_prompts_with_executor(eval_prompts)
        
        for result in eval_results:
            save_result_to_file(result, output_path)

def analyze_unstructured_response(method: str, number_of_patients: int = 50):
    """Analyze unstructured response accuracy."""
    llm_selector = configure_azure_llm(
        deployment_name="o3-mini",
        api_version=API_VERSION,
        api_base=API_BASE,
    )
    
    for patient_id in range(number_of_patients):
        output_path = paths.get_patient_file(patient_id, f"unstructured_matched_outputs_{patient_id}.json")
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping patient {patient_id}.")
            continue
        
        if method == "structured":
            matched_outputs_file = paths.get_patient_file(patient_id, f"structured_matched_outputs_{patient_id}.json")
            if not os.path.exists(matched_outputs_file):
                print(f"FILTERED: Skipping patient {patient_id} due to missing structured matched outputs.")
                continue
            
            filtered_status = load_gpt_output(matched_outputs_file)
            print(f"Processing patient {patient_id} with path match: {filtered_status['path_match']}")
            if filtered_status["path_match"] == "false":
                print(f"Skipping patient {patient_id} due to path or treatment mismatch.")
                continue
            
            structured_file = paths.get_patient_file(patient_id, f"structured_patient_{patient_id}.json")
            gold_path_load_json = load_gpt_output(structured_file)
            if gold_path_load_json is None:
                print(f"Error loading gold path for patient {patient_id}. Skipping...")
                continue
            gold_path_load = gold_path_load_json["final_path"]
        else:
            path_file = paths.get_patient_file(patient_id, f"path_{patient_id}.txt")
            gold_path_load = load_gpt_output(path_file)["final_path"]
        
        clinical_note_file = paths.get_patient_file(patient_id, f"synthetic_unstructured_{patient_id}.txt")
        with open(clinical_note_file, "r", encoding="utf-8") as f:
            clinical_note = f.read()
        
        predicted_output_file = paths.get_patient_file(patient_id, f"unstructured_predicted_output_patient_{patient_id}.json")
        predicted_output = load_gpt_output(predicted_output_file)
        predicted_path = predicted_output["final_path"]
        gold_path = gold_path_load

        prompt_compare_dir = prompt_compare_paths(
            final_path_original=gold_path,
            final_path_generated=predicted_path
        )
        
        eval_prompts_compare_paths = [prompt_compare_dir]
        eval_results_compare_paths = process_prompts_with_executor(eval_prompts_compare_paths, timeout=100)
        
        for result in eval_results_compare_paths:
            save_result_to_file(result, output_path)
        
        paths_matched = load_gpt_output(output_path)["path_match"]
        if paths_matched == "true":
            print(f"Paths matched for patient {patient_id}, no need for unstructured analysis.")
            continue
        
        # First analysis
        prompt_comparison_1 = prompt_compare_unstructured_1(
            guideline_path=paths.guideline_path,
            clinical_note=clinical_note,
            gt_path=gold_path,
            predicted_path=predicted_path
        )
        eval_prompts_comparison_1 = [prompt_comparison_1]
        eval_results_comparison_1 = MultithreadedExecutor(
            llm_selector, List, timeout=100, max_retries=2
        ).process_prompts(eval_prompts_comparison_1, n_threads=30)
        
        analysis_1_path = paths.get_patient_file(patient_id, f"unstructured_analysis_{patient_id}_1.json")
        for result in eval_results_comparison_1:
            save_result_to_file(result, analysis_1_path)
        
        # Second analysis
        prompt_comparison_2 = prompt_compare_unstructured_2(
            guideline_path=paths.guideline_path,
            clinical_note=clinical_note,
            gt_path=gold_path,
            predicted_path=predicted_path
        )
        eval_prompts_comparison_2 = [prompt_comparison_2]
        eval_results_comparison_2 = MultithreadedExecutor(
            llm_selector, List, timeout=100, max_retries=2
        ).process_prompts(eval_prompts_comparison_2, n_threads=30)

        analysis_2_path = paths.get_patient_file(patient_id, f"unstructured_analysis_{patient_id}_2.json")
        for result in eval_results_comparison_2:
            save_result_to_file(result, analysis_2_path)

def return_results(number_of_patients: int = 50, method: str = "structured"):
    """Return analysis results and patient IDs to keep."""
    correct = 0
    total = 0
    total_filtered_1 = 0
    total_filtered_2 = 0
    ids_to_keep = []
    
    for patient_id in range(number_of_patients):
        print(f"Processing patient {patient_id} for final analysis...")
        
        # if method == "structured":
        matched_outputs_file = paths.get_patient_file(patient_id, f"unstructured_matched_outputs_{patient_id}.json")
        if not os.path.exists(matched_outputs_file):
            print(f"FILTERED 1: Skipping patient {patient_id} due to missing unstructured matched outputs.")
            total_filtered_1 += 1
            continue
    
        matched_outputs = load_gpt_output(matched_outputs_file)
        if matched_outputs["path_match"] == "true":
            print(f"CORRECT: Patient {patient_id} has path match.")
            correct += 1
            total += 1
            ids_to_keep.append(patient_id)
            continue
        
        analysis_1_file = paths.get_patient_file(patient_id, f"unstructured_analysis_{patient_id}_1.json")
        analysis_2_file = paths.get_patient_file(patient_id, f"unstructured_analysis_{patient_id}_2.json")
        
        with open(analysis_1_file, "r", encoding="utf-8") as f:
            status_1 = f.read()
        with open(analysis_2_file, "r", encoding="utf-8") as f:
            status_2 = f.read()
        
        if "path 1" in status_1.lower() and "path 2" in status_2.lower():
            total += 1
            ids_to_keep.append(patient_id)
            print(f"INCORRECT: Patient {patient_id} has unstructured analysis 1 and 2 mismatch.")
        else:
            print(f"FILTERED 2: Skipping patient {patient_id} due to unstructured analysis mismatch.")
            total_filtered_2 += 1

    print(f"Total patients: {total}")
    print(f"Correctly matched patients: {correct}")
    print(f"Total filtered patients (structured analysis 1): {total_filtered_1}")
    print(f"Total filtered patients (unstructured analysis 2): {total_filtered_2}")
    return ids_to_keep

def move_synthetic_set(ids: List[int], destination_dir: str = None):
    """Copy patient notes to filtered directory."""
    for patient_id in ids:
        
        note_src = paths.get_patient_file(patient_id, f"synthetic_unstructured_{patient_id}.txt")
        path_src = paths.get_patient_file(patient_id, f"path_{patient_id}.txt")
        patient_dest = os.path.join(destination_dir, f"patient_{patient_id}")
        
        if os.path.exists(paths.get_filtered_patient_dir(patient_id)):
            print(f"Destination directory already exists. Skipping copy for patient {patient_id}.")
            continue

        with open(note_src, "r", encoding="utf-8") as f:
            note_content = f.read()
        
        path = load_gpt_output(path_src)
        if "final_path" in path:
            path = path["final_path"]
        datapoint = {"patient_note": note_content, "label": path}
        os.makedirs(patient_dest, exist_ok=True)
        with open(os.path.join(patient_dest, f"patient_{patient_id}.json"), "w", encoding="utf-8") as f:
            json.dump(datapoint, f)
        print(f"Copied patient data for patient {patient_id}.")

if __name__ == "__main__":
    generation_method = "structured"  # Change to "unstructured" if needed
    n = 250
    # Initialize path configuration - CHANGE BASE PATH HERE IF NEEDED
    paths = PathConfig(base_path = BASE_PATH, method=generation_method)

    generate_path(n)
    print(f"Generated paths for {n} patients.")
    
    if generation_method == "structured":
        generate_structured_data(n)
        print(f"Generated structured data for {n} patients.")
        generate_treatment_from_structured(n)
        print(f"Generated treatment from structured data for {n} patients.")
    
    generate_unstructured(n, method=generation_method)
    generate_prediction_from_unstructured(n, method=generation_method)
    print(f"Generated predictions from unstructured data for {n} patients.")
    
    analyze_unstructured_response(generation_method, n)
    print(f"Analyzed unstructured responses for {n} patients.")
    
    ids_to_keep = return_results(n, method=generation_method)
    move_synthetic_set(ids_to_keep, destination_dir=paths.filtered_patients_dir)
    print(f"Returned results for {n} patients.")