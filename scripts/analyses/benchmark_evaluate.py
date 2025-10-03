from collections import Counter
import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from datetime import datetime

from sam_mtb_utils.common import configure_azure_llm
from sam_mtb_utils.factuality import MultithreadedExecutor
from prompts import prompt_unstructured_to_path

import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

    # Required packages
from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
    

# === Utilities ===

def load_gpt_output(file_path: Path) -> Optional[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL) or \
                re.search(r"(\{.*?\})", text, re.DOTALL)

        if not match:
            print(f"No JSON object found in {file_path}.")
            return None

        json_str = match.group(1)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            return json.loads(json_str)

    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
        return None

def deepseek_complete(prompt):
    endpoint = "https://sc-zn-mbrg247a-swedencentral.services.ai.azure.com/models"
    model_name = "DeepSeek-R1-0528"

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(os.getenv("DEEPSEEK_KEY")),
        api_version="2024-05-01-preview"
    )
    prompt = prompt + "Please limit your thinking to 100 words and output your answer in a JSON format."

    try:
        response = client.complete(
            messages=[
                UserMessage(content=prompt),
            ],
            model=model_name,
            max_tokens=16384
        )
    except:
        return None

    return response.choices[0].message.content.split("</think>")[-1].strip()

def llama_complete(prompt):
    #Authenticate by trying az login first, then a managed identity, if one exists on the system)
    scope = "api://trapi/.default"
    credential = get_bearer_token_provider(ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    ), scope)
    
    deployment_name = "gcr-llama-33-70b-shared" #"meta-llama/Llama-3.3-70B-Instruct"
    api_version='5'
    endpoint = 'https://trapi.research.microsoft.com/gcr/shared/'
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    
    )
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ]
    )
    
    #Parse out the message and print
    response_content = response.choices[0].message.content
    return response_content

def execute_prompt(llm, prompt: str, output_path: Path, ground_truth: Optional[str] = None):
    result=None
    if llm=="deepseek":
        result=deepseek_complete(prompt)
        if not result:
            print(f"DeepSeek failed to generate a response for prompt: {prompt}")
            return
    if llm=="llama":
        result=llama_complete(prompt)
        if not result:
            print(f"Llama failed to generate a response for prompt: {prompt}")
            return
    else:
        result = MultithreadedExecutor(llm, List, timeout=300, max_retries=2).process_prompts([prompt], n_threads=30)[0]
        result=result.parsed_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
        if ground_truth:
            f.write("\n\n---\n\n")
            f.write(f"Ground Truth: {ground_truth}\n")

def compare_lists(lists, return_final_consistency=False):
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



def process_patient_json(patient_json_path: Path, llm_eval, results_dir: Path, guideline_path: str, generation):
    try:
        with open(patient_json_path, "r", encoding="utf-8") as f:
            patient_data = json.load(f)
        clinical_note = patient_data["patient_note"]
        ground_truth = patient_data["label"]
        if ground_truth == "No common path":
            print(f"Skipping patient {patient_json_path.stem} due to 'No common path' label.")
            return None, None, None, None
    except Exception as e:
        print(f"Failed to load patient JSON: {patient_json_path}, error: {e}")
        return None

    patient_id = patient_json_path.stem.replace("patient_", "")
    output_dir = results_dir / f"patient_{patient_id}"
    prediction_output_path = output_dir / f"path_prediction_{patient_id}.json"

    if not prediction_output_path.exists() or (prediction_output_path.exists() and (load_gpt_output(prediction_output_path) is None or load_gpt_output(prediction_output_path).get("final_path") is None)):
        if not generation:
            return None,None,None,None #if you don't want to generate anything
        print(f"Generating prediction for patient {patient_id} at location {prediction_output_path}")
        prompt = prompt_unstructured_to_path(guideline_path, clinical_note)
        execute_prompt(llm_eval, prompt, prediction_output_path, ground_truth)

    model_prediction = load_gpt_output(prediction_output_path)
    if model_prediction is None or model_prediction.get("final_path") is None:
        print(f"Model prediction missing or malformed for {patient_id}")
        return None, None, None, None

    model_prediction_path = model_prediction.get("final_path")
    model_prediction_path_list = [p.strip() for p in re.split(r'\s*->\s*|\s*>\s*|\s*→\s*', model_prediction_path)]

    if isinstance(ground_truth, list):
        ground_truth = '->'.join(ground_truth)
    human_prediction_path_list = [p.strip() for p in re.split(r'\s*->\s*|\s*>\s*|\s*→\s*', ground_truth)]

    if model_prediction.get("reason") and "NOT_GUIDELINE_COMPLIANT" in model_prediction.get("reason"):
        model_prediction_path_list = ["NOT_GUIDELINE_COMPLIANT"]

    partial_match = compare_lists([human_prediction_path_list, model_prediction_path_list])
    return patient_id, partial_match, human_prediction_path_list, model_prediction_path_list



def filter_path_17_node(lst):
    if lst:
        if lst[-1]=="NSCL-17-10" and len(lst) > 1:
            lst=lst[:-1]
    if lst:
        if lst[-1]=="NSCL-17-1" and len(lst) > 1:
            lst=lst[:-1]
    return lst

# === Main Pipeline ===

def generate_final_prediction(model_to_eval: str, results_json: Path, benchmark_dir: Path, guideline_path: str, output_dir:Path, benchmark_experiment_name:str, generation):
    API_BASE = "https://rwe-aoai.openai.azure.com"
    API_VERSION = "2024-12-01-preview"

    llm_eval = configure_azure_llm(model_to_eval, API_VERSION, API_BASE)
    if model_to_eval == "gpt-5":
        llm_eval = configure_azure_llm(model_to_eval, API_VERSION, API_BASE, reasoning_effort="minimal")
    if model_to_eval == "gpt-5-high":
        llm_eval = configure_azure_llm("gpt-5", API_VERSION, API_BASE, reasoning_effort="high")
    if model_to_eval == "gpt-5-med":
        llm_eval = configure_azure_llm("gpt-5", API_VERSION, API_BASE)
    if model_to_eval=="deepseek":
        llm_eval = "deepseek"
    if model_to_eval=="llama":
        llm_eval = "llama"

    results_dir = output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    partial_matched, treatment_matched = [], []
    guideline_compliance_correct=0
    guideline_compliance_false_negative=0
    guideline_compliance_false_positive=0

    for patient_json_file in Path(benchmark_dir).rglob("patient_*.json"):
        if not patient_json_file.is_file():
            continue

        processed_patient = process_patient_json(
            patient_json_file, llm_eval, results_dir, guideline_path, generation
        )
        if not processed_patient:
            print("Skipping patient due to processing issue")
            # breakpoint()
            continue
        patient_id, partial_match,  human_path, model_path = processed_patient
        if patient_id is None:
            continue

        partial_matched.append(partial_match)
        human_path = filter_path_17_node(human_path)
        model_path = filter_path_17_node(model_path)
        treatment_match_score = human_path[-1] == model_path[-1]
        treatment_matched.append(treatment_match_score)
        if "NOT_GUIDELINE_COMPLIANT" in human_path:
            if "NOT_GUIDELINE_COMPLIANT" in model_path:
                guideline_compliance_correct += 1
            else:
                guideline_compliance_false_negative += 1
        elif "NOT_GUIDELINE_COMPLIANT" in model_path:
            guideline_compliance_false_positive += 1
    avg_path = np.mean(partial_matched) if partial_matched else 0
    se_path = np.std(partial_matched) / np.sqrt(len(partial_matched)) if partial_matched else 0

    avg_treat = np.mean(treatment_matched) if treatment_matched else 0
    se_treat = np.std(treatment_matched) / np.sqrt(len(treatment_matched)) if treatment_matched else 0

    summary = {
        "model": model_to_eval,
        "average_path_match": avg_path,
        "se_path_match": se_path,
        "average_treatment_match": avg_treat,
        "se_treatment_match": se_treat,
        "total_patients_matched": len(treatment_matched),
        "benchmark": benchmark_experiment_name,
        "guideline_compliance_correct": guideline_compliance_correct,
        "guideline_compliance_false_negative": guideline_compliance_false_negative,
        "guideline_compliance_false_positive": guideline_compliance_false_positive
    }

    # --- Append if exists ---
    all_results = []
    if results_json.exists():
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results = data
                else:
                    # handle old single-result format
                    all_results = [data]
        except Exception as e:
            print(f"Warning: could not read existing {results_json}, starting fresh. Error: {e}")

    all_results.append(summary)

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n✅ Results appended to {results_json}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against benchmark data.")
    parser.add_argument("--model", required=True, help="Model to evaluate (e.g., gpt-5-med)")
    parser.add_argument("--results_json", default="../CancerGUIDE_Internal/results/evaluation_results.json", type=Path, help="Path to output results.json")
    parser.add_argument("--benchmark_dir", default="../CancerGUIDE_Internal/data/benchmarks/synthetic_bench/synthetic_final_unstructured", type=Path)
    parser.add_argument("--guideline_path", default="../CancerGUIDE_Internal/data/nsclsc_guideline.json")
    parser.add_argument("--output_dir", default="../CancerGUIDE_Internal/results/benchmark_results", type=Path)
    parser.add_argument("--benchmark_experiment", default="unstructured", type=str, help="Subdirectory name for the benchmark experiment")
    parser.add_argument("--generation", action="store_true", help="Whether to generate new predictions or only evaluate existing ones")
    args = parser.parse_args()

    generate_final_prediction(
        model_to_eval=args.model,
        results_json=args.results_json,
        benchmark_dir=args.benchmark_dir,
        guideline_path=args.guideline_path,
        output_dir=args.output_dir,
        benchmark_experiment_name=args.benchmark_experiment,
        generation=args.generation
    )
