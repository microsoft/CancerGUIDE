import json
import json5
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import os

from sam_mtb_utils.common import configure_azure_llm
from sam_mtb_utils.factuality import MultithreadedExecutor

from prompts import prompt_unstructured_to_path

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
import argparse

from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
    
BASE_PATH = Path("../CancerGUIDE_Internal")

# Configuration
class Settings:
    """Centralized configuration settings."""

    def __init__(self, args: argparse.Namespace):
        # API Configuration
        self.API_BASE = "https://rwe-aoai.openai.azure.com"
        self.API_VERSION = "2024-12-01-preview"
        self.DEPLOYMENT_NAME = args.model

        # File Paths
        self.BASE_PATH = BASE_PATH
        self.GUIDELINE_PATH = self.BASE_PATH / "data" / "nsclsc_guideline.json"
        self.RESULTS_PATH = self.BASE_PATH / "results" / f"rollout_results/rollout_experiment_{self.DEPLOYMENT_NAME}"
        self.ANNOTATIONS_PATH = self.BASE_PATH / "data" / "benchmarks/human_annotations"

        # Processing Parameters
        self.START_INDEX = 100
        self.END_INDEX = 122
        self.NUM_ITERATIONS = 10
        self.MAX_THREADS = 30
        self.TIMEOUT = 400
        self.MAX_RETRIES = 3
        self.HUMAN_ANNOTATIONS = 360

        # Benchmark Settings
        self.CONSTRUCT_BENCHMARK = True
        self.BENCHMARK_THRESHOLD = .9
        self.BENCHMARK_OUTPUT_DIR = self.BASE_PATH / "data" / f"benchmarks/consistency_bench/{self.DEPLOYMENT_NAME}"
        os.makedirs(self.BENCHMARK_OUTPUT_DIR, exist_ok=True)

def filter_path_17_node(lst):
    if lst:
        if lst[-1]=="NSCL-17-10" and len(lst) > 1:
            lst=lst[:-1]
    if lst:
        if lst[-1]=="NSCL-17-1" and len(lst) > 1:
            lst=lst[:-1]
    return lst

# Core Data Classes
class PatientData:
    """Handles patient data loading and management."""

    def __init__(self, annotations_path: Path, settings: Settings):
        self.annotations_path = annotations_path
        self.settings = settings
        self._validate_path()
    
    def _validate_path(self):
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_path}")
    
    def load_ground_truth(self) -> Dict[int, str]:
        """Load ground truth labels for patients."""
        ground_truth = {}
        
        for patient_id in range(self.settings.HUMAN_ANNOTATIONS):
            file_path = self.annotations_path / f"patient_{patient_id}.json"
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'label' in data:
                        ground_truth[patient_id] = data['label']
            except Exception as e:
                print(f"Error loading ground truth for patient {patient_id}: {e}")
        
        return ground_truth
    
    def load_clinical_notes(self) -> Dict[int, str]:
        """Load clinical notes for patients."""
        clinical_notes = {}

        for patient_id in range(self.settings.HUMAN_ANNOTATIONS):
            file_path = self.annotations_path / f"patient_{patient_id}.json"
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'patient_note' in data:
                        clinical_notes[patient_id] = data['patient_note']
            except Exception as e:
                print(f"Error loading clinical note for patient {patient_id}: {e}")
        
        return clinical_notes


class FileManager:
    """Handles file operations and validation."""
    
    @staticmethod
    def is_valid_json_file(file_path: Path) -> bool:
        """Check if file exists and contains valid JSON content."""
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return False
                return FileManager.extract_json(content) is not None
        except Exception:
            return False
    
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict]:
        """
        Extract and parse JSON from text.
        - If <think>...</think> exists, only consider JSON after the last </think>.
        - Otherwise, fall back to scanning the whole text for JSON.
        Handles both fenced code blocks and loose JSON.
        """
        after_think = None

        if "</think>" in text:
            # Cut everything before the last </think>
            after_think = text.split("</think>")[-1]

        # Use either the portion after </think> or the full text
        search_region = after_think if after_think is not None else text

        # Try fenced ```json blocks
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", search_region, re.DOTALL)
        if not match:
            # Try loose { ... } block
            match = re.search(r"({.*?})", search_region, re.DOTALL)

        if not match:
            return None

        try:
            return json5.loads(match.group(1))
        except Exception:
            return None
    
    @staticmethod
    def ensure_dir(path: Path):
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)


# Core Analysis Classes
class PathwayPredictor:
    """Generates pathway predictions using LLM."""

    def __init__(self, guideline_path: Path, settings: Settings):
        self.guideline_path = guideline_path
        self.settings = settings
        self.llm = self._setup_llm()
        self.executor = MultithreadedExecutor(
            self.llm, List, 
            timeout=self.settings.TIMEOUT, 
            max_retries=self.settings.MAX_RETRIES
        )
    
    def _setup_llm(self):
        """Configure the Azure LLM client."""
        if self.settings.DEPLOYMENT_NAME == "gpt-5-med":
            return configure_azure_llm(
            deployment_name="gpt-5",
            api_version=self.settings.API_VERSION,
            api_base=self.settings.API_BASE
        )
        if self.settings.DEPLOYMENT_NAME == "gpt-5":
            return configure_azure_llm(
            deployment_name="gpt-5",
            api_version=self.settings.API_VERSION,
            api_base=self.settings.API_BASE,
            reasoning_effort="minimal"
        )
        if self.settings.DEPLOYMENT_NAME == "gpt-5-high":
            return configure_azure_llm(
            deployment_name="gpt-5",
            api_version=self.settings.API_VERSION,
            api_base=self.settings.API_BASE,
            reasoning_effort="high"
        )
        return configure_azure_llm(
            deployment_name=self.settings.DEPLOYMENT_NAME,
            api_version=self.settings.API_VERSION,
            api_base=self.settings.API_BASE,
        )
    
    def deepseek_complete(self, prompt):
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

        return response.choices[0].message.content
    def llama_complete(self, prompt):
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

    def predict_pathway(self, clinical_note: str) -> str:
        """Generate a single pathway prediction."""
        prompt = prompt_unstructured_to_path(str(self.guideline_path), clinical_note)
        if self.settings.DEPLOYMENT_NAME == "deepseek":
            results=self.deepseek_complete(prompt)
            return results
        if self.settings.DEPLOYMENT_NAME == "llama":
            results=self.llama_complete(prompt)
            return results
        print("starting generation  ...")
        results = self.executor.process_prompts([prompt], n_threads=self.settings.MAX_THREADS)
        print("generation complete")
        return results[0].parsed_output
    
    def generate_multiple_predictions(self, clinical_note: str, patient_id: int, results_dir: Path):
        """Generate multiple predictions for a patient across iterations."""
        patient_dir = results_dir / f"patient_{patient_id}"
        FileManager.ensure_dir(patient_dir)
        
        for iteration in range(self.settings.NUM_ITERATIONS):
            output_file = patient_dir / f"patient_{patient_id}_iteration_{iteration}.json"
            
            # if FileManager.is_valid_json_file(output_file):
            if output_file.exists():
                print(f"  Skipping patient {patient_id}, iteration {iteration} - already exists")
                continue
            # breakpoint()
            print(f"  Generating prediction for patient {patient_id}, iteration {iteration}")
            if patient_id<135:
                print("debug and skip 75")
                continue
            prediction = self.predict_pathway(clinical_note)
            if not prediction:
                print("ERROR IN GENERATION")
                continue

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(prediction)


class ConsistencyAnalyzer:
    """Analyzes consistency across multiple pathway predictions."""

    def __init__(self, guideline_data: Dict, settings: Settings):
        self.guideline_data = guideline_data
        self.settings = settings
        self.llm = self._setup_llm()
        self.executor = MultithreadedExecutor(
            self.llm, List,
            timeout=self.settings.TIMEOUT,
            max_retries=self.settings.MAX_RETRIES
        )
        self.patient_data = PatientData(self.settings.ANNOTATIONS_PATH, self.settings)
        self.clinical_notes = self.patient_data.load_clinical_notes()
    
    def _setup_llm(self):
        """Configure the Azure LLM client."""
        return configure_azure_llm(
            deployment_name=self.settings.DEPLOYMENT_NAME,
            api_version=self.settings.API_VERSION,
            api_base=self.settings.API_BASE,
        )
    
    def load_patient_predictions(self, patient_id: int, results_dir: Path) -> List[str]:
        """Load all pathway predictions for a patient."""
        predictions = []
        
        for iteration in range(self.settings.NUM_ITERATIONS):
            file_path = results_dir / f"patient_{patient_id}" / f"patient_{patient_id}_iteration_{iteration}.json"
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                data = FileManager.extract_json(text)
                if data and 'reason' in data and data['reason'] and 'NOT_GUIDELINE_COMPLIANT' in data['reason']:
                    data['final_path'] = 'NOT_GUIDELINE_COMPLIANT'
                
                if data and 'final_path' in data:
                    predictions.append(data['final_path'])
                else:
                    predictions.append('')
                    
            except FileNotFoundError:
                predictions.append('')
        
        return predictions
    
    def compare_lists(self, lists, return_final_consistency=True):
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

    
    def compare_pathways(self, pathways):
        pathways_clean = []
        for model_prediction_path in pathways:
            pathways_clean.append([p.strip() for p in re.split(r'\s*->\s*|\s*>\s*|\s*→\s*', model_prediction_path)])
        path_match_fraction, treatment_match_fraction, path_mode, treatment_mode = self.compare_lists(pathways_clean)
        return  path_match_fraction, treatment_match_fraction, path_mode, treatment_mode
    
    def analyze_patient_consistency(self, patient_id: int, results_dir: Path):
        """Analyze consistency for a single patient."""
        patient_dir = results_dir / f"patient_{patient_id}"
        comparison_file = patient_dir / f"matched_outputs_{patient_id}_k_{self.settings.NUM_ITERATIONS}.json"
        
        if FileManager.is_valid_json_file(comparison_file):
            print(f"  Skipping consistency analysis for patient {patient_id} - already exists")
            return
        
        pathways = self.load_patient_predictions(patient_id, results_dir)
        
        if len(pathways) < self.settings.NUM_ITERATIONS or all(not p.strip() for p in pathways):
            print(f"  Insufficient data for patient {patient_id}")
            return
        
        print(f"  Analyzing consistency for patient {patient_id}")
        path_match_fraction, treatment_match_fraction, path_mode, treatment_mode = self.compare_pathways(pathways)
        results = {"final_path_match": path_match_fraction, "final_treatment_score": treatment_match_fraction,
                   "final_path_mode": path_mode, "final_treatment_mode": treatment_mode}

        FileManager.ensure_dir(patient_dir)
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        if path_match_fraction>=self.settings.BENCHMARK_THRESHOLD and self.settings.CONSTRUCT_BENCHMARK:
            datapoint={}
            clinical_note = self.clinical_notes[patient_id]
            label = path_mode

            # Populate the datapoint
            datapoint["patient_note"] = clinical_note
            datapoint["label"] = label

            # Save the datapoint
            output_location = self.settings.BENCHMARK_OUTPUT_DIR / "path_filter"
            os.makedirs(output_location, exist_ok=True)
            with open(output_location / f"patient_{patient_id}.json", 'w', encoding='utf-8') as f:
                json.dump(datapoint, f, indent=4)
        if treatment_match_fraction>=self.settings.BENCHMARK_THRESHOLD and self.settings.CONSTRUCT_BENCHMARK:
            datapoint={}
            clinical_note = self.clinical_notes[patient_id]
            label = path_mode

            # Populate the datapoint
            datapoint["patient_note"] = clinical_note
            datapoint["label"] = label

            # Save the datapoint
            output_location = self.settings.BENCHMARK_OUTPUT_DIR / "treatment_filter"
            os.makedirs(output_location, exist_ok=True)
            with open(output_location / f"patient_{patient_id}.json", 'w', encoding='utf-8') as f:
                json.dump(datapoint, f, indent=4)
        print(f"  Consistency analysis for patient {patient_id} complete: {path_match_fraction:.2f} path match, {treatment_match_fraction:.2f} treatment match")


# Main Analysis Pipeline
class ClinicalPathwayAnalyzer:
    """Main orchestrator for the clinical pathway analysis."""

    def __init__(self, args):
        self.settings = Settings(args)
        self.patient_data = PatientData(self.settings.ANNOTATIONS_PATH, self.settings)
        self.predictor = PathwayPredictor(self.settings.GUIDELINE_PATH, self.settings)
        
        # Load guideline data for consistency analysis
        with open(self.settings.GUIDELINE_PATH, 'r') as f:
            guideline_data = json.load(f)
        self.consistency_analyzer = ConsistencyAnalyzer(guideline_data, self.settings)

        # Load ground truth and determine patients to analyze
        self.ground_truth = self.patient_data.load_ground_truth()
        if self.settings.CONSTRUCT_BENCHMARK:
            self.patient_ids = range(self.settings.HUMAN_ANNOTATIONS)
        else:
            self.patient_ids = list(self.ground_truth.keys())[
                self.settings.START_INDEX:self.settings.END_INDEX
            ]

    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting clinical pathway analysis...")
        self._print_status()
        
        # Step 1: Generate pathway predictions
        print("\n=== Step 1: Generating Pathway Predictions ===")
        self._generate_predictions()
        
        # Step 2: Analyze consistency
        print("\n=== Step 2: Analyzing Pathway Consistency ===")
        self._analyze_consistency()
        
        
        # Step 4: Generate summary report
        print("\n=== Step 4: Generating Summary Report ===")
        self._generate_report()
        
        print("\nAnalysis complete!")
    
    def _generate_predictions(self):
        """Generate pathway predictions for all patients."""
        clinical_notes = self.patient_data.load_clinical_notes()
        
        for patient_id in self.patient_ids:
            if patient_id not in clinical_notes:
                print(f"No clinical note found for patient {patient_id}")
                continue
            
            print(f"Processing patient {patient_id}...")
            self.predictor.generate_multiple_predictions(
                clinical_notes[patient_id], patient_id, self.settings.RESULTS_PATH
            )
    
    def _analyze_consistency(self):
        """Analyze consistency across iterations for all patients."""
        for patient_id in self.patient_ids:
            self.consistency_analyzer.analyze_patient_consistency(
                patient_id, self.settings.RESULTS_PATH
            )
    
    def _generate_report(self):
        """Generate final summary report."""
        matches = 0
        total = 0
        errors = []
        
        for patient_id in self.patient_ids:
            file_path = (self.settings.RESULTS_PATH / f"patient_{patient_id}" / 
                        f"matched_outputs_{patient_id}_k_{self.settings.NUM_ITERATIONS}.json")
            
            if not FileManager.is_valid_json_file(file_path):
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                result = FileManager.extract_json(f.read())
            
            if not result:
                continue
            
            total += 1
            if result.get('final_path_match', False):
                matches += 1
            else:
                error = result.get('error_analysis', 'Unknown error')
                errors.append(f"Patient {patient_id}: {error}")
        
        print(f"\n=== SUMMARY REPORT ===")
        if total > 0:
            print(f"Patients analyzed: {total}")
            print(f"Consistent predictions: {matches}")
            print(f"Consistency rate: {matches/total*100:.1f}%")
            
            if errors:
                print(f"\nInconsistencies found in {len(errors)} patients:")
                for error in errors[:10]:  # Show first 10 errors
                    print(f"  - {error}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more")
        else:
            print("No valid data found for analysis")
    
    def _print_status(self):
        """Print current processing status."""
        print(f"\n=== PROCESSING STATUS ===")
        print(f"Patient range: {self.settings.START_INDEX}-{self.settings.END_INDEX}")
        print(f"Total patients: {len(self.patient_ids)}")
        print(f"Iterations per patient: {self.settings.NUM_ITERATIONS}")
        print(f"Results directory: {self.settings.RESULTS_PATH}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run clinical pathway analysis")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="Model to use for analysis (e.g., gpt-4.1, gpt-5, gpt-5-med, deepseek)"
    )
    args = parser.parse_args()

    analyzer = ClinicalPathwayAnalyzer(args)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()