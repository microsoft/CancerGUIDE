import json
from typing import List

def prompt_compare_paths(final_path_original: str, final_path_generated: str) -> str:
    """ Create a prompt to compare the original and generated patient paths and treatments, identifying if corruption occurred in the creation of a structured synthetic patient that 
    changes the true path."""

    prompt = f"""You are an expert in evaluating if two generated paths through guidelines are exact matches.
    
    First source:
    - path 1: {final_path_original}

    Second source:
    - path 2: {final_path_generated}

    Please evaluate the similarity of the first path compared to the second path. Return a json with keys "path_match" with values as strings "true", "false", or "partial".
    Paths are only considered partially correct if the entire path matches except if the last node is on NSCL-17, since that node is a "Surveillance" node and can be inferred from the context of the path.
    If there are differences in the paths besides the last node or 2 nodes (if the second to last node is also on NSCL-17), then the paths are considered different and you should return "false". THIS IS TRUE EVEN IF THE NODES ARE ON THE SAME NSCLC PAGE (e.g. NSCL-1-1 and NSCL-1-2 are different nodes, even if they are on the same page AND THUS THE PATH IS INCORRECT).
    If the paths are IDENTICAL, return "true".
    If the paths CONTAIN DIFFERENCES return "false"-- EVEN IF THE DIFFERENCES ARE ON THE SAME PAGE (I.E. NSCLC-7-11 VS NSCLC-7.12 IS FALSE). 
    If the paths are identical except for any nodes on page NSCL-17, return "partial". Otherwise, if the paths differ in any other way, return "false".
    """
    return prompt

def prompt_path(guideline_path: str, target_node: str) -> str:
    """Create a prompt to predict the path and treatment for a synthetic patient based on structured patient information and the NCCN guidelines."""
    with open(guideline_path, 'r') as f:
        guideline_data = json.load(f)
    prompt = f"""You are an expert in creating synthetic patient paths through clinical guidelines. Your task is to generate a valid path through the provided NCCN guideline, ensuring that the path includes nodes from the NSCLC page with ID {target_node}.
    In the final_path, include every node that the patient has gone through in the format "nsclc_id-node ID", e.g. NSCL-1-1. DO NOT SKIP ANY NODES (i.e. NSCL-10-1 must be included even if this is a parent node to a more informative node). The final treatment should be the last node in the path, which is the recommended treatment according to the guideline (e.g. "Resection + chemo OR chemoradiation") or the last node that we have information about.

    Guideline data: {guideline_data}

    Please return the final path as a json with the key "final_path" and the value as a string that describes the path taken through the guideline. 
    Ensure that the path is complete and includes all subnodes. For example, if the patient has to go through surveillance, be sure to include that as part of their path.
    However, if the patient needs to wait for a certain result before proceeding, then you should not continue populating the path and the path should end with that node as we wait for the patient results.
    Again, the final node should be WHAT THE PATIENT'S NEXT STEP IS, NOT WHAT THE PATIENT HAS ALREADY DONE AND NOT WHAT THE PATIENT SHOULD DO IN THE FUTURE.
    For example, if the patient needs a biopsy, the path should end with next steps as getting a biopsy and not continue beyond with assumptions regarding the biopsy results.
    This is a synthetic patient, so you can include any path through the guideline that is valid, but it should be a realistic path that a patient might take.

    Terminal nodes are nodes with "recommendation" in the node, indicating that the "recommendation" is the final treatment for the patient and we cannot proceed. All final nodes should either be terminal nodes or nodes that we lack information to proceed from."""
    return prompt

def prompt_path_structured(guideline_path:str, columns: List[str], target_path: str) -> str:
    """
    Create a prompt to generate a random path through the NCCN guidelines and an associated synthetic patient with given structured fields.
    """
    with open(guideline_path, 'r') as f:
        guideline_data = json.load(f)
    prompt = f"""You are an expert in generating synthetic patient data based on clinical guidelines. Your task is to create a synthetic patient that follows a target path through the NCCN guidelines.
    The synthetic patient should have the following structured fields: {columns}.

    The path that the patient should follow is: {target_path}.

    The full clinical guidelines are: {guideline_data}

    Please return a json with associated structured fields as well as the key "final_path" as the target_path. 
    Ensure the format of "final_path" is a list of nodes with subformat "nsclc_id-node ID", e.g. [NSCL-1-1, NSCL-12-1....]. 
    If you make any inferences/assumptions, please codify them in the structured data (i.e. include "operability":true if you assume patient is operable).
    Ensure that the structured data doesn't include additional information that would change the path (i.e. indicating of treatment/surgery if the path does not indicate that the patient has already received that treatment)."""
    return prompt

def prompt_structured_path(guideline_path: str, patient_information:str) -> str:
    """Create a prompt to predict the path and treatment for a synthetic patient based on structured patient information and the NCCN guidelines."""
    with open(guideline_path, 'r') as f:
        guideline_data = json.load(f)
    prompt = f"""You are an expert in navigating through the provided NCCN guideline to recommend a path and treatment for a synthetic patient. Your task is to generate the valid path through the guideline based on the provided patient information.
    Please return a json with keys: "final_path" and "final_treatment". The "final_path" should be a string that describes the path taken through the guideline, and the "final_treatment" should be a string that describes the final treatment recommendation based on the path taken.
    In the final_path, include every node that the patient has gone through in the format "nsclc_id-node ID", e.g. NSCL-1-1. DO NOT SKIP ANY NODES (i.e. NSCL-10-1 must be included even if this is a parent node to a more informative node). The final treatment should be the last node in the path, which is the recommended treatment according to the guideline (e.g. "Resection + chemo OR chemoradiation"). 
    Do not assume that the patient has started the final treatment yet, but the final treatment should be the one recommended by the guideline path as a next step. Do not assume that the patient has a certain response, only use the information provided in the patient_information.
    The patient information is: {patient_information}.
    The full clinical guidelines are: {guideline_data}

Terminal nodes are nodes with "recommendation" in the node, indicating that the "recommendation" is the final treatment for the patient and we cannot proceed. All final nodes should either be terminal nodes or nodes that we lack information to proceed from.
    """
    return prompt


def prompt_structured_to_unstructured(patient_data: dict, clinical_note_example: str, guideline_path: str, target_path:str) -> str:
    """
    Create a prompt to generate unstructured clinical notes from structured patient data.
    """
    with open(guideline_path, 'r') as f:
        guideline_data = json.load(f)
    return f"""Generate a clinical note based on the provided patient data.
Guideines: {guideline_data}
Target path: {target_path}
    Clinical note example: {clinical_note_example}
Structured clinical data: {json.dumps(patient_data, indent=2)}
Please return the clinical note as a string. Do not include the path text in the clinical note-- your objective is to generate a note that a physician would be able to read and regenerate the target path on their own.

MAKE SURE THAT THE NOTE FAITHFULLY REFLECTS THE STRUCTURED DATA AND THE TARGET PATH. DO NOT ADD ANY INFORMATION THAT IS NOT PRESENT IN THE STRUCTURED DATA OR THE TARGET PATH. DO NOT ADD ANY INFORMATION THAT IS NOT RELEVANT TO THE PATIENT'S CONDITION OR TREATMENT. 

DO NOT INCLUDE AN ASSESSMENT OR PLAN. Your objective should be that someone is able to read the note and regenerate the target path.
"""

def prompt_unstructured_to_path(guideline_path: str, clinical_note: str) -> str:
    """Create a prompt to predict the path and treatment for a synthetic patient based on unstructured clinical notes and the NCCN guidelines."""
    with open(guideline_path, 'r') as f:
        guideline_data = json.load(f)
    prompt = f"""You are an expert in navigating through the provided NCCN guideline to identify the path followed for a synthetic patient based on unstructured clinical notes. Your task is to generate the valid path through the guideline based on the provided clinical note.
    Please return a json with only the key: "final_path". The "final_path" should be a string that describes the path taken through the guideline or "NOT_GUIDELINE_COMPLIANT".
    In the final_path (if guideline compliant), include every node that the patient has gone through in the format "nsclc_id-node ID", e.g. NSCL-1-1. DO NOT SKIP ANY NODES (i.e. NSCL-10-1 must be included even if this is a parent node to a more informative node). Connect nodes with "->" to indicate the path taken through the guideline.
    The final treatment should be the last node in the path, which is the recommended treatment according to the guideline (e.g. "Resection + chemo OR chemoradiation").
    Do not assume that the patient has started the final treatment yet-- you do not need to traverse the guidelines to a node with no children nodes, as much as identify where the patient is in the path and what the next step is.
    
    Terminal nodes are nodes with "recommendation" in the node, indicating that the "recommendation" is the final treatment for the patient and we cannot proceed. All final nodes should either be terminal nodes or nodes that we lack information to proceed from. 
    If the patient has progressed through a path that is NOT GUIDELINE COMPLIANT, then you should return the "final_path" as "NOT_GUIDELINE_COMPLIANT", as well as an additional key "reason" that explains why the path is not compliant with the guidelines.
    If path is guideline compliant, you should only return the "final_path" key. 
 
    If the path is not guideline compliant, you should return both "final_path" and "reason" in json structure: final_path: NOT_GUIDELINE_COMPLIANT, reason: "explanation of why the path is not compliant"

    The clinical note is: {clinical_note}
    The full clinical guidelines are: {guideline_data}
    """
    return prompt

def prompt_compare_unstructured_1(guideline_path: str, clinical_note: str, gt_path: str, predicted_path:str) -> str:
    """Create a prompt to compare the unstructured clinical note and the path and treatment for a synthetic patient based on unstructured clinical notes and the NCCN guidelines."""
    with open(guideline_path, 'r') as f:
        guideline_data = json.load(f)
    prompt = f"""You are an expert in selecting the correct path through the provided NCCN guideline based on unstructured clinical notes. Your task is to evaluate the two paths based on the provided clinical note and the guideline.
    
    The full clinical guidelines are: {guideline_data}

    The clinical note is: {clinical_note}

    Path 1: {gt_path}

    Path 2: {predicted_path}

Please return either "path 1" or "path 2" as the correct path based on the clinical note and the guideline. ONLY RETURN THE string "PATH 1" OR  "PATH 2", DO NOT RETURN ANY ADDITIONAL TEXT.    """
    return prompt

def prompt_compare_unstructured_2(guideline_path: str, clinical_note: str, gt_path: str, predicted_path:str) -> str:
    """Create a prompt to compare the unstructured clinical note and the path and treatment for a synthetic patient based on unstructured clinical notes and the NCCN guidelines."""
    with open(guideline_path, 'r') as f:
        guideline_data = json.load(f)
    prompt = f"""You are an expert in selecting the correct path through the provided NCCN guideline based on unstructured clinical notes. Your task is to evaluate the two paths based on the provided clinical note and the guideline.
    
    The full clinical guidelines are: {guideline_data}

    The clinical note is: {clinical_note}

    Path 1: {predicted_path}

    Path 2: {gt_path}

    Please return either "path 1" or "path 2" as the correct path based on the clinical note and the guideline. ONLY RETURN THE string "PATH 1" OR  "PATH 2", DO NOT RETURN ANY ADDITIONAL TEXT.
    """
    return prompt

def prompt_generate_unstructured_patient(clinical_note, target_path, guideline_path) -> str:
    with open(guideline_path, 'r') as f:
        guidelines = json.load(f)
    """Create a prompt to generate unstructured clinical notes for a synthetic patient."""
    prompt = f"""You are an expert in generating unstructured clinical notes for synthetic patients. Your task is to create a realistic clinical note for a patient that follows the corresponding clinical pathway.
    The clinical note should match the : {clinical_note}
    The target path for the patient is: {target_path}
    The full clinical guidelines are: {guidelines}
    MAKE SURE THAT THE NOTE FAITHFULLY REFLECTS THE THE TARGET PATH. DO NOT INCLUDE AN ASSESSMENT OR PLAN. 
    Do not include the path text in the clinical note-- your objective is to generate a note that a physician would be able to read and regenerate the target path on their own.
    """
    return prompt