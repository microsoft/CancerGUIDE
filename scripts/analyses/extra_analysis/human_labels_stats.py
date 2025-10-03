import json
from pathlib import Path
import re
from collections import Counter

# Path to the folder
folder_path = Path("/home/azureuser/cloudfiles/code/rwep_experiments/alyssa/RLFollow_clean/data/benchmarks/human_annotations")

# Initialize counters
total_chars = 0
num_notes = 0
unique_paths = set()
unique_final_treatments = set()

# Regex to match NSCL-17-anynumber
nscl_pattern = re.compile(r'NSCL-17-\d+')

# Iterate through all files in the folder
for file_path in folder_path.iterdir():
    if file_path.suffix in ['.json', '.jsonl']:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue  # skip files that cannot be parsed

            note = data.get("patient_note", "")
            total_chars += len(note)
            num_notes += 1

            # Extract label
            label = data.get("label", "")
            if not label:
                continue

            # Track unique full paths
            unique_paths.add(label)

            # Process final treatment
            parts = label.split("->")
            # Handle NSCL-17 rule
            if parts[-1].startswith("NSCL-17"):
                # find last element before the NSCL-17 pattern
                for i in range(len(parts)-2, -1, -1):
                    if not nscl_pattern.match(parts[i]):
                        final = parts[i]
                        break
                else:
                    final = parts[-1]
            else:
                final = parts[-1]
            unique_final_treatments.add(final)

# Compute average character count
average_chars = total_chars / num_notes if num_notes > 0 else 0

print("Average patient note length:", average_chars)
print("Unique label paths:", len(unique_paths))
print("Unique final treatments:", len(unique_final_treatments))
