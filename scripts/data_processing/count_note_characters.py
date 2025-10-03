import os
import json
import argparse

def count_characters_in_notes(directory):
    results = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    note = data.get("patient_note", "")
                    results[filepath] = len(note)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count characters in clinical_note fields of JSON files.")
    parser.add_argument("directory", type=str, help="Directory containing JSON files.")
    args = parser.parse_args()

    counts = count_characters_in_notes(args.directory)
    total=[count for count in counts.values()]
    print(f"Average characters per note: {sum(total)/len(total) if total else 0}")
