import json
import os

def load_medredqa_dir(data_dir):
    """
    Load the three MedRedQA JSON files from a directory and convert them into
    a unified format: {"q": "...", "a": "...", **other_fields}
    """
    all_data = []
    required_fields = {"question", "response"}

    for filename in [
        "medredqa_train.json",
        "medredqa_val.json",
        "medredqa_test.json",
    ]:
        path = os.path.join(data_dir, filename)

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue
        except json.JSONDecodeError as err:
            print(f"Invalid JSON in {path}: {err}")
            continue
        except Exception as err:
            print(f"Unexpected error while reading {path}: {err}")
            continue

        if not isinstance(data, list):
            print(f"Skipping {path}: expected a list of records.")
            continue

        for item in data:
            try:
                if not isinstance(item, dict):
                    raise ValueError("Each record must be a dictionary.")

                missing_fields = required_fields - set(item.keys())
                if missing_fields:
                    raise ValueError(f"Missing fields: {sorted(missing_fields)}")

                unified = {
                    "q": item["question"],
                    "a": item["response"],
                    **item,
                }
                all_data.append(unified)

            except Exception as err:
                print(f"Skipping one record in {filename}: {err}")
                continue

    return all_data


medredqa_dir = "medredqa"                # Directory containing the MedRedQA files
output_json = "medredqa_qa.json"         # Output JSON file for MedRedQA only

med_redqa = load_medredqa_dir(medredqa_dir)

with open(output_json, "w", encoding="utf-8") as file:
    json.dump(med_redqa, file, indent=2, ensure_ascii=False)

print(f"MedRedQA QA data has been converted to {output_json}. Total records: {len(med_redqa)}.")