import json
import os
import re
from collections import defaultdict


def load_medredqa_dir(data_dir):
    """
    Load the three MedRedQA JSON files from a directory and convert them into
    a unified format: {"q": "...", "a": "...", **other_fields}
    """
    all_data = []
    required_fields = {"question", "response"}

    # NEW: Production cleaning variables
    seen_questions = set()  # Duplicate tracking
    MIN_LENGTH = 10  # Minimum content length
    skipped_stats = defaultdict(int)  # Cleaning statistics

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
                    skipped_stats["invalid_format"] += 1
                    continue  # CHANGED: use continue instead of raise

                missing_fields = required_fields - set(item.keys())
                if missing_fields:
                    skipped_stats["missing_fields"] += 1
                    continue  # CHANGED: use continue instead of raise

                # NEW: Extract and validate content
                question = str(item["question"]).strip()
                answer = str(item["response"]).strip()

                # NEW: Empty content filtering
                if not question or not answer:
                    skipped_stats["empty_content"] += 1
                    continue

                # NEW: Length validation
                if len(question) < MIN_LENGTH or len(answer) < MIN_LENGTH:
                    skipped_stats["too_short"] += 1
                    continue

                # NEW: Duplicate detection (hash-based)
                q_hash = hash(question.lower())
                if q_hash in seen_questions:
                    skipped_stats["duplicates"] += 1
                    continue
                seen_questions.add(q_hash)

                # NEW: Basic text normalization
                question = re.sub(r'\s+', ' ', question)  # Normalize whitespace
                answer = re.sub(r'\s+', ' ', answer)

                unified = {
                    "q": question,
                    "a": answer,
                    **{k: str(v).strip() for k, v in item.items() if k not in ["question", "response"]},
                    # IMPROVED: safe dict unpack
                }
                all_data.append(unified)

            except Exception as err:
                skipped_stats["processing_error"] += 1
                continue

    # NEW: Print cleaning statistics
    print("Cleaning statistics:")
    total_processed = sum(skipped_stats.values()) + len(all_data)
    print(f"  Total records processed: {total_processed}")
    print(f"  Clean records kept: {len(all_data)} ({len(all_data) / total_processed * 100:.1f}%)")
    for reason, count in sorted(skipped_stats.items()):
        if count > 0:
            print(f"  Skipped {reason}: {count}")

    return all_data


medredqa_dir = "medredqa"  # Directory containing the MedRedQA files
output_json = "medredqa_qa_clean.json"  # CHANGED: More descriptive output name

med_redqa = load_medredqa_dir(medredqa_dir)

with open(output_json, "w", encoding="utf-8") as file:
    json.dump(med_redqa, file, indent=2, ensure_ascii=False)

print(f"MedRedQA QA data has been converted to {output_json}. Total clean records: {len(med_redqa)}.")