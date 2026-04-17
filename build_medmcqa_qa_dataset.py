import pandas as pd
import json
import os

data_dir = "medmcqa"                     # Directory containing the MedMCQA files
output_json = "medmcqa_qa.json"          # Output JSON file for MedMCQA only
required_columns = {"question", "opa", "opb", "opc", "opd", "cop", "subject_name", "topic_name"}

# Load train, validation, and test splits
dfs = []

try:
    for filename in ["train.csv", "validation.csv", "test.csv"]:
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path, encoding="utf-8")
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"Missing columns in {filename}: {sorted(missing_columns)}")

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    results = []
    skipped_rows = 0

    for _, row in df_all.iterrows():
        try:
            question = row["question"]
            choices = [
                row["opa"],
                row["opb"],
                row["opc"],
                row["opd"],
            ]

            cop = int(row["cop"])
            if cop < 1 or cop > 4:
                raise ValueError(f"Invalid cop value: {cop}")

            answer = choices[cop - 1]

            results.append({
                "q": question,
                "a": answer,
                "source": "MedMCQA",
                "type": "MCQ",
                "exp": str(row.get("exp", "")),
                "subject": row["subject_name"],
                "topic": row["topic_name"],
            })

        except Exception:
            skipped_rows += 1
            continue

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"MedMCQA QA data has been converted to {output_json}. Total records: {len(results)}.")
    print(f"Skipped rows: {skipped_rows}.")

except FileNotFoundError as err:
    print(f"File not found: {err}")
except pd.errors.EmptyDataError:
    print("One or more input CSV files are empty.")
except ValueError as err:
    print(f"Data validation error: {err}")
except Exception as err:
    print(f"Unexpected error: {err}")