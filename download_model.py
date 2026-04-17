import requests
from pathlib import Path

# Set the Hugging Face model repository and base download URL
repo_id = "sentence-transformers/all-MiniLM-L6-v2"
base_url = f"https://huggingface.co/{repo_id}/resolve/main"

file_names = [
    "config.json",
    "modules.json",
    "pytorch_model.bin",          # If this returns 404, fall back to model.safetensors
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]

# Local model directory
model_dir = Path("model") / "all-MiniLM-L6-v2"
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Preparing downloads in: {model_dir}")
print()


def download_single_file(url, local_path: Path, timeout=(10, 30)):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")

    try:
        response = requests.get(url, stream=True, timeout=timeout)

        # If pytorch_model.bin is unavailable, try model.safetensors instead
        if response.status_code == 404:
            print(f"{url} not found (404).")
            if url.endswith("/pytorch_model.bin"):
                fallback_url = url.replace("/pytorch_model.bin", "/model.safetensors")
                print(f"Trying fallback URL: {fallback_url}")
                response = requests.get(fallback_url, stream=True, timeout=timeout)
                if response.status_code == 200:
                    url = fallback_url
                    print("Downloaded model.safetensors instead.")
                else:
                    print("model.safetensors is also not available. Skipping this file.")
                    return

        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0
        chunk_size = 8192

        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        percent = (downloaded_size / total_size) * 100
                        print(f"Progress: {percent:6.2f}%", end="\r")

        print("Download completed.\n")

    except requests.exceptions.Timeout:
        print(f"Timeout occurred while downloading: {url}\n")
    except requests.exceptions.RequestException as err:
        print(f"Request error while downloading {url}: {err}\n")
    except OSError as err:
        print(f"File system error while saving {local_path}: {err}\n")
    except Exception as err:
        print(f"Unexpected error while downloading {url}: {err}\n")


# Download each file one by one
for file_name in file_names:
    file_url = f"{base_url}/{file_name}"
    local_path = model_dir / file_name
    download_single_file(file_url, local_path)

print("All files have been downloaded.")
print("You can now set the following path in build_dual_medvector.py:")
print("   model_dir = Path('model') / 'all-MiniLM-L6-v2'")
print("to load the local model directly without connecting to the Hugging Face Hub.")
