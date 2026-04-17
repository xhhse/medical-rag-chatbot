# build_dual_medvector.py

import os
import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import torch


# Load the local all-MiniLM-L6-v2 embedding model.
# Using a local model path is the most stable option if the full model files
# have already been downloaded.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "model/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_DIR, device=DEVICE)

# Alternative: load from a Hugging Face mirror if needed.
# model = SentenceTransformer("obrizum/all-MiniLM-L6-v2")


# Define an embedding wrapper so LangChain can use SentenceTransformer.
class MiniLMEncoder:
    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, text):
        return model.encode([text])[0].tolist()


embedding_model = MiniLMEncoder()


# Load the MedMCQA and MedRedQA JSON files.
def load_medmcqa_qa(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_medredqa_qa(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


medmcqa_data = load_medmcqa_qa("medmcqa_qa.json")      # MedMCQA: multiple-choice questions
medredqa_data = load_medredqa_qa("medredqa_qa.json")   # MedRedQA: patient questions and answers


# Build a FAISS vector database.
def create_db(data, name):
    print(f"Building vector database for {name} ({len(data)} records)...")
    docs = []

    for idx, item in enumerate(data):
        # Use the question and answer as the embedding content.
        content = f"Q: {item['q']}\nA: {item['a']}"
        doc = Document(
            page_content=content,
            metadata={
                **item,  # Keep all original fields as metadata
                "qa_id": str(idx),
                "source": item.get("source", name),
            },
        )
        docs.append(doc)

    # Create a FAISS vector store using the custom embedding model.
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
    )

    # Save the vector store to vectorstores/{name}.
    save_dir = os.path.join("vectorstores", name)
    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(save_dir)

    print(f"Saved vector database to {save_dir}")


# Create two separate vector stores.
if __name__ == "__main__":
    os.makedirs("vectorstores", exist_ok=True)

    # MedMCQA: multiple-choice question bank
    create_db(medmcqa_data, "medmcqa_full")

    # MedRedQA: patient question-answer bank
    create_db(medredqa_data, "medredqa_full")

    print("All vector databases have been created successfully.")