"""
chains.py - Multilingual medical RAG pipeline using local sentence-transformers
and two FAISS vector stores: medmcqa_full and medredqa_full.

Supported response languages:
- Traditional Chinese
- English
- Swedish
"""

import os
from typing import Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()

MODEL_DIR = "model/all-MiniLM-L6-v2"
VECTORSTORE_DIR_MCQA = "vectorstores/medmcqa_full"
VECTORSTORE_DIR_MRQA = "vectorstores/medredqa_full"


def get_llm() -> ChatOpenAI:
    """Create and validate the OpenAI chat model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or len(api_key) < 20:
        raise ValueError("OPENAI_API_KEY is missing or invalid. Please check your .env file.")

    print(f"API key loaded successfully (length: {len(api_key)}).")

    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        api_key=api_key,
        timeout=30,
        max_retries=2,
    )


class MiniLMEncoder:
    """LangChain-compatible embedding wrapper for SentenceTransformer."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


@st.cache_resource
def load_sentence_model() -> SentenceTransformer:
    """Load the local SentenceTransformer model once and cache it."""
    return SentenceTransformer(MODEL_DIR, device=DEVICE)


@st.cache_resource
def get_embedding_model() -> MiniLMEncoder:
    """Create and cache the embedding wrapper."""
    return MiniLMEncoder(load_sentence_model())


@st.cache_resource
def get_llm_cached() -> ChatOpenAI:
    """Create and cache the OpenAI chat model."""
    return get_llm()


@st.cache_resource
def load_vectorstores() -> Tuple[Optional[FAISS], Optional[FAISS]]:
    """Load MedMCQA and MedRedQA FAISS vector stores once and cache them."""
    embedding_model = get_embedding_model()

    try:
        print("Loading vector stores...")

        db_medmcqa = FAISS.load_local(
            VECTORSTORE_DIR_MCQA,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        db_medredqa = FAISS.load_local(
            VECTORSTORE_DIR_MRQA,
            embedding_model,
            allow_dangerous_deserialization=True,
        )

        print("Loaded medmcqa_full successfully.")
        print("Loaded medredqa_full successfully.")
        return db_medmcqa, db_medredqa

    except FileNotFoundError as err:
        print(f"Vector store not found: {err}")
        print(
            "Please confirm that vectorstores/ contains medmcqa_full and "
            "medredqa_full, then run build_dual_medvector.py."
        )
        return None, None
    except Exception as err:
        print(f"Failed to load vector stores: {err}")
        return None, None


def detect_language(text: str) -> str:
    """Detect the language of the user's question."""
    try:
        return detect(text)
    except Exception:
        return "zh"


def get_response_language(lang: str) -> str:
    """Map detected language codes to human-readable language names."""
    if lang == "en":
        return "English"
    if lang == "sv":
        return "Swedish"
    return "Traditional Chinese"


def hybrid_retrieve(question: str, k: int = 4):
    """Perform hybrid retrieval with simple language-aware routing."""
    print(f"Starting hybrid retrieval: {question[:30]}...")

    db_medmcqa, db_medredqa = load_vectorstores()
    if db_medmcqa is None or db_medredqa is None:
        print("Vector stores are unavailable. Returning an empty result list.")
        return []

    lang = detect_language(question)
    print(f"Detected language: {lang}")

    docs = []

    try:
        if lang == "sv":
            docs.extend(db_medmcqa.similarity_search(question, k=2))
            docs.extend(db_medredqa.similarity_search(question, k=2))
        elif lang == "en":
            docs.extend(db_medredqa.similarity_search(question, k=3))
            docs.extend(db_medmcqa.similarity_search(question, k=1))
        else:
            docs.extend(db_medredqa.similarity_search(question, k=2))
            docs.extend(db_medmcqa.similarity_search(question, k=2))

        print(f"Hybrid retrieval completed. Retrieved {len(docs)} documents.")
        return docs[:k]

    except Exception as err:
        print(f"Retrieval error: {err}")
        return []


PROMPT_TEMPLATE = """
You are a professional medical assistant. Answer the question strictly based on the provided information.
Do not diagnose, do not prescribe medication, and do not claim to be a doctor.

Detected language code: {lang}
Detected response language: {response_language}

User information: {user_info}
Conversation history: {history}

Reference materials:
{context}

Question: {question}

Please answer in the same language as the user's question.

Language rules:
- If the user's question is in English, answer in English.
- If the user's question is in Traditional Chinese, answer in Traditional Chinese.
- If the user's question is in Swedish, answer in Swedish.
- If the user's question is in another language, respond in that same language if possible.
- Do not switch languages unless the user explicitly asks you to.

Please follow this structure:
1. Recommendation
2. Possible causes
3. When to seek medical care
4. Source summary (mention whether the information comes from MedMCQA or MedRedQA)

Rules:
- If the available information is insufficient, say clearly that no definite conclusion can be made and advise the user to consult a doctor.
- All responses are for informational purposes only and are not a substitute for professional medical advice.
"""


def get_suggestion_chain(
    question: str,
    history: str = "",
    user_info: Optional[Dict] = None,
) -> Dict:
    """Main entry point for app.py."""
    if user_info is None:
        user_info = {}

    print(f"Processing question: {question[:50]}...")

    lang = detect_language(question)
    response_language = get_response_language(lang)
    docs = hybrid_retrieve(question)
    print(f"Retrieved {len(docs)} source documents.")

    if not docs:
        return {
            "result": (
                "The vector stores are not ready, or no relevant content was found. "
                "Please confirm that vectorstores/ contains medmcqa_full and "
                "medredqa_full, and that build_dual_medvector.py has been executed."
            ),
            "source_documents": [],
        }

    context = "\n\n".join(
        [
            f"[{i + 1}] {doc.page_content[:250]}\nSource: {doc.metadata.get('source', 'Unknown')}"
            for i, doc in enumerate(docs)
        ]
    )

    print("Generating OpenAI response...")
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    messages = prompt.format_messages(
        user_info=user_info,
        history=history,
        context=context,
        question=question,
        lang=lang,
        response_language=response_language,
    )

    llm = get_llm_cached()

    try:
        response = llm.invoke(messages)
    except Exception as err:
        print(f"OpenAI call failed: {err}")
        return {
            "result": (
                "The LLM is temporarily unavailable. Please try again later, or "
                "check your OpenAI API key and network connection."
            ),
            "source_documents": [],
        }

    source_documents = []
    for i, doc in enumerate(docs):
        source_documents.append(
            {
                "_id": doc.metadata.get("qa_id", f"doc_{i}"),
                "source": doc.metadata.get("source", "Medical data"),
                "type": doc.metadata.get("type", "mixed"),
                "question": doc.page_content[:100],
                "answer": doc.page_content[:200],
            }
        )

    print(f"Response generated successfully with {len(docs)} sources.")
    return {
        "result": response.content,
        "source_documents": source_documents,
    }


if __name__ == "__main__":
    print("Starting chains.py self-test...")

    db1, db2 = load_vectorstores()
    if db1 is not None and db2 is not None:
        print("Vector store loading test passed.")
    else:
        print("Vector store loading test failed.")

    test_questions = [
        "頭痛怎麼辦？",
        "What should I do if I have a headache?",
        "Vad ska jag göra om jag har huvudvärk?",
    ]

    for q in test_questions:
        print(f"\nTesting retrieval: {q}")
        docs_test = hybrid_retrieve(q)
        for i, doc in enumerate(docs_test):
            print(f"[{i + 1}] {doc.page_content[:100]}...")

    print("\nTesting full RAG flow with multilingual queries...")
    for q in test_questions:
        suggestion = get_suggestion_chain(q, history="")
        print(f"\nQuestion: {q}")
        print("Result preview:")
        print(suggestion["result"][:300])

    print("Done.")