"""
utils.py - Chat history helpers for Streamlit
"""

from typing import Any, Dict, List, Optional

import streamlit as st


def init_state() -> None:
    """Initialize session state keys used by the chat app."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "user_info" not in st.session_state:
        st.session_state["user_info"] = {}


def normalize_references(
    references: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, str]]:
    """Normalize references into a stable display format."""
    normalized: List[Dict[str, str]] = []

    for ref in references or []:
        normalized.append(
            {
                "source": str(ref.get("source", "Unknown source")),
                "type": str(ref.get("type", "Medical QA")),
                "question": str(ref.get("question", "")),
                "answer": str(ref.get("answer", "")),
            }
        )

    return normalized


def set_chat_message(
    role: str,
    content: str,
    references: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Append one chat message to session history."""
    init_state()

    if role not in {"user", "assistant"}:
        role = "assistant"

    st.session_state["history"].append(
        {
            "role": role,
            "content": content,
            "references": normalize_references(references),
        }
    )


def render_references(references: List[Dict[str, str]]) -> None:
    """Render reference sources in an expander."""
    if not references:
        return

    with st.expander("References"):
        for ref in references:
            st.markdown(f"**{ref['source']}**: {ref['question']}")
            if ref["answer"]:
                st.caption(ref["answer"])


def write_history() -> None:
    """Render the full chat history in Streamlit."""
    init_state()

    for message in st.session_state["history"]:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        references = normalize_references(message.get("references", []))

        with st.chat_message(role if role in {"user", "assistant"} else "assistant"):
            st.markdown(content)

            if role == "assistant":
                render_references(references)