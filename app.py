"""
Swedish Medical QA Chatbot - Production Version

Supports:
- OpenAI API
- FAISS vector stores: medmcqa_full / medredqa_full
- Multilingual input/output: Traditional Chinese, English, and Swedish
- Unified chat roles: "user" / "assistant"
"""

import streamlit as st
import chains
import utils
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {}

# Page configuration
st.set_page_config(
    page_title="Swedish Medical QA Chatbot",
    page_icon="🩺",
    layout="wide",
)

# Page title and warning banner
st.title("Medical QA Chatbot")
st.markdown("### Important Notice")
st.warning(
    """
This system is provided for medical information purposes only and is not a diagnostic tool.

- If you feel unwell, please seek medical care promptly.
- For emergencies in Sweden, call 112.
- Data sources: MedMCQA and MedRedQA public datasets.
"""
)

# Render chat history
utils.write_history()

# Chat input
if question := st.chat_input("Enter your medical question in Chinese, English, or Swedish..."):
    question = question.strip()

    if question:
        # Store the user message
        utils.set_chat_message("user", question)

        try:
            # Build recent conversation history
            history_text = "\n".join(
                [
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in st.session_state["history"][-8:]
                ]
            )

            # Call the RAG pipeline
            suggestion = chains.get_suggestion_chain(
                question=question,
                history=history_text,
                user_info=st.session_state["user_info"],
            )

            # Store the assistant response
            utils.set_chat_message(
                "assistant",
                suggestion["result"],
                suggestion.get("source_documents", []),
            )

        except FileNotFoundError:
            utils.set_chat_message(
                "assistant",
                (
                    "Vector stores are not ready yet. Please run "
                    "`python build_dual_medvector.py` first."
                ),
            )
        except Exception as err:
            st.error(f"System error: {err}")
            utils.set_chat_message(
                "assistant",
                "Sorry, the system is temporarily unavailable. Please try again later.",
            )

# Render chat history again so the latest message is visible after rerun
utils.write_history()

# Recent answer summary
if (
    st.session_state["history"]
    and len(st.session_state["history"]) > 1
    and st.session_state["history"][-1]["role"] == "assistant"
):
    with st.expander("Question Answer Summary", expanded=False):
        last_ai = st.session_state["history"][-1]
        references = last_ai.get("references", [])

        if references:
            st.subheader("References")
            for reference in references[:3]:
                with st.container(border=True):
                    st.markdown(f"**Source**: {reference.get('source', 'Unknown')}")
                    st.markdown(f"**Type**: {reference.get('type', 'Medical QA')}")
                    st.markdown("**Question**")
                    st.caption(reference.get("question", ""))
                    st.markdown("**Answer**")
                    st.caption(reference.get("answer", ""))
                    st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    """
**Technical stack**: LangChain RAG + MedMCQA + MedRedQA + FAISS  
**Language support**: Traditional Chinese / English / Swedish  
**Status**: Production-ready prototype
"""
)

st.markdown("This is the end of the chat area.")