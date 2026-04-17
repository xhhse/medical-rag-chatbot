# Medical QA Chatbot

This project is a multilingual medical question-answering system built with Streamlit, LangChain, FAISS, SentenceTransformers, and the OpenAI API.

The application follows a retrieval-augmented generation (RAG) architecture. User questions are embedded with a local sentence-transformer model, matched against two FAISS vector stores, and then passed to a large language model to generate grounded answers based on retrieved medical references.

## Project Structure

```text
.
├── app.py
├── chains.py
├── utils.py
├── download_model.py
├── build_dual_medvector.py
├── build_medmcqa_qa_dataset.py
├── build_medredqa_qa_dataset.py
├── medmcqa_qa.json
├── medredqa_qa.json
├── model/
│   └── all-MiniLM-L6-v2/
├── vectorstores/
│   ├── medmcqa_full/
│   └── medredqa_full/
├── medmcqa/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── medredqa/
│   ├── medredqa_train.json
│   ├── medredqa_val.json
│   └── medredqa_test.json
└── .env
```

## System Overview

The system is divided into three main layers:

### 1. Model and data preparation
- `download_model.py` downloads the local embedding model.
- `build_medmcqa_qa_dataset.py` converts MedMCQA raw CSV files into a QA JSON format.
- `build_medredqa_qa_dataset.py` converts MedRedQA raw JSON files into a QA JSON format.
- `build_dual_medvector.py` builds the FAISS vector stores from the processed JSON files.

### 2. Retrieval and generation
- `chains.py` loads the local embedding model.
- It performs language-aware retrieval from the two FAISS indices.
- It sends the retrieved context, history, and user question to the OpenAI model.
- The response is generated in the same language as the user input whenever possible.

### 3. User interface
- `app.py` implements the Streamlit chat application.
- `utils.py` manages chat history and reference rendering.
- The interface displays answers, source documents, and conversation history.

## Data Sources

### MedMCQA
MedMCQA is a large-scale medical multiple-choice question answering dataset based on medical exam questions. It includes:
- Question text.
- Answer choices.
- Correct answer option.
- Subject name.
- Topic name.
- Explanation field.

The dataset is downloaded from Kaggle and converted into a QA-style JSON file before indexing.

[MedMCQA dataset](https://www.kaggle.com/datasets/thedevastator/medmcqa-medical-mcq-dataset)

### MedRedQA
MedRedQA is a medical consumer question-answering dataset containing expert responses. The PubMed-enriched variant adds related biomedical context when available.

The dataset is downloaded from CSIRO and converted into a QA-style JSON file before indexing.

[MedRedQA dataset](https://data.csiro.au/collection/csiro:62454)

## Embedding Model

The local embedding model is `sentence-transformers/all-MiniLM-L6-v2`.

It is used because it is:
- lightweight,
- fast for local inference,
- suitable for semantic retrieval,
- practical for a local RAG pipeline.

The model is downloaded once by `download_model.py` and stored locally in:

```text
model/all-MiniLM-L6-v2/
```

This allows the application to run without downloading model files at runtime.

## Vector Store Design

Two separate FAISS indices are maintained:

- `vectorstores/medmcqa_full`
- `vectorstores/medredqa_full`

Each stored document contains:
- the combined question and answer text in `page_content`,
- original dataset metadata in `metadata`.

Keeping the indices separate makes it easier to:
- control retrieval behavior by language,
- preserve dataset provenance,
- distinguish exam-style from consumer-style medical content.

## Retrieval Strategy

The retrieval layer uses language-aware routing:

- English questions prioritize MedRedQA.
- Traditional Chinese questions use mixed retrieval from both datasets.
- Swedish questions are supported through the same multilingual pipeline.

The detected language is passed into the generation prompt so the final answer can be produced in the same language as the user’s input whenever possible.

## Response Generation

After retrieval, the system builds a prompt containing:
- user question,
- conversation history,
- user metadata,
- retrieved context passages.

The LLM is instructed to:
- answer only from the provided references,
- avoid diagnosis or treatment advice,
- respond in the same language as the user,
- include a short source summary in the final answer.

## Chat Interface

The Streamlit UI uses:
- `st.chat_input()` for user prompts,
- `st.chat_message()` for rendering the conversation,
- `st.session_state["history"]` for preserving chat history across reruns.

Each message stores:
- `role` (`user` or `assistant`),
- `content`,
- `references`.

This makes it possible to re-render the full chat log after every interaction.

## Reference Handling

Assistant responses include the source documents used during retrieval. These references are normalized and displayed in a collapsible summary area for transparency.

Reference metadata typically includes:
- source dataset,
- question text,
- answer text,
- question type.

## Safety Notes

This app is intended for informational purposes only. It does not provide medical diagnosis, treatment plans, or prescriptions.

If symptoms are severe or urgent, users should seek immediate professional medical care.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/xhhse/medical-rag-chatbot.git
cd medical-rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows
```bash
.venv\Scripts\activate
```

#### macOS / Linux
```bash
source .venv/bin/activate
```

### 4. Install dependencies manually
```bash
pip install streamlit langchain langchain-community langchain-openai sentence-transformers langdetect python-dotenv faiss-cpu torch
```

### 5. Download the local embedding model
```bash
python download_model.py
```

### 6. Prepare the datasets
```bash
python build_medmcqa_qa_dataset.py
python build_medredqa_qa_dataset.py
```

### 7. Build the FAISS vector stores
```bash
python build_dual_medvector.py
```

## Usage

### 1. Add your OpenAI API key
Create a `.env` file in the project root and add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

### 3. Open the app in your browser
After the server starts, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 4. Ask a question
You can enter questions in:
- Traditional Chinese
- English
- Swedish

The app will retrieve relevant medical references and generate a response in the same language whenever possible.

## Main Files

- `download_model.py`  
  Downloads the local SentenceTransformer model from Hugging Face.

- `build_medmcqa_qa_dataset.py`  
  Converts MedMCQA raw CSV files into QA JSON.

- `build_medredqa_qa_dataset.py`  
  Converts MedRedQA raw JSON files into QA JSON.

- `build_dual_medvector.py`  
  Builds FAISS vector stores for both medical datasets.

- `chains.py`  
  Handles embedding, retrieval, prompt construction, and LLM response generation.

- `utils.py`  
  Manages chat history and reference rendering in Streamlit.

- `app.py`  
  Implements the Streamlit user interface.

## Technical Stack

- Streamlit
- LangChain
- FAISS
- SentenceTransformers
- OpenAI API
- langdetect