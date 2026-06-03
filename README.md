# Arabic Query Agent

Arabic-language query agent that routes questions to SQL, RAG, or hybrid pipelines.

## Overview

This is a demo project that accepts Arabic natural language questions and intelligently routes them to the appropriate answering pipeline:

- **SQL** — for questions about structured data (e.g. employee records, salaries, departments)
- **RAG** — for questions about unstructured documents (e.g. company remote work policy)
- **Hybrid** — for questions that require both structured data and document context

The agent is built with a Streamlit interface and uses an LLM-based router to classify each query before passing it to the right pipeline.

## Project Structure

```
arabic-query-agent/
├── app.py              # Streamlit UI
├── router.py           # Query classifier and pipeline router
├── sql_demo.py         # SQL agent (SQLite + LangChain)
├── rag_demo.py         # RAG pipeline (ChromaDB + HuggingFace embeddings)
├── benchmark.py        # Routing accuracy evaluation
└── policy.txt          # Auto-generated remote work policy document
```

## Setup

1. Clone the repository and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install streamlit langchain langchain-groq langchain-huggingface langchain-community chromadb sentence-transformers python-dotenv pandas
   ```

3. Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_key_here
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```
