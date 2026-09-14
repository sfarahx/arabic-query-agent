# Arabic Query Agent
**QCRI LLM Lab, Summer 2026**

An Arabic natural language query agent that classifies incoming questions and routes them to the appropriate answering pipeline: SQL, RAG, or a hybrid of both. Designed as generic, schema-agnostic infrastructure intended for use across multiple projects.

> **Note:** This is a personal exploratory repo, not official or final internship work — built to learn the concepts (LangChain agents, Text-to-SQL, RAG, hybrid routing) needed for the actual project.

---

## How It Works

```
Arabic question
      │
      ▼
 classify_query()          ← LLM decides: SQL / RAG / HYBRID
      │
  ┌───┴──────────────┐
  │                  │
 SQL               HYBRID
  │           ┌─────┴─────┐
  │     reframe_sql   reframe_rag      ← query reshaped per source
  │           │             │
  │     run_sql_agent  run_rag_agent   ← run in parallel
  │           └─────┬─────┘
  │              blend()              ← LLM combines both results
  └──────────────────┤
                     ▼
              Arabic answer
```

- **SQL** — translates the query to English, then uses a LangChain SQL agent to query SQLite (structured data: records, counts, rankings).
- **RAG** — translates to English (embedding model is English-only), retrieves relevant chunks from a ChromaDB vector store (policy/document context).
- **HYBRID** — reframes the query twice (SQL reframe strips policy questions; RAG reframe strips names), runs both pipelines in parallel, then blends the results into one answer.

---

## Project Structure

```
arabic-query-agent/
├── app.py           # Streamlit UI with RTL Arabic support
├── router.py        # Classifier, reframers, hybrid orchestration
├── sql_demo.py      # SQL agent (SQLite + LangChain)
├── rag_demo.py      # RAG pipeline (ChromaDB + HuggingFace embeddings)
├── benchmark.py     # Routing accuracy evaluation
├── company.db       # Demo SQLite database (employees table)
└── policy.txt        # Auto-generated on startup from rag_demo.py
```

---

## Setup

```bash
pip install streamlit langchain langchain-groq langchain-huggingface langchain-community chromadb sentence-transformers python-dotenv pandas
```
```
GROQ_API_KEY=your_key_here   # in .env
```
```bash
streamlit run app.py
```

---

## Demo Data

Ships with a pre-seeded SQLite employees table and a sample remote-work policy document, used to test both the SQL and RAG pipelines end to end.

---

## Benchmark

The router is evaluated on **27 labeled Arabic queries** across two difficulty tiers (13 basic, 14 hard — ambiguous phrasing, reverse HYBRID, answer-is-a-name SQL cases).

```bash
python benchmark.py
```
Results are saved to `benchmark_results.json` with per-tier and per-route breakdowns.

---

## Limitations & Planned Improvements

- **Embedding model is English-only** (`all-MiniLM-L6-v2`) — queries are translated as a workaround. Planned fix: swap to `paraphrase-multilingual-MiniLM-L12-v2`.
- **Name transliteration inconsistency** — Arabic names can transliterate differently (e.g. ليلى → Leila/Layla), which may cause SQL lookups to miss records.
- **Hardcoded data sources** — DB path and documents folder are fixed in `sql_demo.py`/`rag_demo.py`. Externalizing via config file is planned, along with removing the hardcoded `_setup_database()` so the agent connects to existing DBs.
- **No error handling** for empty SQL/RAG results — needs graceful fallback.
- **Parallel-only HYBRID** — SQL and RAG run independently with no dependency between them; multi-step reasoning (one feeding the other) isn't supported yet.
- **Groq free tier** — 100,000 tokens/day; running the full benchmark uses a significant chunk of that.
