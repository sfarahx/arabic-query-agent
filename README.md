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
- **RAG** — retrieves relevant chunks directly on the Arabic query using a multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`), no translation step (policy/document context).
- **HYBRID** — reframes the query twice (SQL reframe strips policy questions; RAG reframe strips names), runs both pipelines in parallel, then blends the results into one answer. If one or both sources return an empty/unhelpful result, falls back gracefully instead of blending garbage.

---

## Project Structure

```
arabic-query-agent/
├── app.py           # Streamlit UI (RTL Arabic support, plus a Data tab for browsing the DB/policy)
├── router.py        # Classifier, reframers, hybrid orchestration, empty-result handling
├── sql_demo.py      # SQL agent (SQLite + LangChain)
├── rag_demo.py      # RAG pipeline (ChromaDB + HuggingFace embeddings)
├── config.py        # Loads config.yaml
├── config.yaml       # DB path and documents folder/files — used by rag_demo.py and app.py
├── benchmark.py     # Routing accuracy evaluation
├── company.db       # Demo SQLite database (employees table)
└── policy.txt        # Auto-generated on startup from rag_demo.py (opt-in via config)
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

## Known Limitations

- **Name transliteration inconsistency** — Arabic names can transliterate differently (e.g. ليلى → Leila/Layla), which may cause SQL lookups to miss records.
- **Config externalization is only half done** — `rag_demo.py` and `app.py` read DB/document paths from `config.yaml`, but `sql_demo.py` is still hardcoded to `company.db` and still runs `_setup_database()` on every startup, recreating the demo table. Pointing it at `config.py` and dropping `_setup_database()` is still pending.
- **Parallel-only HYBRID** — SQL and RAG run independently with no dependency between them; multi-step reasoning (one feeding the other) isn't supported yet.
- **Groq free tier** — 100,000 tokens/day; running the full benchmark uses a significant chunk of that.
