# Arabic Query Agent — Project 11
**QCRI LLM Lab, Summer 2026**

An Arabic natural language query agent that classifies incoming questions and routes them to the appropriate answering pipeline: SQL, RAG, or a hybrid of both. Designed as generic, schema-agnostic infrastructure intended for use across multiple projects.

---

## How It Works

Every query goes through three stages:

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

### SQL pipeline
- Translates the Arabic query to English
- Uses a LangChain SQL agent to query SQLite
- Returns structured data (employee records, counts, rankings, etc.)

### RAG pipeline
- Translates the Arabic query to English before retrieval (embedding model is English-only)
- Retrieves relevant chunks from a ChromaDB vector store
- Returns policy or document context

### HYBRID pipeline
- Reshapes the query twice before running in parallel:
  - **SQL reframe** — strips policy questions, focuses on data retrieval only
  - **RAG reframe** — strips person names, focuses on policy lookup only
- A final blend step combines both results and reasons over them to produce a complete answer

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
└── policy.txt       # Auto-generated on startup from rag_demo.py
```

---

## Setup

**1. Create and activate a virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

**2. Install dependencies:**
```bash
pip install streamlit langchain langchain-groq langchain-huggingface \
            langchain-community chromadb sentence-transformers \
            python-dotenv pandas
```

**3. Create a `.env` file:**
```
GROQ_API_KEY=your_key_here
```

**4. Run the app:**
```bash
streamlit run app.py
```

---

## Demo Data

The demo ships with a pre-seeded SQLite database and a remote work policy document.

**Employees table:**
| Name | Department | Salary |
|---|---|---|
| Ahmed Al-Mansouri | HR | 8,000 |
| Sara Al-Khalidi | Engineering | 12,000 |
| Omar Farouk | HR | 7,500 |
| Layla Hassan | Engineering | 11,000 |
| Khalid Al-Rashid | Finance | 9,000 |

**Remote Work Policy:**
- Engineering → 3 remote days/week
- HR → on-site minimum 4 days/week (1 remote day allowed)
- Finance → not eligible for remote work
- All requests require department head approval
- Work week: Sunday–Thursday (Friday–Saturday are weekend)

---

## Benchmark

The router is evaluated on **27 labeled Arabic queries** split into two difficulty tiers:

| Tier | Queries | Tests |
|---|---|---|
| Original | 13 | Basic routing accuracy |
| Hard | 14 | Ambiguous phrasing, reverse HYBRID, answer-is-a-name SQL |

To run:
```bash
python benchmark.py
```

Results are saved to `benchmark_results.json` with per-tier and per-route breakdowns.

---

## Known Limitations

- **English-only embedding model** — `all-MiniLM-L6-v2` does not natively understand Arabic. Queries are translated to English before retrieval as a workaround. Replacing with `paraphrase-multilingual-MiniLM-L12-v2` is the planned fix.
- **Name transliteration inconsistency** — Arabic names can transliterate differently (e.g. ليلى → Leila or Layla), which may cause SQL lookups to miss records whose names are stored under a different spelling.
- **Hardcoded data sources** — `sql_demo.py` and `rag_demo.py` currently point to fixed file paths. Making these configurable at runtime is a planned improvement before the agent is used as shared infrastructure.
- **Parallel HYBRID only** — the current architecture runs SQL and RAG in parallel with no dependency between them. Multi-step reasoning (where SQL results feed into the RAG query or vice versa) is not yet supported.
- **Groq free tier limit** — the free tier allows 100,000 tokens/day. Running the full benchmark consumes a significant portion of this.

---

## Planned Improvements

- [ ] Swap embedding model to `paraphrase-multilingual-MiniLM-L12-v2`
- [ ] Externalize DB path and documents folder via config file
- [ ] Remove hardcoded `_setup_database()` — agent should connect to existing DBs
- [ ] Add graceful error handling when SQL or RAG returns no results
- [ ] Runtime schema introspection for use as shared infrastructure (pending mentor approval)