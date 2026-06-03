import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sql_demo import run_sql_agent
from rag_demo import run_rag_agent

load_dotenv()

llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

def classify_query(query: str) -> str:
    prompt = f"""You are a query classifier. Given a user question, decide whether it should be answered using:
- SQL: if it can be fully answered from structured data alone (numbers, counts, rankings, salaries, lists, comparisons — even if the answer happens to be a person's name)
- RAG: if it asks about policies, rules, or descriptive information with no reference to specific people or records
- HYBRID: ONLY if it genuinely requires BOTH structured data AND policy knowledge to answer — neither source alone is sufficient

IMPORTANT rules:
- "Who has the highest salary?" → SQL (pure data ranking, no policy needed)
- "How many employees are in each department?" → SQL (pure data aggregation)
- "Is Ahmed allowed to work remotely?" → HYBRID (need SQL for Ahmed's department, RAG for the remote work policy)
- "Which employees are eligible for remote work?" → HYBRID (need RAG for the eligibility rule, SQL to find who qualifies)
- "What is the remote work policy?" → RAG (pure policy question, no individual data needed)
- Do NOT choose HYBRID just because the answer involves a person's name — choose HYBRID only when a policy or rule is also required

Reply with only one word: SQL, RAG, or HYBRID

Question: {query}"""

    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    return decision if decision in ["SQL", "RAG", "HYBRID"] else "HYBRID"

def _reframe_as_data_fetch(query: str) -> str:
    """
    Rewrite the user's question into a pure data-retrieval task before sending
    to the SQL agent. This prevents the agent from returning "I don't know"
    when it finds factual data but can't answer the policy part of the question
    (e.g. "is X allowed to do Y?" has no answer in the DB — but the DB does
    know X's department, which is all we need here).
    """
    prompt = f"""You are a query rewriter for a database retrieval system.
Rewrite the question below into a plain data-fetch request in English.
- Extract only the factual information needed from the database (names, departments, roles, salaries, etc.)
- Remove any policy, eligibility, or permission questions — those will be answered separately
- Return only the rewritten English query, nothing else

Original question: {query}"""
    return llm.invoke(prompt).content.strip()

def _reframe_for_rag(query: str) -> str:
    """
    Strip person-specific details before RAG lookup.
    In HYBRID mode, RAG's job is to retrieve policy/rules — not answer about
    individuals. Asking RAG "is Layla Hassan allowed to work remotely?" causes
    it to say "I can't determine" because the policy doc has no record of her.
    Asking "what is the remote work policy?" retrieves the full policy, which
    the blend step can then apply to the SQL-fetched person data.
    """
    prompt = f"""Rewrite the following question as a general policy or rule lookup query.
- Remove any specific person's name or personal details
- Focus only on the type of policy, rule, or document being referenced
- Return only the rewritten query in English, nothing else

Original question: {query}"""
    return llm.invoke(prompt).content.strip()

def run_hybrid(query: str) -> str:
    # Reframe separately for each source:
    # SQL  → data-fetch task (who is this person, what are their attributes?)
    # RAG  → policy-lookup task (what are the rules for this topic?)
    # Both run in parallel; the blend step combines them to answer the original question.
    sql_fetch_query = _reframe_as_data_fetch(query)
    rag_fetch_query = _reframe_for_rag(query)
    print(f"[Hybrid] SQL reframed query: {sql_fetch_query}")
    print(f"[Hybrid] RAG reframed query: {rag_fetch_query}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_sql = executor.submit(run_sql_agent, sql_fetch_query)
        fut_rag = executor.submit(run_rag_agent, rag_fetch_query)
        sql_result = fut_sql.result()
        rag_result = fut_rag.result()

    # Let the blend step reason over both regardless of direction
    blend_prompt = f"""You are a helpful assistant. A user asked a question that requires both database records and policy/document knowledge to answer.

User question: {query}

Database records:
{sql_result}

Policy/document information:
{rag_result}

Instructions:
- Reason over BOTH sources together to form a complete answer
- The database may tell you facts about a person (e.g. their department)
- The policy may tell you rules that apply to that fact (e.g. what that department is allowed)
- Or the policy may define criteria and the database may tell you who meets them
- Work out the answer by combining both — do not rely on either source alone
- Answer in the same language as the question"""

    return llm.invoke(blend_prompt).content

def route(query: str) -> str:
    decision = classify_query(query)
    print(f"[Router] Decision: {decision}")

    if decision == "SQL":
        return run_sql_agent(query)
    elif decision == "RAG":
        return run_rag_agent(query)
    else:
        return run_hybrid(query)