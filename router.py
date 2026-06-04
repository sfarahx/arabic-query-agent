from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sql_demo import run_sql_agent
from rag_demo import run_rag_agent

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Phrases that indicate an agent returned no useful result.
# Used by _is_empty_result() to catch silent failures before blending.
_EMPTY_SIGNALS = [
    "i don't know",
    "i do not know",
    "no results",
    "not found",
    "no information",
    "cannot answer",
    "can't answer",
    "insufficient",
    "لا أعرف",
    "لا توجد معلومات",
    "لا يمكنني",
    "لم يتم العثور",
]

def _is_empty_result(text: str) -> bool:
    """
    Returns True if the agent result is empty or meaningless.
    Catches two failure modes:
      - Truly empty string
      - LLM saying "I don't know" / "no results" in English or Arabic
    """
    if not text or len(text.strip()) < 10:
        return True
    lower = text.lower()
    return any(signal in lower for signal in _EMPTY_SIGNALS)

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
- "Can Engineering employees work from home?" → RAG (a department mentioned as a group is a policy question — no individual named, no DB lookup needed)
- "Is the Finance department eligible for remote work?" → RAG (department as a category, answer is in the policy document)
- Do NOT choose HYBRID just because the answer involves a person's name — choose HYBRID only when a policy or rule is also required
- Do NOT choose HYBRID when a department is mentioned as a group — a department name is NOT the same as an individual's name. HYBRID requires a specific named person.

Reply with only one word: SQL, RAG, or HYBRID

Question: {query}"""

    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    return decision if decision in ["SQL", "RAG", "HYBRID"] else "HYBRID"

def _reframe_as_data_fetch(query: str) -> str:
    """
    Rewrite the user's question into a pure data-retrieval task before sending
    to the SQL agent. This prevents the agent from returning "I don't know"
    when it finds factual data but can't answer the policy part of the question.
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
    individuals. Removing the person's name prevents RAG from saying
    "I can't determine" and forces it to return the relevant policy.
    """
    prompt = f"""Rewrite the following question as a general policy or rule lookup query.
- Remove any specific person's name or personal details
- Focus only on the type of policy, rule, or document being referenced
- Return only the rewritten query in English, nothing else

Original question: {query}"""
    return llm.invoke(prompt).content.strip()

def run_hybrid(query: str) -> str:
    sql_fetch_query = _reframe_as_data_fetch(query)
    rag_fetch_query = _reframe_for_rag(query)
    print(f"[Hybrid] SQL reframed query: {sql_fetch_query}")
    print(f"[Hybrid] RAG reframed query: {rag_fetch_query}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_sql = executor.submit(run_sql_agent, sql_fetch_query)
        fut_rag = executor.submit(run_rag_agent, rag_fetch_query)
        sql_result = fut_sql.result()
        rag_result = fut_rag.result()

    sql_empty = _is_empty_result(sql_result)
    rag_empty = _is_empty_result(rag_result)

    print(f"[Hybrid] SQL empty: {sql_empty} | RAG empty: {rag_empty}")

    # If both sources failed, return early — don't send empty inputs to blend
    if sql_empty and rag_empty:
        return "لم يتم العثور على معلومات كافية للإجابة على هذا السؤال."

    # If one source failed, answer from the other alone and be transparent about it
    if sql_empty:
        print("[Hybrid] SQL returned no results — answering from RAG only")
        return rag_result + "\n\n(ملاحظة: لم يتم العثور على بيانات مطابقة في قاعدة البيانات)"

    if rag_empty:
        print("[Hybrid] RAG returned no results — answering from SQL only")
        return sql_result + "\n\n(ملاحظة: لم يتم العثور على سياسة ذات صلة في الوثائق)"

    # Both sources returned results — blend normally
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
        result = run_sql_agent(query)
        if _is_empty_result(result):
            return "لم يتم العثور على نتائج مطابقة في قاعدة البيانات."
        return result

    elif decision == "RAG":
        result = run_rag_agent(query)
        if _is_empty_result(result):
            return "لم يتم العثور على معلومات ذات صلة في الوثائق."
        return result

    else:
        return run_hybrid(query)