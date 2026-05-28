from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sql_demo import run_sql_agent
from rag_demo import run_rag_agent

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def classify_query(query: str) -> str:
    prompt = f"""You are a query classifier. Given a user question, decide whether it should be answered using:
- SQL: if it asks about specific data, numbers, counts, lists of records, or what items exist in the system
- RAG: if it asks about policies, rules, or descriptive information
- HYBRID: if it needs both structured data AND policy/descriptive information to answer fully

Reply with only one word: SQL, RAG, or HYBRID

Question: {query}"""
    
    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    return decision if decision in ["SQL", "RAG", "HYBRID"] else "RAG"

def combine_results(query: str, sql_result: str, rag_result: str) -> str:
    prompt = f"""You are a helpful assistant. A user asked the following question:
{query}

Here is data retrieved from a database:
{sql_result}

Here is information retrieved from documents:
{rag_result}

Combine both pieces of information into a single, coherent answer in the same language as the question."""

    response = llm.invoke(prompt)
    return response.content

def route(query: str) -> str:
    decision = classify_query(query)
    print(f"[Router] Decision: {decision}")
    
    if decision == "SQL":
        return run_sql_agent(query)
    elif decision == "RAG":
        return run_rag_agent(query)
    else:  # HYBRID
        sql_result = run_sql_agent(query)
        rag_result = run_rag_agent(query)
        return combine_results(query, sql_result, rag_result)

        