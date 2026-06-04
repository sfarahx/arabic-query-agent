import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import sqlite3

load_dotenv()

def _setup_database():
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER
        )
    """)

    cursor.executemany("INSERT OR IGNORE INTO employees VALUES (?,?,?,?)", [
        (1, "Ahmed Al-Mansouri", "HR", 8000),
        (2, "Sara Al-Khalidi", "Engineering", 12000),
        (3, "Omar Farouk", "HR", 7500),
        (4, "Layla Hassan", "Engineering", 11000),
        (5, "Khalid Al-Rashid", "Finance", 9000),
    ])

    conn.commit()
    conn.close()

_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def _create_agent():
    _setup_database()
    db = SQLDatabase.from_uri("sqlite:///company.db")
    return create_sql_agent(llm=_llm, db=db, verbose=False, handle_parsing_errors=True)

# Create agent once at module level so it's not recreated on every call
_agent = _create_agent()

def run_sql_agent(query: str) -> str:
    try:
        transliterate_prompt = f"""Translate the following query to English.
- Translate all text including proper nouns and names
- For names, use their standard English transliteration
- Preserve the original meaning exactly
- Return only the translated query, nothing else

Query: {query}"""

        transliterated = _llm.invoke(transliterate_prompt).content.strip()
        print(f"[SQL] Transliterated query: {transliterated}")

        response = _agent.invoke(transliterated)
        print(f"[SQL] Raw response: {response}")

        # Safely extract output — agent response structure can vary
        if isinstance(response, dict):
            output = response.get("output", "").strip()
        else:
            output = str(response).strip()

        if not output:
            return "لم يتم العثور على نتائج في قاعدة البيانات."

        return output

    except Exception as e:
        print(f"[SQL] Exception: {type(e).__name__}: {e}")
        return "لم يتم العثور على نتائج مطابقة في قاعدة البيانات."