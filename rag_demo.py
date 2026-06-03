import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def _setup_documents():
    with open("policy.txt", "w", encoding="utf-8") as f:
        f.write("""
        Work Week Policy:
        The standard work week runs from Sunday to Thursday.
        Friday and Saturday are the official weekend days.

        Remote Work Policy:
        Employees in Engineering are eligible for 3 remote days per week.
        Employees in HR must be on-site at least 4 days per week.
        Employees in Finance are not eligible for remote work.
        All remote work requests must be approved by the department head.
        """)

def _create_retriever():
    _setup_documents()

    loader = TextLoader("policy.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

# Create retriever once at module level
_retriever = _create_retriever()
_llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

_answer_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

def run_rag_agent(query: str) -> str:
    # Step 1: Translate query to English for retrieval.
    # The embedding model (all-MiniLM-L6-v2) is English-only — passing Arabic
    # text directly produces meaningless embeddings and wrong chunk retrieval.
    translate_prompt = f"""Translate the following query to English. Return only the translated text, nothing else.

Query: {query}"""
    english_query = _llm.invoke(translate_prompt).content.strip()
    print(f"[RAG] Translated query for retrieval: {english_query}")

    # Step 2: Retrieve using English query so embeddings align with policy chunks
    docs = _retriever.invoke(english_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"[RAG] Retrieved context:\n{context}")

    # Step 3: Answer in the original language (Arabic query preserved here)
    response = (_answer_prompt | _llm).invoke({
        "context": context,
        "question": query
    })
    return response.content
