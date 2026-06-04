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

    # Multilingual model — supports Arabic, English, and mixed documents natively.
    # No translation step needed before retrieval.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

# Create retriever once at module level
_retriever = _create_retriever()
_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

_answer_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

def run_rag_agent(query: str) -> str:
    # Retrieve directly using the original query — no translation needed.
    # paraphrase-multilingual-MiniLM-L12-v2 handles Arabic queries natively.
    docs = _retriever.invoke(query)

    if not docs:
        print("[RAG] No relevant documents retrieved")
        return "لم يتم العثور على سياسة ذات صلة بهذا السؤال."

    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"[RAG] Retrieved context:\n{context}")

    response = (_answer_prompt | _llm).invoke({
        "context": context,
        "question": query
    })
    return response.content
