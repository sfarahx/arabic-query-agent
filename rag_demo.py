import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def _setup_documents():
    with open("policy.txt", "w", encoding="utf-8") as f:
        f.write("""
        Remote Work Policy:
        Employees in Engineering are eligible for 3 remote days per week.
        Employees in HR must be on-site at least 4 days per week.
        Employees in Finance are not eligible for remote work.
        All remote work requests must be approved by the department head.
        """)

def _create_chain():
    _setup_documents()

    loader = TextLoader("policy.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,)

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# Create chain once at module level so it's not recreated on every call
_chain = _create_chain()

def run_rag_agent(query: str) -> str:
    response = _chain.invoke(query)
    return response.content

