import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from config import config   # ← new

load_dotenv()

def _setup_documents(docs_dir: str, filenames: list):
    """Demo only — writes policy.txt. Not called in production."""
    os.makedirs(docs_dir, exist_ok=True)
    policy_path = os.path.join(docs_dir, filenames[0])
    with open(policy_path, "w", encoding="utf-8") as f:
        f.write("""...""")  # keep your existing content here

def _create_retriever():
    docs_dir = config["documents"]["dir"]              # ← from config
    filenames = config["documents"]["files"]           # ← from config

    if config["documents"].get("seed_demo_data", False): # ← opt-in only
        _setup_documents(docs_dir, filenames)

    # Load all configured files
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    for filename in filenames:
        path = os.path.join(docs_dir, filename)
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        all_chunks.extend(splitter.split_documents(docs))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma.from_documents(all_chunks, embeddings)
    return vectorstore.as_retriever()

# Create retriever once at module level
_retriever = _create_retriever()
_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

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
