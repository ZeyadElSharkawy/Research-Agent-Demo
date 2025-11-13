# File: src/utils/retrieval_utils.py
# Purpose: Load and query Chroma vector store using HuggingFace embeddings

from pathlib import Path
from typing import List
try:
    # Newer packages (preferred)
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # Fallback to older community packages that may still be installed
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The ingest script persists the Chroma DB at the repository root: <repo>/vector_store
VECTOR_DIR = PROJECT_ROOT / "vector_store"


# -------------------------------------------------
# Embedding loader
# -------------------------------------------------
def get_embedding_model():
    """Return the same HuggingFace model used in ingest.py."""
    # Use higher-quality embeddings to improve retrieval accuracy.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def load_vectorstore(persist_directory: str | Path | None = None):
    """
    Load a Chroma vectorstore from disk.

    If persist_directory is None, use the repo-level VECTOR_DIR (same location used by ingest.py).
    """
    if persist_directory is None:
        persist_directory = VECTOR_DIR

    # Ensure the directory is a string
    if isinstance(persist_directory, Path):
        persist_directory = str(persist_directory)

    embedding_model = get_embedding_model()

    # Create Chroma instance pointing at the persisted DB. Use embedding_function name expected by Chroma.
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )

    return vectorstore


# -------------------------------------------------
# Retrieve top-N docs given a query
# -------------------------------------------------
def retrieve_docs(query: str, top_k: int = 3):
    """
    Retrieve top-k documents for a given query using the Chroma vectorstore.
    Compatible with LangChain >= 0.2 retriever API.
    """
    print("Loading vectorstore from:", VECTOR_DIR)
    vectorstore = load_vectorstore(VECTOR_DIR)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Different LangChain versions expose different retriever methods. Try common ones in order.
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
            return docs
        if hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)
            return docs
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
            return docs

        # If none of the above exist, attempt a generic call and expect a list back
        docs = retriever(query)
        return docs
    except Exception as e:
        print("⚠️ Retrieval error:", e)
        return []



# -------------------------------------------------
# Format docs (for display or context passing)
# -------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved documents into a readable string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", f"Document {i}")
        department = doc.metadata.get("department", "Unknown")
        formatted.append(f"### {title} ({department})\n{doc.page_content[:1000]}...\n")

    return "\n\n".join(formatted)


# File: src/utils/retrieval_utils.py (updated)
# Add this function to convert LangChain Documents to the format expected by reranker

def convert_docs_to_reranker_format(docs):
    """Convert LangChain Document objects to the format expected by reranker."""
    converted_docs = []
    for doc in docs:
        converted_doc = {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        converted_docs.append(converted_doc)
    return converted_docs