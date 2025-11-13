"""
File: src/agents/retriever_agent.py
Purpose:
    Acts as a retrieval agent that searches the local Chroma vectorstore
    for the most relevant documents given a user query.

Usage:
    python src/agents/retriever_agent.py
"""

from pathlib import Path
from utils.retrieval_utils import retrieve_docs, format_docs


# -------------------------------------------------
# Main retrieval agent
# -------------------------------------------------
# File: src/agents/retriever_agent.py (ensure it returns documents properly)

def run_retriever(query: str, top_k: int = 5):
    """
    Retrieve relevant documents for a given query.
    Returns list of documents for pipeline use.
    """
    print(f"🔍 Retrieving {top_k} documents for: '{query}'")
    
    try:
        docs = retrieve_docs(query, top_k=top_k)
        if not docs:
            print("⚠️ No relevant documents found.")
            return []
        
        print(f"✅ Retrieved {len(docs)} documents")
        
        # Log document types for debugging
        for i, doc in enumerate(docs):
            doc_type = type(doc).__name__
            has_content = hasattr(doc, 'page_content')
            has_metadata = hasattr(doc, 'metadata')
            print(f"  Document {i+1}: {doc_type}, has_content: {has_content}, has_metadata: {has_metadata}")
            if has_content:
                print(f"    Preview: {doc.page_content[:100]}...")
        
        return docs
        
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return []


# -------------------------------------------------
# CLI entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    VECTOR_DIR = PROJECT_ROOT / "src" / "vector_store"

    if not VECTOR_DIR.exists():
        print(f"❌ Vector store not found at {VECTOR_DIR}. Run 'ingest.py' first.")
    else:
        print("Retriever Agent Ready ✅")
        print("Type your query below (or press Ctrl+C to exit).")

        while True:
            try:
                user_query = input("\n💬 Enter your query: ").strip()
                if not user_query:
                    continue
                run_retriever(user_query)
            except KeyboardInterrupt:
                print("\n👋 Exiting retriever agent.")
                break
