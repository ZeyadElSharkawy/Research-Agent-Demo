"""
File: src/ingest.py
Purpose:
    Load all processed text documents from processed_docs/,
    generate embeddings, and store them in a vector database (Chroma).

Usage:
    pip install langchain chromadb sentence-transformers tqdm
    python src/ingest.py

Output:
    vector_store/
        chroma.sqlite
        index/
"""

import os
import json
import csv
from pathlib import Path
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# Path setup (relative, repo-safe)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "processed_docs"
VECTOR_DIR = PROJECT_ROOT / "vector_store"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"


# -------------------------------------------------
# Embedding model
# -------------------------------------------------
def get_embedding_model():
    """
    Return a HuggingFace embedding model.
    You can replace with OpenAIEmbeddings if you have an API key.
    """
    # Use a higher-quality embedding model. Change model_name if you prefer a different tradeoff.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# -------------------------------------------------
# Load single processed document
# -------------------------------------------------
def load_single_document(title: str = "new_duck_exec_onboarding_customer_funnel_from0106to3110"):
    """
    Loads a single processed document by title from processed_docs/
    Returns a list with one LangChain Document.
    """
    documents = []

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Missing metadata.csv in {PROCESSED_DIR}")

    with open(METADATA_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("title") != title:
                continue
                
            text_path = PROJECT_ROOT / row["processed_text_path"]
            meta_path = PROJECT_ROOT / row["processed_meta_path"]

            if not text_path.exists():
                print(f"[skip] Missing text file: {text_path}")
                continue

            try:
                text = text_path.read_text(encoding="utf-8")
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                metadata.pop("processed_text_path", None)
                metadata.pop("processed_meta_path", None)

                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
            except Exception as e:
                print(f"[error] Failed loading {text_path}: {e}")
                continue

    if not documents:
        raise FileNotFoundError(f"No document found with title: {title}")
    
    print(f"Loaded 1 document: {title}")
    return documents


# -------------------------------------------------
# Split large documents into chunks
# -------------------------------------------------
def chunk_documents(documents, chunk_size=1000, overlap=200):
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    split_docs = splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs


# -------------------------------------------------
# Build and persist vector store
# -------------------------------------------------
def build_vector_store(documents):
    """
    Create embeddings and persist them in a local Chroma DB.
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    embedding_model = get_embedding_model()
    print("Generating embeddings and creating vector store...")

    # Create vectorstore from documents. New langchain-chroma may persist automatically.
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding_function=embedding_model,
            persist_directory=str(VECTOR_DIR),
        )
    except TypeError:
        # Fallback for versions that expect 'embedding' kwarg
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=str(VECTOR_DIR),
        )

    # Older langchain-chroma used .persist(); newer versions persist on construction.
    if hasattr(vectorstore, "persist"):
        try:
            vectorstore.persist()
        except Exception:
            pass

    print(f"Vector store successfully saved to {VECTOR_DIR}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build Chroma vector store from processed documents or a single title")
    parser.add_argument("--title", "-t", help="Title of the processed document to build embeddings for (from metadata.csv). If omitted, all processed docs are used when supported.")
    parser.add_argument("--all", action="store_true", help="Force building embeddings for all processed documents listed in metadata.csv (not implemented for single-file mode)")
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed dir: {PROCESSED_DIR}")
    print(f"Vector store dir: {VECTOR_DIR}\n")

    # Load the specified document title or default title
    title = args.title or "new_duck_exec_onboarding_customer_funnel_from0106to3110"
    try:
        docs = load_single_document(title=title)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run load_docs.py first to process the document or check the --title value.")
        return

    if not docs:
        print("No documents found.")
        return

    chunks = chunk_documents(docs)
    build_vector_store(chunks)


if __name__ == "__main__":
    main()
