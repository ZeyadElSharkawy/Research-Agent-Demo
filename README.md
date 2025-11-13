# NexaCore AI Research Assistant

This repository contains a multi-agent Retrieval-Augmented-Generation (RAG) research assistant that:

- Accepts user queries and returns answers grounded in uploaded company documents.
- Stores embeddings in a local Chroma DB for semantic search.
- Allows uploading documents (PDF, Office 365 formats: DOCX/DOC/XLSX/XLS, CSV, TXT, MD) which are processed and added to the knowledge base.

This Streamlit app is ready to run locally and can be deployed to Streamlit Cloud.

## Quickstart (local)

1. Create a Python 3.10+ venv and activate it.

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Process and ingest the provided document (if needed):

```powershell
# Convert raw files into processed text + metadata
python src/utils/load_docs.py

# Build the vector store from processed documents
python src/utils/ingest.py
```

4. Run the research pipeline directly (optional):

```powershell
python src/graph/research_graph.py
```

5. Run the Streamlit UI:

```powershell
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app and point it to the repository and the branch to deploy.
3. Ensure `requirements.txt` is present (this repo includes one). Streamlit Cloud will install dependencies automatically.

Notes:

- The app may download embedding models (HuggingFace) at runtime which can be large. For faster startup, pre-build the vector store locally and push the `vector_store/` folder (careful: large) or use a small embedding model.
- For production usage, move large models to a server with sufficient RAM/CPU or use a managed embedding provider.

## Project Structure (relevant files)

- `app.py` — Streamlit UI and upload flow
- `src/utils/load_docs.py` — Extracts text + metadata from uploaded documents
- `src/utils/ingest.py` — Builds embeddings and persists Chroma vector store
- `src/graph/research_graph.py` — Orchestrates the multi-agent pipeline
- `src/utils/retrieval_utils.py` — Loads vector store and retrieves relevant documents

If you want, I can add a small smoke test that confirms retrieval returns at least one document after ingest. Enjoy!
