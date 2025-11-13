# File: src/agents/reranker_agent.py
# Purpose: Re-rank retrieved document chunks by semantic relevance to the query.

import torch
from sentence_transformers import CrossEncoder

# -------------------------------
# Reranker Class
# -------------------------------
class RerankerAgent:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to: {self.device}")

    def rerank(self, query: str, docs: list, top_k: int = 5):
        """
        Reranks the retrieved document chunks based on semantic relevance to the query.
        Each doc in docs is expected to be a dict with 'content' and 'metadata'.
        """
        if not docs:
            print("⚠️ No documents to rerank.")
            return []

        pairs = [(query, d['content']) for d in docs]
        scores = self.model.predict(pairs)

        # Combine docs and scores
        for i, s in enumerate(scores):
            docs[i]["score"] = float(s)

        ranked_docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]

        print(f"✅ Re-ranked top {len(ranked_docs)} documents.")
        return ranked_docs


# -------------------------------
# Interactive Test
# -------------------------------
if __name__ == "__main__":
    reranker = RerankerAgent()

    print("Re-Ranker Agent Ready 🔍")
    query = input("\n💬 Enter query: ")

    # Example dummy docs (replace later with retriever output)
    dummy_docs = [
        {"content": "Chatbots handle workflow status inquiries automatically.", "metadata": {"source": "Chatbot Docs"}},
        {"content": "Expense approvals are done by finance workflow triggers.", "metadata": {"source": "Finance Guide"}},
        {"content": "The chatbot script logs API key checks and opens tickets for unresolved workflows.", "metadata": {"source": "Support KB"}},
    ]

    ranked = reranker.rerank(query, dummy_docs, top_k=2)

    print("\n🏆 Top Ranked Results:")
    for d in ranked:
        print(f"\n[{d['metadata']['source']}] (score={d['score']:.4f})")
        print(d['content'])
