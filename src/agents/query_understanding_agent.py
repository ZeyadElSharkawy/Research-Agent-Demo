# File: src/agents/query_understanding_agent.py
"""
Purpose:
    The Query Understanding Agent refines and expands user queries before retrieval.
    It uses an LLM (e.g., Gemini or Hugging Face) to:
      - Clarify ambiguous phrasing
      - Expand shorthand or domain terms
      - Optionally classify intent (helpful for downstream agents)

Usage:
    python -m src.agents.query_understanding_agent
"""

from transformers import pipeline


# -------------------------------------------------
# 1. Load the lightweight model
# -------------------------------------------------
# We're using a small open Hugging Face model suitable for local use.
# You can replace this with an API call to Gemini if you prefer consistency.

model_name = "google/flan-t5-base"
llm = pipeline("text2text-generation", model=model_name)

# -------------------------------------------------
# 2. Define prompt template
# -------------------------------------------------
PROMPT_TEMPLATE = """
You are a query understanding agent.
Your job is to reformulate vague or incomplete user queries into a clear, specific, and well-structured search question.

Examples:
- Input: "workflow script bots"
  Output: "Show me diagnostic chatbot scripts that handle workflow status issues."

- Input: "approval delay issue"
  Output: "Explain possible causes and troubleshooting steps for delayed workflow approvals."

Now reformulate this query clearly:
"{query}"
"""


# -------------------------------------------------
# 3. Define main function
# -------------------------------------------------
def reformulate_query(user_query: str) -> str:
    """Uses an LLM to rewrite the user query for clarity and retrieval accuracy."""
    prompt = PROMPT_TEMPLATE.format(query=user_query)
    response = llm(prompt, max_new_tokens=60, do_sample=False)
    refined_query = response[0]["generated_text"].strip()
    return refined_query


# -------------------------------------------------
# 4. Example interactive loop (for testing)
# -------------------------------------------------
if __name__ == "__main__":
    print("Query Understanding Agent Ready ✅")
    print("Type your query below (or Ctrl+C to exit)\n")

    while True:
        user_query = input("💬 Enter your query: ").strip()
        if not user_query:
            print("Please enter a query.\n")
            continue

        print("\n🤔 Reformulating query...")
        refined = reformulate_query(user_query)
        print(f"\n✅ Reformulated Query:\n{refined}\n")
