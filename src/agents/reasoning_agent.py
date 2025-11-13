# File: src/agents/reasoning_agent.py
# Purpose: Use Gemini to reason over retrieved context and synthesize grounded answers.

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Initialize Gemini model
# -----------------------------
def get_gemini_reasoning_model():
    """Load Gemini reasoning model from Google Generative AI."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GOOGLE_API_KEY environment variable.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # or gemini-1.5-pro for more reasoning power
        temperature=0.4,
        max_output_tokens=512
    )
    return llm


# -----------------------------
# Main Reasoning Agent Logic
# -----------------------------
def run_reasoning_agent():
    print("Reasoning Agent Ready 🧠 (Gemini-powered)")
    print("Type your query below (or Ctrl+C to exit)\n")

    llm = get_gemini_reasoning_model()

    while True:
        query = input("💬 Enter your query: ").strip()
        if not query:
            continue

        context_text = input("📄 Enter retrieved context (paste text or short doc summary): ").strip()
        if not context_text:
            print("⚠️ Please provide some context for reasoning.\n")
            continue

        prompt = f"""
        You are an intelligent reasoning agent.
        Using the following context, answer the question precisely and clearly.
        Be concise, factual, and grounded only in the given context.

        CONTEXT:
        {context_text}

        QUESTION:
        {query}

        ANSWER:
        """

        print("\n🤔 Synthesizing answer...\n")

        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            print(f"✅ Final Answer:\n{answer}\n")
        except Exception as e:
            print(f"❌ Reasoning failed: {e}\n")


if __name__ == "__main__":
    run_reasoning_agent()
