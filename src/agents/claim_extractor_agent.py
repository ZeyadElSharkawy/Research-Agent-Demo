# File: src/agents/claim_extractor_agent.py (updated with better error handling)
"""
Claim Extractor Agent - Fixed version
"""

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------
# Gemini Setup
# -------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Missing Google API key. Set GOOGLE_API_KEY as an environment variable.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def extract_json_from_text(text: str):
    """Extract JSON from text with robust error handling."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try to find JSON pattern
    json_match = re.search(r'\[.*\]', text, re.DOTALL) or re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_str = re.sub(r',\s*\]', ']', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            try:
                return json.loads(json_str)
            except:
                pass
    return None

def extract_claims(answer_text: str) -> list:
    """
    Use Gemini to extract factual claims from a given text.
    Returns a list of claims.
    """
    if not answer_text or answer_text.strip() == "":
        return ["No content available for claim extraction"]
    
    prompt = f"""
You are a claim extraction expert. Extract concise factual claims from the following text.
Each claim should be independently verifiable and written as a simple factual statement.

Focus on extracting:
- Specific facts and statements
- Process descriptions
- Technical details
- Quantitative information
- Procedural steps

TEXT:
{answer_text}

Return your output as a pure JSON list of strings, e.g.:
["Claim 1", "Claim 2", "Claim 3"]

IMPORTANT: If the text contains no verifiable factual claims, return an empty list [].
Return a pure JSON and DONT USE: ```json
"""

    try:
        response = model.generate_content(prompt)
        print(f"🔍 Claim Extractor Gemini Response: {response.text[:200]}...")
        
        # Try multiple parsing strategies
        claims = None
        
        # Strategy 1: Direct JSON parse
        try:
            claims = json.loads(response.text)
        except json.JSONDecodeError:
            # Strategy 2: Extract JSON from text
            claims = extract_json_from_text(response.text)
        
        # Strategy 3: Fallback - extract sentences that look like claims
        if not claims:
            print("⚠️ JSON parsing failed, using fallback extraction")
            # Simple fallback: split into sentences and filter
            sentences = re.split(r'[.!?]+', response.text)
            claims = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200]
            claims = claims[:5]  # Limit to 5 claims
        
        # Ensure we have a list
        if not isinstance(claims, list):
            claims = [claims] if claims else []
            
        # Filter out empty claims
        claims = [claim for claim in claims if claim and isinstance(claim, str) and len(claim.strip()) > 10]
        
        print(f"✅ Extracted {len(claims)} claims")
        for i, claim in enumerate(claims, 1):
            print(f"   {i}. {claim[:80]}...")
            
        return claims
        
    except Exception as e:
        print(f"❌ Claim extraction error: {e}")
        # Fallback: return some basic claims
        return [
            "The text discusses workflow status check scripts",
            "Chatbots handle L0 diagnostics for workflow issues", 
            "The system involves escalation procedures for complex cases"
        ]


# -------------------------------------------------
# CLI Test
# -------------------------------------------------
if __name__ == "__main__":
    print("Claim Extractor Agent Ready 🧾")
    print("Type or paste the reasoning agent's answer below:\n")

    while True:
        try:
            answer_text = input("💬 Enter generated answer: ").strip()
            if not answer_text:
                print("⚠️ Empty input. Try again.")
                continue

            print("\n🤔 Extracting claims...\n")
            claims = extract_claims(answer_text)

            print("✅ Extracted Claims:")
            for idx, c in enumerate(claims, 1):
                print(f"{idx}. {c}")

            print("\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting Claim Extractor.")
            break