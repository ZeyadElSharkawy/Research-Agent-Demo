# File: src/agents/fact_checker_agent.py
"""
Fact Checker Agent
Purpose: Verify extracted claims against retrieved documents for accuracy.
"""

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Gemini Setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Missing Google API key.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, handling markdown code blocks and malformed JSON."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try to find JSON pattern
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                return json.loads(json_str)
            except:
                pass
    
    # If all else fails, create a fallback structure
    return {"fallback": {"verification_status": "NOT_SUPPORTED", "confidence": 0, "evidence": "JSON parsing failed", "explanation": "Could not parse verification results"}}

# File: src/agents/fact_checker_agent.py (update the prompt)
def fact_check_claims(claims: list, context_docs: list) -> dict:
    """
    Verify each claim against the provided context documents.
    Returns a dict with verification results for each claim.
    """
    # Prepare context text from documents
    context_text = "\n\n".join([
        doc.page_content if hasattr(doc, 'page_content') else 
        doc.get('content', '') if isinstance(doc, dict) else 
        str(doc)
        for doc in context_docs
    ])
    
    # Prepare claims list for the prompt - use actual claim texts as keys
    claims_dict = {f"Claim {i+1}: {claim}": claim for i, claim in enumerate(claims)}
    claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
    
    prompt = f"""
**TASK**: You are a factual verification expert. Verify each claim against the provided context documents.

**CONTEXT DOCUMENTS**:
{context_text}

**CLAIMS TO VERIFY**:
{claims_text}

**VERIFICATION CRITERIA**:
- SUPPORTED: Claim is clearly and directly supported by evidence in the context
- PARTIALLY_SUPPORTED: Some evidence exists but is incomplete or indirect  
- NOT_SUPPORTED: No evidence found in the context
- CONTRADICTED: Evidence directly contradicts the claim

**REQUIRED OUTPUT FORMAT** (JSON only):
{{
  "The workflow status check script for chatbots resolves L0 issues": {{
    "verification_status": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED|CONTRADICTED",
    "confidence": 0-100,
    "evidence": "Direct quotes from context that support/contradict",
    "explanation": "Brief reasoning for the verdict"
  }},
  "Next claim text here": {{
    "verification_status": "SUPPORTED",
    "confidence": 95,
    "evidence": "Specific quote from document...",
    "explanation": "This is directly stated in the context..."
  }}
}}

**IMPORTANT**: 
- Use the EXACT claim text as keys (copy from the CLAIMS TO VERIFY section)
- Be strict - only mark as SUPPORTED if clear evidence exists
- Include direct quotes as evidence
- Confidence should reflect how strongly the evidence supports the claim

**START YOUR RESPONSE**:
```json
{{
"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        print("🔍 Gemini Response:")
        print(response_text)
        print("-" * 50)
        
        # Extract and parse JSON
        results = extract_json_from_text(response_text)
        
        # Ensure all claims are in the results using actual claim texts
        final_results = {}
        for i, claim in enumerate(claims):
            # Try to find this claim in the results
            claim_found = False
            for result_key, result_value in results.items():
                if claim in result_key or f"Claim {i+1}" in result_key:
                    final_results[claim] = result_value
                    claim_found = True
                    break
            
            if not claim_found:
                # Create default entry for missing claims
                final_results[claim] = {
                    "verification_status": "NOT_SUPPORTED",
                    "confidence": 0,
                    "evidence": "Claim not found in verification results",
                    "explanation": "Verification system did not process this claim"
                }
        
        return final_results
        
    except Exception as e:
        print(f"❌ Fact checking error: {e}")
        # Return fallback results with actual claim texts
        return {
            claim: {
                "verification_status": "NOT_SUPPORTED",
                "confidence": 0,
                "evidence": f"System error: {str(e)}",
                "explanation": "Fact checking failed due to system error"
            }
            for claim in claims
        }

def calculate_overall_confidence(verified_claims: dict) -> float:
    """Calculate overall confidence score from individual claim verifications."""
    if not verified_claims:
        return 0.0
    
    total_confidence = 0
    count = 0
    
    for claim_data in verified_claims.values():
        confidence = claim_data.get('confidence', 0)
        status = claim_data.get('verification_status', 'NOT_SUPPORTED')
        
        # Weight confidence by verification status
        weight = 1.0
        if status == "SUPPORTED":
            weight = 1.0
        elif status == "PARTIALLY_SUPPORTED":
            weight = 0.6
        elif status == "NOT_SUPPORTED":
            weight = 0.1
        elif status == "CONTRADICTED":
            weight = 0.0
        
        total_confidence += confidence * weight
        count += 1
    
    return round(total_confidence / count, 2) if count > 0 else 0.0

if __name__ == "__main__":
    # Test the fact checker with better test data
    print("🧪 Testing Fact Checker Agent...")
    
    test_claims = [
        "Chatbots handle workflow status inquiries automatically.",
        "Expense approvals are done by finance workflow triggers."
    ]
    
    test_docs = [
        {
            "content": "Our chatbot system automatically handles workflow status inquiries and can provide real-time updates on request status. For complex issues, the chatbot escalates to human agents.",
            "metadata": {"source": "Chatbot System Documentation"}
        },
        {
            "content": "Expense approval workflows are triggered automatically when employees submit expense reports. The finance department's workflow system processes these approvals based on company policies.",
            "metadata": {"source": "Finance Process Guide"}
        }
    ]
    
    print("📋 Claims to verify:")
    for i, claim in enumerate(test_claims, 1):
        print(f"  {i}. {claim}")
    
    print("\n📚 Context documents:")
    for doc in test_docs:
        print(f"  - {doc['content'][:100]}...")
    
    print("\n" + "="*60)
    results = fact_check_claims(test_claims, test_docs)
    
    print("\n✅ Fact Check Results:")
    print("="*60)
    
    for claim, result in results.items():
        status = result['verification_status']
        confidence = result['confidence']
        evidence_preview = result['evidence'][:100] + "..." if len(result['evidence']) > 100 else result['evidence']
        
        # Color coding for status
        status_emoji = {
            "SUPPORTED": "✅",
            "PARTIALLY_SUPPORTED": "⚠️", 
            "NOT_SUPPORTED": "❌",
            "CONTRADICTED": "🚫"
        }.get(status, "❓")
        
        print(f"\n{status_emoji} Claim: {claim}")
        print(f"   Status: {status} (Confidence: {confidence}%)")
        print(f"   Evidence: {evidence_preview}")
        print(f"   Explanation: {result['explanation']}")
    
    # Calculate overall confidence
    overall_conf = calculate_overall_confidence(results)
    print(f"\n📊 Overall Confidence Score: {overall_conf}%")