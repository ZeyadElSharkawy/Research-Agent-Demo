# File: src/agents/final_answer_agent.py (fixed version)
"""
Final Answer Agent - Fixed to properly handle fact checker results
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
model = genai.GenerativeModel("gemini-2.0-flash-lite")

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
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                return json.loads(json_str)
            except:
                pass
    return {}

def generate_final_answer(original_query: str, verified_claims: dict, context_docs: list) -> dict:
    """
    Generate a final answer incorporating verified claims and citations.
    Returns a dict with the answer and metadata.
    """
    
    print(f"🔍 Final Answer Agent received {len(verified_claims)} verified claims")
    
    # Debug: Print what we actually received
    print("📊 Verified claims structure:")
    for key, value in list(verified_claims.items())[:3]:  # Show first 3
        print(f"  Key: {key[:50]}...")
        print(f"  Value type: {type(value)}")
        if isinstance(value, dict):
            print(f"  Status: {value.get('verification_status', 'MISSING')}")
    
    # Prepare verification summary
    verification_summary = []
    supported_claims = []
    partially_supported_claims = []
    not_supported_claims = []
    contradicted_claims = []
    
    # Handle different claim key formats (claim_1_text vs actual claim text)
    for claim_key, verification in verified_claims.items():
        # Extract the actual claim text from the key or use the key itself
        actual_claim = claim_key
        if claim_key.startswith("claim_") and "_text" in claim_key:
            # This is a formatted key like "claim_1_text", we need to map it back to the original claim
            # For now, we'll use a simplified approach
            actual_claim = f"Claim {claim_key}"
        
        status = verification.get('verification_status', 'UNKNOWN')
        confidence = verification.get('confidence', 0)
        
        verification_summary.append(f"- {actual_claim} [{status}, Confidence: {confidence}%]")
        
        if status == 'SUPPORTED':
            supported_claims.append(actual_claim)
        elif status == 'PARTIALLY_SUPPORTED':
            partially_supported_claims.append(actual_claim)
        elif status == 'NOT_SUPPORTED':
            not_supported_claims.append(actual_claim)
        elif status == 'CONTRADICTED':
            contradicted_claims.append(actual_claim)
    
    print(f"📊 Claim breakdown: {len(supported_claims)} supported, {len(partially_supported_claims)} partial, {len(not_supported_claims)} not supported, {len(contradicted_claims)} contradicted")
    
    # Prepare context with sources
    context_with_sources = []
    sources_used = set()
    for i, doc in enumerate(context_docs):
        source = ""
        content = ""
        
        if hasattr(doc, 'metadata'):
            source = doc.metadata.get('source', f'Document {i+1}')
            content = doc.page_content
        elif isinstance(doc, dict):
            source = doc.get('metadata', {}).get('source', f'Document {i+1}')
            content = doc.get('content', '')
        else:
            source = f'Document {i+1}'
            content = str(doc)
            
        sources_used.add(source)
        context_with_sources.append(f"[Source: {source}]\n{content}")
    
    # Calculate overall confidence
    overall_confidence = calculate_overall_confidence(verified_claims)
    
    # If we have mostly supported claims, generate a positive answer
    if len(supported_claims) > 0:
        prompt = f"""
**TASK**: You are a final answer synthesizer. Create a comprehensive, well-structured answer to the user's query using the verified claims and context provided.

**ORIGINAL QUERY**: {original_query}

**VERIFICATION RESULTS**:
- SUPPORTED Claims: {len(supported_claims)}
- PARTIALLY SUPPORTED: {len(partially_supported_claims)} 
- NOT SUPPORTED: {len(not_supported_claims)}
- CONTRADICTED: {len(contradicted_claims)}

**SUPPORTING CONTEXT DOCUMENTS**:
{chr(10).join(context_with_sources)}

**INSTRUCTIONS**:
1. Create a clear, well-structured answer focusing on the SUPPORTED claims
2. Include specific details and examples from the context
3. Cite your sources using [Source: ...] notation
4. Be honest about any limitations or uncertainties
5. The overall confidence in this answer is {overall_confidence}%

**FINAL ANSWER**:
"""
    else:
        # If no supported claims, be honest about limitations
        prompt = f"""
**TASK**: You are a final answer synthesizer. The fact-checking process found limited support for claims related to the user's query.

**ORIGINAL QUERY**: {original_query}

**VERIFICATION RESULTS**:
- SUPPORTED Claims: {len(supported_claims)}
- PARTIALLY SUPPORTED: {len(partially_supported_claims)}
- NOT SUPPORTED: {len(not_supported_claims)} 
- CONTRADICTED: {len(contradicted_claims)}

**CONTEXT DOCUMENTS**:
{chr(10).join(context_with_sources)}

**INSTRUCTIONS**:
1. Honestly state that limited verified information was found
2. Mention what the documents do contain that might be related
3. Suggest what additional information would be needed
4. The overall confidence in available information is {overall_confidence}%

**FINAL ANSWER**:
"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Create result structure
        result = {
            "final_answer": response_text,
            "confidence_score": overall_confidence,
            "verified_sources": list(sources_used),
            "limitations": f"Based on verification: {len(supported_claims)} supported, {len(not_supported_claims)} not supported claims"
        }
        
        # Add claim breakdown for transparency
        result["claim_breakdown"] = {
            "supported": len(supported_claims),
            "partially_supported": len(partially_supported_claims),
            "not_supported": len(not_supported_claims),
            "contradicted": len(contradicted_claims)
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Final answer generation error: {e}")
        # Fallback structure
        return {
            "final_answer": f"I encountered an error while generating the final answer: {str(e)}",
            "confidence_score": overall_confidence,
            "verified_sources": list(sources_used),
            "limitations": "System error during answer generation",
            "claim_breakdown": {
                "supported": len(supported_claims),
                "partially_supported": len(partially_supported_claims), 
                "not_supported": len(not_supported_claims),
                "contradicted": len(contradicted_claims)
            }
        }

def calculate_overall_confidence(verified_claims: dict) -> float:
    """Calculate overall confidence score from individual claim verifications."""
    if not verified_claims:
        return 0.0
    
    total_confidence = 0
    valid_claims = 0
    
    for claim_key, verification in verified_claims.items():
        confidence = verification.get('confidence', 0)
        status = verification.get('verification_status', 'NOT_SUPPORTED')
        
        # Only count claims that were actually verified
        if status in ['SUPPORTED', 'PARTIALLY_SUPPORTED', 'NOT_SUPPORTED', 'CONTRADICTED']:
            # For SUPPORTED claims, use the confidence directly
            if status == 'SUPPORTED':
                total_confidence += confidence
            # For PARTIALLY_SUPPORTED, use reduced confidence
            elif status == 'PARTIALLY_SUPPORTED':
                total_confidence += confidence * 0.7
            # For NOT_SUPPORTED and CONTRADICTED, they don't contribute positively
            valid_claims += 1
    
    if valid_claims == 0:
        return 0.0
    
    return min(100.0, round(total_confidence / valid_claims, 2))

if __name__ == "__main__":
    # Test with the actual fact checker results structure
    print("🧪 Testing Final Answer Agent with actual structure...")
    
    test_verified_claims = {
        "claim_1_text": {
            "verification_status": "SUPPORTED", 
            "confidence": 95,
            "evidence": "Test evidence",
            "explanation": "Test explanation"
        },
        "claim_2_text": {
            "verification_status": "SUPPORTED",
            "confidence": 100, 
            "evidence": "Test evidence",
            "explanation": "Test explanation"
        }
    }
    
    test_docs = [{"content": "Test content", "metadata": {"source": "Test Source"}}]
    
    result = generate_final_answer("Test query", test_verified_claims, test_docs)
    print(f"✅ Test result: {result['confidence_score']}% confidence")