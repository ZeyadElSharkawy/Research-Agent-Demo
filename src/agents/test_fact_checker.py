# File: src/agents/test_fact_checker.py
"""
Test script for the fact checker agent
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.agents.fact_checker_agent import fact_check_claims, extract_json_from_text

def test_fact_checker():
    """Test the fact checker with realistic data"""
    
    print("🧪 Testing Fact Checker Agent...")
    
    test_claims = [
        "Chatbots handle workflow status inquiries automatically.",
        "Expense approvals are done by finance workflow triggers.",
        "All documents are processed within 24 hours."  # This one might not be supported
    ]
    
    test_docs = [
        {
            "content": "Our chatbot system automatically handles workflow status inquiries and can provide real-time updates on request status. For complex issues, the chatbot escalates to human agents. The system processes thousands of status checks daily without human intervention.",
            "metadata": {"source": "Chatbot System Documentation v2.1"}
        },
        {
            "content": "Expense approval workflows are triggered automatically when employees submit expense reports. The finance department's workflow system processes these approvals based on company policies and amount thresholds. Manual review is required for expenses over $5000.",
            "metadata": {"source": "Finance Process Guide 2024"}
        },
        {
            "content": "Workflow automation has reduced processing time by 65% across all departments. Chatbots now handle 80% of routine status inquiries, freeing up human agents for complex cases.",
            "metadata": {"source": "Annual Automation Report"}
        }
    ]
    
    print("📋 Claims to verify:")
    for i, claim in enumerate(test_claims, 1):
        print(f"  {i}. {claim}")
    
    print("\n📚 Context documents:")
    for doc in test_docs:
        source = doc['metadata']['source']
        preview = doc['content'][:80] + "..." if len(doc['content']) > 80 else doc['content']
        print(f"  - [{source}]: {preview}")
    
    print("\n" + "="*70)
    print("🔄 Running fact checking...")
    print("="*70)
    
    results = fact_check_claims(test_claims, test_docs)
    
    print("\n✅ FACT CHECK RESULTS:")
    print("="*70)
    
    supported_count = 0
    total_confidence = 0
    
    for claim, result in results.items():
        status = result.get('verification_status', 'UNKNOWN')
        confidence = result.get('confidence', 0)
        evidence = result.get('evidence', 'No evidence provided')
        explanation = result.get('explanation', 'No explanation provided')
        
        # Status emojis and colors
        status_config = {
            "SUPPORTED": ("✅", "🟢"),
            "PARTIALLY_SUPPORTED": ("⚠️", "🟡"), 
            "NOT_SUPPORTED": ("❌", "🔴"),
            "CONTRADICTED": ("🚫", "🔴"),
            "UNKNOWN": ("❓", "⚫")
        }
        
        emoji, color = status_config.get(status, ("❓", "⚫"))
        
        print(f"\n{emoji} {color} CLAIM: {claim}")
        print(f"   📊 Status: {status}")
        print(f"   🎯 Confidence: {confidence}%")
        print(f"   📖 Evidence: {evidence[:150]}..." if len(evidence) > 150 else f"   📖 Evidence: {evidence}")
        print(f"   💡 Explanation: {explanation}")
        
        if status == "SUPPORTED":
            supported_count += 1
        total_confidence += confidence
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"✅ Supported claims: {supported_count}/{len(test_claims)}")
    print(f"🎯 Average confidence: {total_confidence/len(test_claims):.1f}%")
    
    overall_score = (supported_count / len(test_claims)) * (total_confidence / len(test_claims))
    print(f"🏆 Overall verification score: {overall_score:.1f}%")

if __name__ == "__main__":
    test_fact_checker()