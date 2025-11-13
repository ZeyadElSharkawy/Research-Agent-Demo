# File: src/graph/research_graph.py (updated)
"""
LangGraph Research Pipeline - Fixed version
"""

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import agents and utils
from agents.query_understanding_agent import reformulate_query
from agents.retriever_agent import run_retriever
from agents.reranker_agent import RerankerAgent
from agents.reasoning_agent import get_gemini_reasoning_model
from agents.claim_extractor_agent import extract_claims
from agents.fact_checker_agent import fact_check_claims
from agents.final_answer_agent import generate_final_answer
from utils.retrieval_utils import convert_docs_to_reranker_format

# Import logger for Streamlit integration
try:
    from utils.streamlit_logger import get_logger
    USE_LOGGER = True
except ImportError:
    USE_LOGGER = False

# Define state structure
class ResearchState(TypedDict):
    original_query: str
    refined_query: str
    retrieved_docs: List[Any]
    reranked_docs: List[Any]
    draft_answer: str
    extracted_claims: List[str]
    verified_claims: Dict[str, Any]
    final_answer: Dict[str, Any]
    error: str

# Initialize agents
reranker = RerankerAgent()

def query_understanding_node(state: ResearchState) -> ResearchState:
    """Node 1: Understand and refine the user query"""
    try:
        print("🔍 Query Understanding Agent working...")
        refined = reformulate_query(state["original_query"])
        return {**state, "refined_query": refined}
    except Exception as e:
        return {**state, "error": f"Query understanding failed: {str(e)}"}

def retrieval_node(state: ResearchState) -> ResearchState:
    """Node 2: Retrieve relevant documents"""
    try:
        print("📚 Retrieval Agent working...")
        # Use the refined query for retrieval
        query = state.get("refined_query", state["original_query"])
        docs = run_retriever(query, top_k=5)  # Get more docs for reranking
        return {**state, "retrieved_docs": docs}
    except Exception as e:
        return {**state, "error": f"Retrieval failed: {str(e)}"}

def reranker_node(state: ResearchState) -> ResearchState:
    """Node 3: Rerank retrieved documents"""
    try:
        print("🎯 Reranker Agent working...")
        query = state.get("refined_query", state["original_query"])
        
        # Convert LangChain Documents to reranker format
        if state["retrieved_docs"]:
            docs_for_reranking = convert_docs_to_reranker_format(state["retrieved_docs"])
            ranked_docs = reranker.rerank(query, docs_for_reranking, top_k=3)
            
            # Convert back to include original document objects for downstream use
            # We'll store both the ranked content and original docs
            final_ranked_docs = []
            for ranked_doc in ranked_docs:
                # Find the original document that matches this content
                for original_doc in state["retrieved_docs"]:
                    if original_doc.page_content == ranked_doc["content"]:
                        final_ranked_docs.append(original_doc)
                        break
            
            return {**state, "reranked_docs": final_ranked_docs}
        else:
            print("⚠️ No documents to rerank.")
            return {**state, "reranked_docs": []}
            
    except Exception as e:
        return {**state, "error": f"Reranking failed: {str(e)}"}

def reasoning_node(state: ResearchState) -> ResearchState:
    """Node 4: Generate draft answer using reasoning"""
    try:
        print("🧠 Reasoning Agent working...")
        query = state.get("refined_query", state["original_query"])
        
        if not state["reranked_docs"]:
            # If no reranked docs, use retrieved docs as fallback
            context_docs = state["retrieved_docs"]
            print("⚠️ Using retrieved docs as fallback for reasoning")
        else:
            context_docs = state["reranked_docs"]
        
        if not context_docs:
            state["draft_answer"] = "No relevant documents found to answer the query."
            return state
        
        # Prepare context from docs
        context_text = "\n\n".join([
            doc.page_content if hasattr(doc, 'page_content') else doc.get('content', '') 
            for doc in context_docs
        ])
        
        # Use reasoning agent logic
        llm = get_gemini_reasoning_model()
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
        
        response = llm.invoke(prompt)
        draft_answer = response.content if hasattr(response, 'content') else str(response)
        
        return {**state, "draft_answer": draft_answer}
    except Exception as e:
        return {**state, "error": f"Reasoning failed: {str(e)}"}

def claim_extraction_node(state: ResearchState) -> ResearchState:
    """Node 5: Extract factual claims from draft answer"""
    try:
        print("📋 Claim Extractor working...")
        claims = extract_claims(state["draft_answer"])
        return {**state, "extracted_claims": claims}
    except Exception as e:
        return {**state, "error": f"Claim extraction failed: {str(e)}"}

def fact_checking_node(state: ResearchState) -> ResearchState:
    """Node 6: Verify claims against documents"""
    try:
        print("✅ Fact Checker working...")
        
        # Use reranked docs if available, otherwise use retrieved docs
        if state["reranked_docs"]:
            context_docs = state["reranked_docs"]
        else:
            context_docs = state["retrieved_docs"]
            print("⚠️ Using retrieved docs for fact checking (no reranked docs)")
        
        if not context_docs:
            # If no documents, create a fallback verification
            fallback_claims = {}
            for claim in state["extracted_claims"]:
                fallback_claims[claim] = {
                    "verification_status": "NOT_SUPPORTED",
                    "confidence": 0,
                    "evidence": "No documents available for verification",
                    "explanation": "Cannot verify claims without source documents"
                }
            return {**state, "verified_claims": fallback_claims}
        
        # Convert to fact checker format
        docs_for_fact_checking = convert_docs_to_reranker_format(context_docs)
        verified = fact_check_claims(state["extracted_claims"], docs_for_fact_checking)
        return {**state, "verified_claims": verified}
    except Exception as e:
        return {**state, "error": f"Fact checking failed: {str(e)}"}

def final_answer_node(state: ResearchState) -> ResearchState:
    """Node 7: Generate final verified answer"""
    try:
        print("🎯 Final Answer Agent working...")
        
        # Use reranked docs if available, otherwise use retrieved docs
        if state["reranked_docs"]:
            context_docs = state["reranked_docs"]
        else:
            context_docs = state["retrieved_docs"]
        
        final = generate_final_answer(
            state["original_query"],
            state["verified_claims"],
            context_docs
        )
        return {**state, "final_answer": final}
    except Exception as e:
        return {**state, "error": f"Final answer generation failed: {str(e)}"}

def error_node(state: ResearchState) -> ResearchState:
    """Handle errors in the pipeline"""
    error_msg = state.get("error", "Unknown error occurred")
    print(f"❌ Pipeline error: {error_msg}")
    return {
        **state,
        "final_answer": {
            "final_answer": f"❌ Pipeline Error: {error_msg}",
            "confidence_score": 0,
            "verified_sources": [],
            "limitations": "System error prevented complete processing"
        }
    }

def build_research_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline"""
    
    # Create graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("query_understanding", query_understanding_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("reranking", reranker_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("claim_extraction", claim_extraction_node)
    workflow.add_node("fact_checking", fact_checking_node)
    workflow.add_node("final_answer", final_answer_node)
    workflow.add_node("error_handler", error_node)
    
    # Define flow
    workflow.set_entry_point("query_understanding")
    
    workflow.add_edge("query_understanding", "retrieval")
    workflow.add_edge("retrieval", "reranking")
    workflow.add_edge("reranking", "reasoning")
    workflow.add_edge("reasoning", "claim_extraction")
    workflow.add_edge("claim_extraction", "fact_checking")
    workflow.add_edge("fact_checking", "final_answer")
    workflow.add_edge("final_answer", END)
    
    # Add error handling - if any node sets an error, go to error handler
    workflow.add_conditional_edges(
        "query_understanding",
        lambda state: "error_handler" if state.get("error") else "retrieval"
    )
    workflow.add_conditional_edges(
        "retrieval", 
        lambda state: "error_handler" if state.get("error") else "reranking"
    )
    workflow.add_conditional_edges(
        "reranking",
        lambda state: "error_handler" if state.get("error") else "reasoning"
    )
    workflow.add_conditional_edges(
        "reasoning",
        lambda state: "error_handler" if state.get("error") else "claim_extraction"
    )
    workflow.add_conditional_edges(
        "claim_extraction",
        lambda state: "error_handler" if state.get("error") else "fact_checking"
    )
    workflow.add_conditional_edges(
        "fact_checking",
        lambda state: "error_handler" if state.get("error") else "final_answer"
    )
    
    return workflow.compile()

# Global graph instance
research_graph = build_research_graph()

def run_research_pipeline(query: str) -> Dict[str, Any]:
    """
    Run the complete multi-agent research pipeline.
    """
    print(f"🚀 Starting research pipeline for: {query}")
    
    initial_state = ResearchState(
        original_query=query,
        refined_query="",
        retrieved_docs=[],
        reranked_docs=[],
        draft_answer="",
        extracted_claims=[],
        verified_claims={},
        final_answer={},
        error=""
    )
    
    try:
        # Capture logs if logger is available
        if USE_LOGGER:
            logger = get_logger()
            # Use real-time mode for agent status updates (lightweight callback)
            with logger.capture_logs(real_time=True):
                result = research_graph.invoke(initial_state)
            # Add logs to result
            result["logs"] = logger.get_logs()
        else:
            result = research_graph.invoke(initial_state)
            result["logs"] = []
        
        return result
    except Exception as e:
        print(f"❌ System Error: {str(e)}")
        return {
            "final_answer": {
                "final_answer": f"❌ System Error: {str(e)}",
                "confidence_score": 0,
                "verified_sources": [],
                "limitations": "Critical system failure"
            },
            "error": str(e),
            "logs": get_logger().get_logs() if USE_LOGGER else []
        }

if __name__ == "__main__":
    # Test the complete pipeline
    test_query = "How many completed EKYC Leads did we accomplish in the 3rd Quarter for year 2025?"
    result = run_research_pipeline(test_query)
    
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    if "final_answer" in result:
        final = result["final_answer"]
        print(f"Answer: {final['final_answer']}")
        print(f"Confidence: {final['confidence_score']}%")
        print(f"Sources: {final['verified_sources']}")
    else:
        print("No final answer generated")
        print(f"Result keys: {result.keys()}")