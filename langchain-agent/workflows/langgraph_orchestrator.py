from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from workflows.crew_tasks import execute_research_crew
from tools.infrastructure_tools import LlamaIndexRAGTool
from datetime import datetime
import json

# State schema for the LangGraph workflow
class ResearchState(TypedDict):
    # Input
    original_query: str
    research_mode: str  # "simple", "standard", "comprehensive"
    
    # Workflow control
    current_phase: str
    retry_count: int
    max_retries: int
    
    # Results
    simple_rag_result: str
    crew_result: Dict[str, Any]
    final_output: str
    
    # Quality metrics
    confidence_score: float
    validation_passed: bool
    
    # Metadata
    start_time: str
    end_time: str
    total_duration_seconds: float
    methods_used: List[str]
    error_log: List[str]

def determine_research_mode(state: ResearchState) -> ResearchState:
    """Analyze the query to determine appropriate research approach"""
    
    query = state["original_query"]
    query_lower = query.lower()
    word_count = len(query.split())
    
    # Simple mode indicators
    simple_indicators = [
        "what is", "define", "meaning of", "how to", "when", "where", "who is"
    ]
    
    # Comprehensive mode indicators  
    comprehensive_indicators = [
        "comprehensive analysis", "research report", "detailed study",
        "expert assessment", "multi-faceted", "investigate thoroughly"
    ]
    
    # Determine mode based on query characteristics
    if any(indicator in query_lower for indicator in simple_indicators) and word_count <= 8:
        mode = "simple"
    elif any(indicator in query_lower for indicator in comprehensive_indicators) or word_count > 20:
        mode = "comprehensive"
    else:
        mode = "standard"
    
    return {
        **state,
        "research_mode": mode,
        "current_phase": "execution",
        "start_time": datetime.now().isoformat(),
        "methods_used": []
    }

def execute_simple_research(state: ResearchState) -> ResearchState:
    """Execute simple RAG-based research"""
    
    try:
        rag_tool = LlamaIndexRAGTool()
        result = rag_tool._run(state["original_query"])
        
        # Simple confidence assessment
        confidence = 0.8 if "error" not in result.lower() else 0.3
        
        return {
            **state,
            "simple_rag_result": result,
            "final_output": result,
            "confidence_score": confidence,
            "validation_passed": confidence > 0.6,
            "current_phase": "complete",
            "methods_used": state["methods_used"] + ["simple_rag"]
        }
        
    except Exception as e:
        error_msg = f"Simple research failed: {str(e)}"
        return {
            **state,
            "simple_rag_result": error_msg,
            "final_output": error_msg,
            "confidence_score": 0.0,
            "validation_passed": False,
            "current_phase": "error",
            "error_log": state.get("error_log", []) + [error_msg],
            "methods_used": state["methods_used"] + ["simple_rag_failed"]
        }

def execute_crew_research(state: ResearchState) -> ResearchState:
    """Execute comprehensive CrewAI research"""
    
    try:
        crew_result = execute_research_crew(state["original_query"])
        
        # Extract confidence from validation results (simplified)
        validation_text = crew_result.get("validation", "").lower()
        if "high confidence" in validation_text:
            confidence = 0.9
        elif "medium confidence" in validation_text or "moderate" in validation_text:
            confidence = 0.7
        elif "low confidence" in validation_text:
            confidence = 0.5
        else:
            confidence = 0.75  # Default for crew research
        
        return {
            **state,
            "crew_result": crew_result,
            "final_output": crew_result["final_report"],
            "confidence_score": confidence,
            "validation_passed": confidence > 0.6,
            "current_phase": "complete",
            "methods_used": state["methods_used"] + ["crewai_research"]
        }
        
    except Exception as e:
        error_msg = f"CrewAI research failed: {str(e)}"
        
        # Fallback to simple research if crew fails
        fallback_state = execute_simple_research(state)
        
        return {
            **fallback_state,
            "error_log": state.get("error_log", []) + [error_msg],
            "methods_used": state["methods_used"] + ["crewai_failed", "fallback_simple"]
        }

def quality_check(state: ResearchState) -> ResearchState:
    """Assess research quality and determine if retry is needed"""
    
    confidence = state["confidence_score"]
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    # Quality thresholds
    if confidence >= 0.7:
        validation_passed = True
        next_phase = "finalization"
    elif confidence >= 0.5 and retry_count < max_retries:
        validation_passed = False
        next_phase = "retry"
    else:
        validation_passed = False
        next_phase = "finalization"  # Give up after max retries
    
    return {
        **state,
        "validation_passed": validation_passed,
        "current_phase": next_phase
    }

def retry_with_different_approach(state: ResearchState) -> ResearchState:
    """Retry research with a different approach"""
    
    current_methods = state.get("methods_used", [])
    retry_count = state.get("retry_count", 0) + 1
    
    # If we tried simple and it failed, try crew
    if "simple_rag" in current_methods and "crewai_research" not in current_methods:
        state_copy = {**state, "retry_count": retry_count, "research_mode": "comprehensive"}
        return execute_crew_research(state_copy)
    
    # If we tried crew and it failed, try simple  
    elif "crewai_research" in current_methods and "simple_rag" not in current_methods:
        state_copy = {**state, "retry_count": retry_count, "research_mode": "simple"}
        return execute_simple_research(state_copy)
    
    # If both failed, just finalize
    else:
        return {
            **state,
            "retry_count": retry_count,
            "current_phase": "finalization"
        }

def finalize_research(state: ResearchState) -> ResearchState:
    """Finalize research with metadata and duration"""
    
    start_time = datetime.fromisoformat(state["start_time"])
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Create final summary if needed
    if not state.get("final_output"):
        final_output = "Research could not be completed successfully."
    else:
        final_output = state["final_output"]
    
    return {
        **state,
        "current_phase": "complete",
        "end_time": end_time.isoformat(),
        "total_duration_seconds": duration,
        "final_output": final_output
    }

# Router functions for LangGraph
def route_by_mode(state: ResearchState) -> str:
    """Route to appropriate research method based on mode"""
    mode = state["research_mode"]
    
    if mode == "simple":
        return "simple_research"
    elif mode in ["standard", "comprehensive"]:
        return "crew_research"
    else:
        return "simple_research"  # Default fallback

def route_after_quality_check(state: ResearchState) -> str:
    """Route based on quality check results"""
    phase = state["current_phase"]
    
    if phase == "retry":
        return "retry"
    elif phase == "finalization":
        return "finalize"
    else:
        return "finalize"  # Default

def build_research_workflow():
    """Build the complete LangGraph workflow"""
    
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("determine_mode", determine_research_mode)
    workflow.add_node("simple_research", execute_simple_research)
    workflow.add_node("crew_research", execute_crew_research)
    workflow.add_node("quality_check", quality_check)
    workflow.add_node("retry", retry_with_different_approach)
    workflow.add_node("finalize", finalize_research)
    
    # Set entry point
    workflow.set_entry_point("determine_mode")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "determine_mode",
        route_by_mode,
        {
            "simple_research": "simple_research",
            "crew_research": "crew_research"
        }
    )
    
    # Both research methods go to quality check
    workflow.add_edge("simple_research", "quality_check")
    workflow.add_edge("crew_research", "quality_check") 
    
    # Quality check routing
    workflow.add_conditional_edges(
        "quality_check",
        route_after_quality_check,
        {
            "retry": "retry",
            "finalize": "finalize"
        }
    )
    
    # Retry goes back to quality check
    workflow.add_edge("retry", "quality_check")
    
    # Finalize ends the workflow
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# Main research function
async def orchestrated_research(query: str, max_retries: int = 2) -> Dict[str, Any]:
    """Main function to execute orchestrated research"""
    
    # Build and run workflow
    app = build_research_workflow()
    
    initial_state = {
        "original_query": query,
        "research_mode": "",
        "current_phase": "start",
        "retry_count": 0,
        "max_retries": max_retries,
        "simple_rag_result": "",
        "crew_result": {},
        "final_output": "",
        "confidence_score": 0.0,
        "validation_passed": False,
        "start_time": "",
        "end_time": "",
        "total_duration_seconds": 0.0,
        "methods_used": [],
        "error_log": []
    }
    
    # Execute workflow
    result = app.invoke(initial_state)
    
    return {
        "query": query,
        "final_answer": result["final_output"],
        "confidence": result["confidence_score"],
        "research_mode": result["research_mode"],
        "methods_used": result["methods_used"],
        "duration_seconds": result["total_duration_seconds"],
        "validation_passed": result["validation_passed"],
        "retry_count": result["retry_count"],
        "errors": result.get("error_log", []),
        "metadata": {
            "start_time": result["start_time"],
            "end_time": result["end_time"],
            "workflow_type": "langgraph_crewai_orchestrated"
        }
    }