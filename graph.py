"""
Graph building module for the Multi-Agent Analyst system.
Defines the LangGraph workflow structure.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import State
from agents import (
    orchestrator_node,
    call_forecasting_node,
    call_rag_node,
    call_db_node,
)

# In-memory checkpointer for conversation memory
memory = MemorySaver()


def next_step_router(state: State) -> str:
    """
    Router function to determine the next step based on state.work.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name or END
    """
    work = state.get("work", {})
    
    if "next_forecasting_payload" in work:
        return "Forecasting_Agent"
    if "next_rag_query" in work:
        return "RAG_Agent"
    if "next_db_query" in work:
        return "Database_Agent"
    
    return END


def build_graph() -> StateGraph:
    """
    Builds and compiles the multi-agent workflow graph.
    
    Returns:
        Compiled LangGraph application
    """
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("Orchestrator_Agent", orchestrator_node)
    builder.add_node("RAG_Agent", call_rag_node)
    builder.add_node("Database_Agent", call_db_node)
    builder.add_node("Forecasting_Agent", call_forecasting_node)
    
    # Add edges
    builder.add_edge(START, "Orchestrator_Agent")
    
    builder.add_conditional_edges(
        "Orchestrator_Agent",
        next_step_router,
        {
            "RAG_Agent": "RAG_Agent",
            "Database_Agent": "Database_Agent",
            "Forecasting_Agent": "Forecasting_Agent",
            END: END
        }
    )
    
    # Return edges to orchestrator for multi-step workflows
    builder.add_edge("RAG_Agent", "Orchestrator_Agent")
    builder.add_edge("Database_Agent", "Orchestrator_Agent")
    builder.add_edge("Forecasting_Agent", "Orchestrator_Agent")
    
    return builder.compile(checkpointer=memory)


# Create the compiled application
app = build_graph()
