"""
Orchestrator Agent module.
Coordinates between RAG, DB, and Forecasting agents.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, SystemMessage

from config import llm
from state import State
from schemas import OrchestratorDecision, ForecastPayload
from forecasting import forecasting_agent
from rag import rag_agent
from db import db_agent


# ---------------------------
# Debug Helper
# ---------------------------
def debug_state(node_name: str, state: State) -> None:
    """Print the full state for debugging."""
    print(f"\n{'='*60}")
    print(f"[{node_name}] STATE DEBUG")
    print(f"{'='*60}")
    print(f"  steps: {state.get('steps', 0)}")
    print(f"  messages count: {len(state.get('messages', []))}")
    print(f"  work: {json.dumps(state.get('work', {}), indent=4, default=str)}")
    print(f"{'='*60}\n")


# ---------------------------
# Orchestrator Configuration
# ---------------------------
ORCH_SYSTEM = SystemMessage(content="""
You are an orchestrator agent that plans step-by-step to answer user queries.

You have access to THREE specialized agents:

1. **Forecasting Agent** (CALL_FORECASTING):
   - Use for future predictions and forecasts
   - Example: "Forecast ticket counts for the next 3 days"
   - Example: "Predict tickets from 2026-02-15 for 5 days"
   - Requires: forecasting_payload with:
     - horizon_days (required): number of days to forecast
     - start_date (optional): start date in YYYY-MM-DD format. Defaults to today if not specified.

2. **RAG Agent** (CALL_RAG):
   - Use for knowledge questions, definitions, explanations
   - Example: "What does NET_500 mean?", "How do I reset a password?"
   - Requires: rag_query as a string

3. **DB Agent** (CALL_DB):
   - Use for historical data, statistics, ticket counts from database
   - Example: "Get ticket count from today till 2 days", "How many tickets last week?"
   - Requires: db_query as a string

Use state.work to see what you already collected from previous agent calls.
Stop when you have enough information and return FINISH with final_answer.
Max 5 steps to prevent infinite loops.
""")

planner = llm.with_structured_output(OrchestratorDecision)


# ---------------------------
# Orchestrator Node
# ---------------------------
def orchestrator_node(state: State) -> dict:
    """Main orchestrator node that decides the next action."""
    debug_state("Orchestrator_Agent", state)
    
    work = state.get("work", {})
    steps = state.get("steps", 0)

    # Prevent infinite loops
    if steps >= 5:
        return {
            "messages": [AIMessage(content="Stopped after 5 steps to avoid looping.")],
            "work": work,
            "steps": steps
        }

    work_json = json.dumps(work, indent=2, default=str)
    print(f"[Orchestrator] What I have already collected (work_json): {work_json}")

    # Get decision from planner
    messages_to_planner = (
        [ORCH_SYSTEM] + 
        state["messages"] + 
        [SystemMessage(content=f"Current state.work JSON:\n{work_json}")]
    )
    print(f"[Orchestrator] Messages to planner: {messages_to_planner}")

    decision = planner.invoke(messages_to_planner)
    print(f"[Orchestrator] Decision: {decision}")

    # Format debug message
    debug_msg = (
        f"DEBUG Orchestrator Decision:\n"
        f"action={decision.action}\n"
        f"reasoning={decision.reasoning or 'N/A'}\n"
        f"forecasting_payload={decision.forecasting_payload.model_dump() if decision.forecasting_payload else None}\n"
        f"rag_query={decision.rag_query}\n"
        f"db_query={decision.db_query}\n"
    )
    print(debug_msg)

    # Handle FINISH action
    if decision.action == "FINISH":
        return {
            "messages": [
                AIMessage(content=debug_msg),
                AIMessage(content=decision.final_answer or "Finished.")
            ],
            "work": work,
            "steps": steps + 1
        }

    # Prepare next action
    new_work = dict(work)
    
    if decision.action == "CALL_FORECASTING":
        new_work["next_forecasting_payload"] = (
            decision.forecasting_payload.model_dump() 
            if decision.forecasting_payload 
            else {"horizon_days": 2}
        )
        
    if decision.action == "CALL_RAG":
        new_work["next_rag_query"] = decision.rag_query or "What information do you need?"
        
    if decision.action == "CALL_DB":
        new_work["next_db_query"] = decision.db_query or "Get ticket count from today till 2 days"

    return {
        "messages": [AIMessage(content=debug_msg)],
        "work": new_work,
        "steps": steps + 1
    }


# ---------------------------
# Agent Call Nodes
# ---------------------------
def call_forecasting_node(state: State) -> dict:
    """Node that calls the Forecasting Agent."""
    debug_state("Forecasting_Agent", state)
    
    work = dict(state.get("work", {}))
    payload = ForecastPayload(**work.get("next_forecasting_payload", {"horizon_days": 2}))
    
    print(f"[Forecasting_Agent] Calling with payload: {payload.model_dump()}")
    result = forecasting_agent(payload)
    print(f"[Forecasting_Agent] Result: {json.dumps(result, indent=2)}")

    work["forecast_result"] = result
    work.pop("next_forecasting_payload", None)

    return {
        "messages": [AIMessage(content=f"DEBUG Forecasting Agent returned:\n{json.dumps(result, indent=2)}")],
        "work": work,
        "steps": state.get("steps", 0)
    }


def call_rag_node(state: State) -> dict:
    """Node that calls the RAG Agent."""
    debug_state("RAG_Agent", state)
    
    work = dict(state.get("work", {}))
    query = work.get("next_rag_query", "What information do you need?")
    
    print(f"[RAG_Agent] Calling with query: {query}")
    result = rag_agent(query)
    print(f"[RAG_Agent] Result: {json.dumps(result, indent=2)}")

    work["rag_result"] = result
    work.pop("next_rag_query", None)

    return {
        "messages": [AIMessage(content=f"DEBUG RAG Agent returned:\n{json.dumps(result, indent=2)}")],
        "work": work,
        "steps": state.get("steps", 0)
    }


def call_db_node(state: State) -> dict:
    """Node that calls the DB Agent."""
    debug_state("Database_Agent", state)
    
    work = dict(state.get("work", {}))
    query = work.get("next_db_query", "Get ticket count from today till 2 days")
    
    print(f"[Database_Agent] Calling with query: {query}")
    result = db_agent(query)
    print(f"[Database_Agent] Result: {json.dumps(result, indent=2)}")

    work["db_result"] = result
    work.pop("next_db_query", None)

    return {
        "messages": [AIMessage(content=f"DEBUG DB Agent returned:\n{json.dumps(result, indent=2)}")],
        "work": work,
        "steps": state.get("steps", 0)
    }
