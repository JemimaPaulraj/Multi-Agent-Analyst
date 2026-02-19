"""
Orchestrator Agent module.
Coordinates between RAG, DB, and Forecasting agents.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import traceable
from langchain_core.messages import AIMessage, SystemMessage

import time
from config import llm, get_logger, log_agent_metrics
from state import State

logger = get_logger("orchestrator")
from schemas import OrchestratorDecision, ForecastPayload
from forecasting import forecasting_agent
from rag import rag_agent
from db import db_agent


# ---------------------------
# Debug Helper
# ---------------------------
def debug_state(node_name: str, state: State) -> None:
    """Log the state for debugging."""
    logger.debug(f"{node_name} | steps={state.get('steps', 0)} | messages={len(state.get('messages', []))} | work={json.dumps(state.get('work', {}), default=str)}")


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


# ---------------------------
# Orchestrator Node
# ---------------------------
@traceable(name="Orchestrator")
def orchestrator_node(state: State) -> dict:
    """Main orchestrator node that decides the next action."""
    debug_state("Orchestrator_Agent", state)
    start_time = time.time()
    
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

    # Get decision from planner
    messages_to_planner = (
        [ORCH_SYSTEM] + 
        state["messages"] + 
        [SystemMessage(content=f"Current state.work JSON:\n{work_json}")]
    )

    response = llm.with_structured_output(OrchestratorDecision, include_raw=True).invoke(messages_to_planner)
    decision = response["parsed"]
    
    # Extract token usage and log per-agent metrics
    raw = response.get("raw")
    tokens_in, tokens_out = 0, 0
    if raw and hasattr(raw, "usage_metadata") and raw.usage_metadata:
        tokens_in = raw.usage_metadata.get("input_tokens", 0)
        tokens_out = raw.usage_metadata.get("output_tokens", 0)
    
    log_agent_metrics("orchestrator", round((time.time() - start_time) * 1000), tokens_in, tokens_out)
    
    # Log orchestrator decision (text only)
    logger.info(f"action={decision.action} | reasoning={decision.reasoning or 'N/A'}")

    # Format debug message
    debug_msg = (
        f"DEBUG Orchestrator Decision:\n"
        f"action={decision.action}\n"
        f"reasoning={decision.reasoning or 'N/A'}\n"
        f"forecasting_payload={decision.forecasting_payload.model_dump() if decision.forecasting_payload else None}\n"
        f"rag_query={decision.rag_query}\n"
        f"db_query={decision.db_query}\n"
    )

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
    start_time = time.time()
    
    work = dict(state.get("work", {}))
    payload = ForecastPayload(**work.get("next_forecasting_payload", {"horizon_days": 2}))
    
    logger.info(f"Forecasting | payload={payload.model_dump()}")
    result = forecasting_agent(payload)
    logger.info(f"Forecasting | result_count={len(result.get('forecast', []))}")
    
    log_agent_metrics("forecasting", round((time.time() - start_time) * 1000))

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
    start_time = time.time()
    
    work = dict(state.get("work", {}))
    query = work.get("next_rag_query", "What information do you need?")
    
    logger.info(f"RAG | query={query}")
    result = rag_agent(query)
    logger.info(f"RAG | sources={len(result.get('sources', []))} | error={result.get('error', False)}")
    
    log_agent_metrics("rag", round((time.time() - start_time) * 1000))

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
    start_time = time.time()
    
    work = dict(state.get("work", {}))
    query = work.get("next_db_query", "Get ticket count from today till 2 days")
    
    logger.info(f"DB | query={query}")
    result = db_agent(query)
    logger.info(f"DB | error={result.get('error', False)}")
    
    log_agent_metrics("db", round((time.time() - start_time) * 1000))

    work["db_result"] = result
    work.pop("next_db_query", None)

    return {
        "messages": [AIMessage(content=f"DEBUG DB Agent returned:\n{json.dumps(result, indent=2)}")],
        "work": work,
        "steps": state.get("steps", 0)
    }
