"""
Orchestrator Agent module.
Coordinates between RAG, DB, and Forecasting agents.
"""

import json
import time

from langsmith import traceable
from langchain_core.messages import AIMessage, SystemMessage

from agents.config import llm, get_logger, log_agent_metrics, add_event, estimate_cost
from core.state import State
from core.schemas import OrchestratorDecision, ForecastPayload
from agents.forecasting import forecasting_agent
from agents.rag import rag_agent
from agents.db import db_agent

logger = get_logger("orchestrator")


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


# ------------------------------------------------------------------------------------------------#
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
    
    latency_sec = round((time.time() - start_time), 3)
    cost_usd = estimate_cost(tokens_in, tokens_out)
    
    # Accumulate tokens in work for total tracking
    work["tokens_in"] = work.get("tokens_in", 0) + tokens_in
    work["tokens_out"] = work.get("tokens_out", 0) + tokens_out
    
    log_agent_metrics("orchestrator", latency_sec, tokens_in, tokens_out)
    
    # Add to request context (single JSON blob) with per-agent metrics
    request_id = state.get("request_id")
    if request_id:
        event_data = {
            "agent": "orchestrator", 
            "orchestrator_decision": decision.action,
            "agent_metrics": {
                "latency_sec": latency_sec,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd
            }
        }
        if decision.rag_query:
            event_data["rag_query"] = decision.rag_query
        if decision.db_query:
            event_data["db_query"] = decision.db_query
        if decision.forecasting_payload:
            event_data["forecasting_payload"] = decision.forecasting_payload.model_dump()
        add_event(request_id, **event_data)

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
    
    result = forecasting_agent(payload)
    
    latency_sec = round((time.time() - start_time), 3)
    log_agent_metrics("forecasting", latency_sec)
    
    # Add to request context (single JSON blob) with per-agent metrics
    request_id = state.get("request_id")
    if request_id:
        add_event(
            request_id, 
            agent="forecasting", 
            forecast_days=len(result.get("forecast", [])),
            agent_metrics={
                "latency_sec": latency_sec,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0
            }
        )

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
    
    result = rag_agent(query)
    
    # Accumulate RAG tokens in work (only if not cached)
    rag_tokens_in = result.get("tokens_in", 0)
    rag_tokens_out = result.get("tokens_out", 0)
    work["tokens_in"] = work.get("tokens_in", 0) + rag_tokens_in
    work["tokens_out"] = work.get("tokens_out", 0) + rag_tokens_out
    
    latency_sec = round((time.time() - start_time), 3)
    cost_usd = estimate_cost(rag_tokens_in, rag_tokens_out)
    
    log_agent_metrics("rag", latency_sec, rag_tokens_in, rag_tokens_out)
    
    # Add to request context (single JSON blob) with per-agent metrics
    request_id = state.get("request_id")
    if request_id:
        event_data = {
            "agent": "rag",
            "cache_hit": result.get("cached", False),
            "sources": result.get("sources", []),
            "agent_metrics": {
                "latency_sec": latency_sec,
                "tokens_in": rag_tokens_in,
                "tokens_out": rag_tokens_out,
                "cost_usd": cost_usd
            }
        }
        if result.get("cache_similarity"):
            event_data["cache_similarity"] = result.get("cache_similarity")
        add_event(request_id, **event_data)

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
    
    result = db_agent(query)
    
    latency_sec = round((time.time() - start_time), 3)
    log_agent_metrics("db", latency_sec)
    
    # Add to request context (single JSON blob) with per-agent metrics
    request_id = state.get("request_id")
    if request_id:
        add_event(
            request_id, 
            agent="db", 
            db_error=result.get("error", False),
            agent_metrics={
                "latency_sec": latency_sec,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0
            }
        )

    work["db_result"] = result
    work.pop("next_db_query", None)

    return {
        "messages": [AIMessage(content=f"DEBUG DB Agent returned:\n{json.dumps(result, indent=2)}")],
        "work": work,
        "steps": state.get("steps", 0)
    }
