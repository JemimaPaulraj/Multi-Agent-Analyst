"""
Tests for Multi-Agent Analyst: API (health) and orchestrator routing.
"""
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root so we can import main, schemas, agents
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from main import app
from schemas import OrchestratorDecision, ForecastPayload
from agents.orchestrator import orchestrator_node

client = TestClient(app)


# ----- API tests -----


def test_health():
    """Health check returns 200 and expected body."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "running" in data["message"].lower()


# ----- Orchestrator routing tests -----


def _make_state(query: str, work: dict | None = None, steps: int = 0) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "work": work or {},
        "steps": steps,
        "request_id": "test-req",
    }


@patch("agents.orchestrator.log_agent_metrics")
@patch("agents.orchestrator.add_event")
@patch("agents.orchestrator.llm")
def test_orchestrator_routes_to_rag(mock_llm, _mock_add_event, _mock_log_metrics):
    """Orchestrator decision CALL_RAG should set next_rag_query in work."""
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": OrchestratorDecision(
            action="CALL_RAG",
            rag_query="What is NET_500?",
            reasoning="Knowledge question.",
        ),
        "raw": None,
    }
    state = _make_state("What is NET_500?")
    out = orchestrator_node(state)
    assert "next_rag_query" in out["work"]
    assert out["work"]["next_rag_query"] == "What is NET_500?"
    assert "next_db_query" not in out["work"]
    assert "next_forecasting_payload" not in out["work"]


@patch("agents.orchestrator.log_agent_metrics")
@patch("agents.orchestrator.add_event")
@patch("agents.orchestrator.llm")
def test_orchestrator_routes_to_db(mock_llm, _mock_add_event, _mock_log_metrics):
    """Orchestrator decision CALL_DB should set next_db_query in work."""
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": OrchestratorDecision(
            action="CALL_DB",
            db_query="Get ticket count from today till 2 days",
            reasoning="Historical data request.",
        ),
        "raw": None,
    }
    state = _make_state("How many tickets in the last 2 days?")
    out = orchestrator_node(state)
    assert "next_db_query" in out["work"]
    assert out["work"]["next_db_query"] == "Get ticket count from today till 2 days"
    assert "next_rag_query" not in out["work"]
    assert "next_forecasting_payload" not in out["work"]


@patch("agents.orchestrator.log_agent_metrics")
@patch("agents.orchestrator.add_event")
@patch("agents.orchestrator.llm")
def test_orchestrator_routes_to_forecasting(mock_llm, _mock_add_event, _mock_log_metrics):
    """Orchestrator decision CALL_FORECASTING should set next_forecasting_payload in work."""
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": OrchestratorDecision(
            action="CALL_FORECASTING",
            forecasting_payload=ForecastPayload(horizon_days=3, start_date=None),
            reasoning="Forecast requested.",
        ),
        "raw": None,
    }
    state = _make_state("Forecast tickets for the next 3 days")
    out = orchestrator_node(state)
    assert "next_forecasting_payload" in out["work"]
    payload = out["work"]["next_forecasting_payload"]
    assert payload["horizon_days"] == 3
    assert "next_rag_query" not in out["work"]
    assert "next_db_query" not in out["work"]


@patch("agents.orchestrator.log_agent_metrics")
@patch("agents.orchestrator.add_event")
@patch("agents.orchestrator.llm")
def test_orchestrator_finish_does_not_set_next_agent(mock_llm, _mock_add_event, _mock_log_metrics):
    """Orchestrator decision FINISH should not add next_rag_query, next_db_query, or next_forecasting_payload."""
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": OrchestratorDecision(
            action="FINISH",
            final_answer="Here is the answer.",
            reasoning="Have enough information.",
        ),
        "raw": None,
    }
    state = _make_state("Summarize", work={"some_key": "value"})
    out = orchestrator_node(state)
    assert "next_rag_query" not in out["work"]
    assert "next_db_query" not in out["work"]
    assert "next_forecasting_payload" not in out["work"]
    assert out["work"].get("some_key") == "value"
