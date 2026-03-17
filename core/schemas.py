"""
Pydantic schemas for structured outputs from LLM calls.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------
# Forecasting Agent Schemas
# ---------------------------
class ForecastPayload(BaseModel):
    """Payload for the Forecasting Agent."""
    horizon_days: int = Field(description="How many future days to forecast")
    start_date: str | None = Field(
        default=None, 
        description="Start date for forecast in YYYY-MM-DD format. Defaults to today if not specified."
    )


# ---------------------------
# DB Agent Schemas
# ---------------------------
class DBQueryDecision(BaseModel):
    """Structured output for the DB Agent's query parsing."""
    date: str = Field(description="Start date in YYYY-MM-DD format")
    days: int = Field(description="Number of days to query")
    reasoning: str = Field(description="Brief explanation of how parameters were extracted")


# ---------------------------
# Orchestrator Schemas
# ---------------------------
class OrchestratorDecision(BaseModel):
    """
    Structured output for the Orchestrator's routing decision.
    """
    action: Literal["CALL_FORECASTING", "CALL_RAG", "CALL_DB", "FINISH"] = Field(
        description="Next action to take"
    )
    reasoning: str | None = Field(default=None, description="Explanation for the decision")
    
    # For CALL_FORECASTING:
    forecasting_payload: ForecastPayload | None = None
    
    # For CALL_RAG:
    rag_query: str | None = None
    
    # For CALL_DB:
    db_query: str | None = None
    
    # For FINISH:
    final_answer: str | None = None
