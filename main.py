"""
FastAPI backend for the Multi-Agent Analyst system.

Observability Strategy:
- CloudWatch Logs: request_id, session_id, user_query, llm_response, errors, stack traces
- CloudWatch Metrics: latency, tokens_in, tokens_out, cost, steps, request_count, error_count
- LangSmith: Full LLM tracing (via @traceable decorator)
"""

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langsmith import traceable

from state import State
from graph import app as langgraph_app
from agents.config import (
    get_logger, 
    log_metrics, 
    estimate_cost,
    generate_request_id,
    log_request,
    log_response,
    log_error
)

logger = get_logger("api")

# Rate limiting (10 requests per 60 seconds per session)
_requests = {}
def rate_limit(user_id: str, limit: int = 10, window: int = 60):
    now = time.time()
    if user_id not in _requests or now - _requests[user_id][0] > window:
        _requests[user_id] = [now, 1]
    elif _requests[user_id][1] >= limit:
        raise HTTPException(429, "Too many requests. Try again later.")
    else:
        _requests[user_id][1] += 1

# FastAPI app
app = FastAPI(
    title="Multi-Agent Analyst API",
    description="API for querying the multi-agent system (RAG, DB, Forecasting)",
    version="1.0.0"
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"  # Session ID for conversation memory


class QueryResponse(BaseModel):
    query: str
    answer: str
    work: dict
    steps: int


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Multi-Agent Analyst API is running"
    )


@traceable(name="Process Query")
def run_multi_agent(query: str, session_id: str) -> dict:
    """Run the multi-agent graph with tracing."""
    init_state: State = {
        "messages": [HumanMessage(content=query)],
        "work": {},
        "steps": 0
    }
    config = {"configurable": {"thread_id": session_id}}
    return langgraph_app.invoke(init_state, config)


@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    """
    Process a user query through the multi-agent system.
    
    Args:
        request: QueryRequest containing the user's query
        
    Returns:
        QueryResponse with the answer and work details
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    rate_limit(request.session_id)
    
    # Generate unique request ID for tracing this specific request
    request_id = generate_request_id()
    start = time.time()
    
    # Log request to CloudWatch Logs (text data for debugging)
    log_request(
        logger=logger,
        request_id=request_id,
        session_id=request.session_id,
        user_query=request.query
    )
    
    try:
        result = run_multi_agent(request.query, request.session_id)
        messages = result.get("messages", [])
        final_answer = messages[-1].content if messages else "No response generated"
        
        # Get token counts from work
        work = result.get("work", {})
        tokens_in = work.get("tokens_in", 0)
        tokens_out = work.get("tokens_out", 0)
        latency_ms = round((time.time() - start) * 1000)
        
        # Log response to CloudWatch Logs (text data for debugging)
        log_response(
            logger=logger,
            request_id=request_id,
            session_id=request.session_id,
            llm_response=final_answer,
            status="success"
        )
        
        # Log numerical metrics to CloudWatch Metrics (for dashboards/alarms)
        log_metrics(
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost(tokens_in, tokens_out),
            steps=result.get("steps", 0)
        )
        
        return QueryResponse(
            query=request.query,
            answer=final_answer,
            work=work,
            steps=result.get("steps", 0)
        )
    
    except Exception as e:
        # Log error to CloudWatch Logs (with stack trace) AND increment error metric
        log_error(
            logger=logger,
            request_id=request_id,
            session_id=request.session_id,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
