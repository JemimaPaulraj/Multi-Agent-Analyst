"""
FastAPI backend for the Multi-Agent Analyst system.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from state import State
from graph import app as langgraph_app

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
    
    try:
        # Initialize state with new message
        init_state: State = {
            "messages": [HumanMessage(content=request.query)],
            "work": {},
            "steps": 0
        }
        
        # Config with session ID for memory
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Run the graph with memory
        result = langgraph_app.invoke(init_state, config)
        
        # Extract final answer from last message
        messages = result.get("messages", [])
        final_answer = messages[-1].content if messages else "No response generated"
        
        return QueryResponse(
            query=request.query,
            answer=final_answer,
            work=result.get("work", {}),
            steps=result.get("steps", 0)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
