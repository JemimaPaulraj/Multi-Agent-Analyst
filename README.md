# Multi-Agent Analyst

A modular multi-agent system built with LangGraph that orchestrates between specialized agents for RAG, database querying, and forecasting tasks.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│                                                             │
│  Decides which agent to call based on the user's query      │
│  - CALL_RAG         → RAG Agent                             │
│  - CALL_DB          → DB Agent                              │
│  - CALL_FORECASTING → Forecasting Agent                     │
│  - FINISH           → Return final answer                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   RAG AGENT   │ │   DB AGENT    │ │  FORECASTING  │
│               │ │               │ │     AGENT     │
│ - Knowledge   │ │ - Ticket      │ │               │
│   questions   │ │   statistics  │ │ - Future      │
│ - Definitions │ │ - Historical  │ │   predictions │
│ - Explanations│ │   data        │ │ - Trend       │
│               │ │ - Counts      │ │   analysis    │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Project Structure

```
Multi-Agent-Analyst/
├── agents/
│   ├── __init__.py         # Package exports
│   ├── rag.py              # RAG agent for knowledge queries
│   ├── db.py               # DB agent for database queries
│   ├── forecasting.py      # Forecasting agent for predictions
│   └── orchestrator.py     # Orchestrator node and agent call nodes
├── config.py               # LLM and timezone configuration
├── graph.py                # LangGraph workflow definition
├── main.py                 # Entry point and CLI
├── schemas.py              # Pydantic models for structured outputs
├── state.py                # Graph state definition
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
└── README.md               # This file
```

## Agents

### Orchestrator
- Plans step-by-step to answer user queries
- Routes to appropriate agents based on the task type
- Maintains state between agent calls
- Limits to 5 steps to prevent infinite loops

### RAG Agent (`agents/rag.py`)
- Handles knowledge questions and explanations
- Use cases: "What does NET_500 mean?", "How do I troubleshoot X?"
- In production: connects to vector database for retrieval-augmented generation

### DB Agent (`agents/db.py`)
- Handles database queries for ticket statistics
- Use cases: "Get ticket count for last 2 days", "How many tickets last week?"
- Parses natural language into structured date/days parameters
- In production: executes SQL queries against the database

### Forecasting Agent (`agents/forecasting.py`)
- Generates future predictions
- Use cases: "Forecast tickets for the next 3 days"
- In production: connects to ML forecasting models

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   # Copy the example file
   copy .env.example .env  # Windows
   cp .env.example .env    # macOS/Linux
   
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Running the System

```bash
python main.py
```

### Programmatic Usage

```python
from main import run_query

# Ask a knowledge question (routes to RAG Agent)
result = run_query("what does NET_500 mean?")

# Request ticket statistics (routes to DB Agent)
result = run_query("Get ticket count from today till 2 days")

# Request forecasts (routes to Forecasting Agent)
result = run_query("Forecast ticket counts for the next 3 days")
```

### Visualizing the Graph

```python
from main import visualize_graph

visualize_graph("my_workflow.png")
```

## Extending the System

### Adding a New Agent

1. Create a new file in `agents/` (e.g., `agents/my_agent.py`)
2. Implement the agent function
3. Add any required schemas to `schemas.py`
4. Export from `agents/__init__.py`
5. Create a call node in `agents/orchestrator.py`
6. Update `OrchestratorDecision` schema with new action type
7. Add the new node and edges in `graph.py`

### Connecting Real Data Sources

Replace the hardcoded implementations in:
- `agents/rag.py` - Connect to your vector database (Pinecone, Weaviate, etc.)
- `agents/db.py` - Connect to your SQL/NoSQL database
- `agents/forecasting.py` - Connect to your ML forecasting service

## Development

### Running Tests

```bash
pip install pytest
pytest tests/
```

### Debugging

The system prints debug information during execution. Set `verbose=False` in `run_query()` to suppress output.
