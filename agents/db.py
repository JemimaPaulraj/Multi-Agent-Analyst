"""
DB Querying Agent module.
Uses SQL Agent to query MySQL database.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

from config import llm

# Database configuration from environment variables
DB_HOST = os.getenv("DB_HOST", "127.0.0.1:3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "company_db")

# Session state
_state = {"db": None, "agent": None}


def configure_db():
    """Configure and return SQLDatabase connection."""
    
    if _state["db"] is not None:
        print("[DB] Using existing database connection")
        return _state["db"]
    
    # Handle host:port format (e.g., "127.0.0.1:3306")
    host = DB_HOST
    port = "3306"
    if ":" in DB_HOST:
        host, port = DB_HOST.split(":")
    
    db_uri = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{host}:{port}/{DB_NAME}"
    print(f"[DB] Connecting to: mysql+pymysql://{DB_USER}:***@{host}:{port}/{DB_NAME}")
    
    try:
        _state["db"] = SQLDatabase.from_uri(db_uri)
        print(f"[DB] Connected to MySQL database: {DB_NAME}")
        return _state["db"]
    except Exception as e:
        print(f"[DB] Failed to connect: {e}")
        return None


def get_sql_agent():
    """Create and return SQL agent."""
    
    if _state["agent"] is not None:
        print("[DB] Using existing SQL agent")
        return _state["agent"]
    
    db = configure_db()
    if db is None:
        return None
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    _state["agent"] = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        agent_type="zero-shot-react-description",
        agent_kwargs={
            "prefix": (
                "You are a helpful data assistant with access to an SQL database.\n"
                "Answer all questions using the database.\n"
                "Return only the final answer clearly in plain English.\n"
                "If you do not know the answer, say 'I don't know'."
            )
        },
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    
    print("[DB] SQL agent created")
    return _state["agent"]


def db_agent(query: str) -> dict:
    """
    DB Querying Agent that uses SQL agent to query MySQL database.
    
    Args:
        query: The user's natural language query about ticket data
        
    Returns:
        Dictionary containing the agent's response
    """
    agent = get_sql_agent()
    
    if agent is None:
        return {
            "agent": "db_agent",
            "query": query,
            "answer": "Database connection failed. Check MySQL configuration.",
            "error": True
        }
    
    try:
        result = agent.invoke({"input": query})
        
        return {
            "agent": "db_agent",
            "query": query,
            "answer": result.get("output", "No response"),
            "error": False
        }
    except Exception as e:
        return {
            "agent": "db_agent",
            "query": query,
            "answer": f"Error executing query: {str(e)}",
            "error": True
        }
