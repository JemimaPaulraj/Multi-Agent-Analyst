"""
Agents package for the Multi-Agent Analyst system.
"""

import sys
from pathlib import Path

# Add paths for imports
_parent = str(Path(__file__).parent.parent)
_current = str(Path(__file__).parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _current not in sys.path:
    sys.path.insert(0, _current)

from config import llm
from forecasting import forecasting_agent
from rag import rag_agent
from db import db_agent
from orchestrator import (
    orchestrator_node,
    call_forecasting_node,
    call_rag_node,
    call_db_node,
)

__all__ = [
    "llm",
    "forecasting_agent",
    "rag_agent",
    "db_agent",
    "orchestrator_node",
    "call_forecasting_node",
    "call_rag_node",
    "call_db_node",
]
