"""
State definitions for the Multi-Agent Analyst graph.
"""

from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    The shared state for the multi-agent workflow.
    
    Attributes:
        messages: List of messages in the conversation (auto-accumulated via add_messages)
        work: Dictionary to store intermediate results from agents
        steps: Counter for the number of orchestrator steps (max 5 to prevent infinite loops)
    """
    messages: Annotated[list, add_messages]
    work: dict
    steps: int
