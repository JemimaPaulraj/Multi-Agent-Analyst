"""
API tests for Multi-Agent Analyst.
Add more test cases here before wiring up CI/CD.
"""
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    """Health check returns 200 and expected body."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "running" in data["message"].lower()
