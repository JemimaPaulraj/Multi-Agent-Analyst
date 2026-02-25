# Multi-Agent Analyst API — production image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (see .dockerignore for exclusions)
COPY main.py state.py graph.py schemas.py ./
COPY agents ./agents

EXPOSE 8000

# Run FastAPI with uvicorn (same as: uvicorn main:app --host 0.0.0.0 --port 8000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
