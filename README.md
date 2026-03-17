# Multi-Agent Analyst

A production-ready multi-agent system built with **LangGraph** that orchestrates specialized agents for RAG (Retrieval-Augmented Generation), database querying, and time-series forecasting.

## Architecture

<img width="1288" height="1160" alt="image" src="https://github.com/user-attachments/assets/c68a3a32-c549-4057-bbe6-ac0995a6a998" />

## Key Features

- **Multi-Agent Orchestration**: LangGraph-based workflow with intelligent routing
- **RAG with Semantic Caching**: FAISS vectorstore + DynamoDB for fast repeated queries
- **Database Querying**: Natural language to SQL via LangChain SQL Agent
- **Time-Series Forecasting**: Prophet model deployed on SageMaker endpoint
- **Human-in-the-Loop**: Manual model approval via SageMaker Model Registry
- **Full Observability**: CloudWatch Logs/Metrics + LangSmith tracing
- **CI/CD Pipeline**: GitHub Actions → ECR → EC2 deployment
- **MLOps**: Automated weekly retraining with drift detection

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **LLM** | AWS Bedrock (Nova Pro) with Guardrails |
| **Framework** | LangGraph, LangChain, FastAPI |
| **Frontend** | Streamlit |
| **Vector Store** | FAISS (local) + S3 (PDFs) |
| **Cache** | DynamoDB (semantic cache) |
| **Database** | MySQL (RDS) |
| **ML/Forecasting** | Prophet, SageMaker Endpoints |
| **Observability** | CloudWatch, LangSmith |
| **Infrastructure** | EC2, ECR, Lambda, EventBridge, SNS |
| **CI/CD** | GitHub Actions |



## Agents

### **Orchestrator Agent**
- Plans step-by-step to answer user queries
- Routes to appropriate agents based on task type
- Maintains conversation state via LangGraph checkpointer
- Limits to 5 steps to prevent infinite loops
- Tracks token usage and costs per request

### **RAG Agent** (`agents/rag.py`)
- Retrieves knowledge from PDF documents stored in S3
- **Semantic Caching**: Similar queries return cached answers instantly
  - FAISS index stores query embeddings for similarity search
  - DynamoDB stores cached answers with TTL (24 hours)
  - Threshold: 85% similarity triggers cache hit
- Saves retrieval data for offline RAGAS evaluation
- **Use cases:**
  - "What does NET_500 mean?"
  - "Explain the troubleshooting steps for error code 404"
  - "What is the password reset policy?"
  - "How do I escalate a critical ticket?"
    
 <img width="3068" height="1574" alt="image" src="https://github.com/user-attachments/assets/c37c158f-18c5-4dbd-9553-6d59f475742a" />


### **DB Agent** (`agents/db.py`)
- Natural language to SQL via LangChain SQL Agent
- Queries MySQL database on AWS RDS
- **Use cases:**
  - "Get ticket count for last 3 days"
  - "How many tickets were created last week?"
  - "Show ticket statistics from January 2026"
  - "What was the average resolution time yesterday?"

### **Forecasting Agent** (`agents/forecasting.py`)
- Calls Prophet model deployed on SageMaker endpoint
- Supports custom start dates and horizons
- **Use cases:**
  - "Forecast tickets for the next 5 days"
  - "Predict ticket volume for next week"
  - "What will be the ticket count from March 1st for 7 days?"
  - "Estimate workload for the coming weekend"

## Project Structure

```
Multi-Agent-Analyst/
├── app/
│   ├── main.py                 # FastAPI backend
│   └── streamlit_app.py        # Streamlit UI
├── core/
│   ├── __init__.py             # Exports State, Schemas
│   ├── state.py                # LangGraph state definition
│   ├── graph.py                # LangGraph workflow
│   └── schemas.py              # Pydantic models for LLM outputs
├── agents/
│   ├── __init__.py             # Agent exports
│   ├── config.py               # LLM, logging, CloudWatch metrics
│   ├── orchestrator.py         # Orchestrator + agent call nodes
│   ├── rag.py                  # RAG agent with semantic cache
│   ├── db.py                   # SQL database agent
│   └── forecasting.py          # SageMaker forecasting agent
├── FAISS_Vectorstore/
│   ├── RAG_index/              # Document embeddings
│   └── Cache_index/            # Semantic cache embeddings
├── eval/
│   ├── ragas_eval.py           # RAGAS evaluation script
│   └── output.json             # Evaluation results
├── lambda/
│   ├── retrain_trigger.py      # Weekly retraining Lambda
│   ├── load_csv_to_rds.py      # Data loading script
│   └── SETUP.md                # Lambda setup guide
├── model_folder/
│   ├── app.py                  # Prophet inference server
│   ├── train                   # SageMaker training script
│   ├── serve                   # SageMaker serving script
│   ├── Dockerfile              # Model container
│   └── DEPLOYMENT_GUIDE.md     # SageMaker deployment guide
├── tests/
│   └── test_app.py             # API and orchestrator tests
├── .github/workflows/
│   ├── ci.yml                  # Test on push/PR
│   └── cd.yml                  # Build → ECR → EC2 deploy
├── Guide/
│   ├── Deploy_LangGraph_to_EC2.md
│   └── Deploy_Model_to_SageMaker.md
├── Data/                       # Sample data files
├── Dockerfile                  # FastAPI container
├── requirements.txt
└── README.md
```

## Setup & Installation

### Prerequisites

- Python 3.10+
- AWS Account with access to:
  - Bedrock (Nova Pro model)
  - S3, DynamoDB, RDS (MySQL)
  - SageMaker, ECR, EC2
  - Secrets Manager, CloudWatch
- OpenAI API key (for embeddings)

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/your-username/Multi-Agent-Analyst.git
cd Multi-Agent-Analyst

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy example file
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# LLM Configuration
BEDROCK_MODEL_ID=amazon.nova-pro-v1:0
OPENAI_API_KEY=sk-...                    # Required for embeddings

# Database Configuration
DB_HOST=your-rds-endpoint.amazonaws.com
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=ticket_database

# S3 Configuration (for RAG documents)
S3_BUCKET=your-bucket-name
S3_PREFIX=RAG_Data/

# SageMaker Configuration
SAGEMAKER_ENDPOINT=prophet-fastapi-endpoint

# Observability (Optional)
LANGSMITH_API_KEY=ls-...
LANGSMITH_PROJECT=multi-agent-analyst
```

### 4. Run the Application

**Option A: FastAPI Backend Only**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Option B: Streamlit Frontend**
```bash
streamlit run app/streamlit_app.py
```

**Option C: Both (separate terminals)**
```bash
# Terminal 1 - Backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
streamlit run app/streamlit_app.py
```

### 5. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Run tests
pytest tests/ -v
```

## API Usage

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Process query through multi-agent system |

### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Forecast tickets for next 3 days",
    "session_id": "user-123"
  }'
```

### Example Response

```json
{
  "query": "Forecast tickets for next 3 days",
  "answer": "Based on the forecast, expected ticket counts are: 2026-02-26: 45, 2026-02-27: 52, 2026-02-28: 48",
  "work": {
    "forecast_result": {...},
    "tokens_in": 1250,
    "tokens_out": 180
  },
  "steps": 2
}
```

## MLOps Pipeline

### Weekly Retraining Flow

```
EventBridge (Weekly) → Lambda → SageMaker Training Job
                                      ↓
                              Model Registry
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
              MAPE < 20%                          MAPE >= 20%
                    ↓                                   ↓
            Auto Deploy                     PendingManualApproval
                    ↓                                   ↓
           Update Endpoint                    SNS Alert → Human Review
                                                        ↓
                                              EventBridge → Lambda → Deploy
```

### Key Features

- **Data Drift Detection**: Alerts when input data distribution changes
- **Automatic Approval**: Models with MAPE < 20% auto-deploy
- **Human-in-the-Loop**: Manual approval triggers deployment via EventBridge
- **Rolling Updates**: Zero-downtime endpoint updates

## Observability

### CloudWatch Logs
- Request/response logging with `request_id` for tracing
- Per-agent metrics (latency, tokens, cost)
- Error logging with stack traces

### CloudWatch Metrics
- `Latency`, `TokensIn`, `TokensOut`, `Cost`, `Steps`
- `ErrorCount` with Agent dimension
- Custom namespace: `MultiAgentAnalyst`

### LangSmith
- Full LLM tracing via `@traceable` decorators
- Token usage tracking
- Prompt/completion inspection

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_app.py::test_health -v
```

## Deployment

### CI/CD Pipeline

1. **Push to main** → CI runs tests
2. **CI passes** → CD builds Docker image
3. **Push to ECR** → Deploy to EC2 via SSH

### Manual Deployment

```bash
# Build and push to ECR
docker build -t multi-agent-analyst .
docker tag multi-agent-analyst:latest <account>.dkr.ecr.us-east-1.amazonaws.com/multi-agent-analyst:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/multi-agent-analyst:latest

# Deploy on EC2
docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/multi-agent-analyst:latest
docker run -d -p 8000:8000 --name multi-agent-analyst \
  -e AWS_REGION=us-east-1 \
  -e MULTI_AGENT_SECRET_ARN=arn:aws:secretsmanager:... \
  multi-agent-analyst:latest
```

## Extending the System

### Adding a New Agent

1. Create `agents/new_agent.py` with your agent function
2. Add schemas to `core/schemas.py` if needed
3. Export from `agents/__init__.py`
4. Add call node in `agents/orchestrator.py`
5. Update `OrchestratorDecision` with new action
6. Add node and edges in `core/graph.py`

### Connecting New Data Sources

- **RAG**: Update `S3_BUCKET` and `S3_PREFIX` in `agents/rag.py`
- **Database**: Update `DB_*` environment variables
- **Forecasting**: Deploy new model to SageMaker, update `SAGEMAKER_ENDPOINT`

## Cost Estimation

| Resource | Estimated Cost |
|----------|---------------|
| EC2 (t3.medium) | ~$30/month |
| RDS (db.t3.micro) | ~$15/month |
| SageMaker Endpoint (ml.t2.medium) | ~$50/month |
| Lambda + EventBridge | ~$1/month |
| Bedrock (Nova Pro) | ~$0.001/request |
| **Total** | **~$100/month** |

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request
