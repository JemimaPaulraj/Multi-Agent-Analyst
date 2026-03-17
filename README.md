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

### **1. Orchestrator Agent**
- Plans step-by-step to answer user queries
- Routes to appropriate agents based on task type
- Maintains conversation state via LangGraph checkpointer
- Limits to 5 steps to prevent infinite loops
- Tracks token usage and costs per request

### **2. RAG Agent** (`agents/rag.py`)
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


### **3. DB Agent** (`agents/db.py`)
- Natural language to SQL via LangChain SQL Agent
- Queries MySQL database on AWS RDS
- **Use cases:**
  - "Get ticket count for last 3 days"
  - "How many tickets were created last week?"
  - "Show ticket statistics from January 2026"
  - "What was the average resolution time yesterday?"
 
<img width="3012" height="1524" alt="image" src="https://github.com/user-attachments/assets/562b6595-15b9-49e1-9c70-3b5d7c823846" />


### **4. Forecasting Agent** (`agents/forecasting.py`)
- Calls Prophet model deployed on SageMaker endpoint
- Supports custom start dates and horizons
- **Use cases:**
  - "Forecast tickets for the next 5 days"
  - "Predict ticket volume for next week"
  - "What will be the ticket count from March 1st for 7 days?"
  - "Estimate workload for the coming weekend"

<img width="3054" height="1526" alt="image" src="https://github.com/user-attachments/assets/732b0117-8070-48bd-9d88-9a6b178e3268" />

<img width="2566" height="274" alt="image" src="https://github.com/user-attachments/assets/e87782ed-8333-46d4-aedd-5c9580b9a387" />


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

<img width="2438" height="876" alt="image" src="https://github.com/user-attachments/assets/8871fd00-f768-45e1-bfa3-16ba9754465e" />

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

<img width="3028" height="1414" alt="image" src="https://github.com/user-attachments/assets/dac6badd-8013-4b29-a71f-26d6ffce5601" />


### CloudWatch Metrics
- `Latency`, `TokensIn`, `TokensOut`, `Cost`, `Steps`
- `ErrorCount` with Agent dimension
- Custom namespace: `MultiAgentAnalyst`

### LangSmith
- Full LLM tracing via `@traceable` decorators

<img width="3072" height="1568" alt="image" src="https://github.com/user-attachments/assets/4f4118f2-fe2d-4ac2-847e-827da7551015" />


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

