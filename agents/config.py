"""
LLM Configuration with AWS Bedrock Guardrails and Token Tracking.

Production Logging Strategy:
- CloudWatch LOGS: Text data (request_id, session_id, user_query, llm_response, errors, stack traces)
- CloudWatch METRICS: Numerical data (latency, tokens, cost, steps, request_count, error_count)
- No duplication: Numbers go to metrics only, text goes to logs only
"""

import os
import logging
import uuid
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()

# ---------------------------
# Logging Setup (Console + CloudWatch)
# ---------------------------
import json
import watchtower
import boto3
import traceback

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
METRICS_NAMESPACE = "MultiAgentAnalyst"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="watchtower")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Only setup logging once
if not getattr(logging.root, '_cw_configured', False):
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    # Suppress noisy library logs (only show our app logs)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    logging.getLogger("langchain_aws").setLevel(logging.WARNING)
    logging.getLogger("langchain_aws.chat_models.bedrock_converse").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # CloudWatch handler only when AWS credentials available (skip in CI / local test)
    try:
        logs_client = boto3.client("logs", region_name=AWS_REGION)
        cloudwatch_handler = watchtower.CloudWatchLogHandler(
            log_group_name="multi-agent-analyst",
            log_stream_name="app-{strftime:%Y-%m-%d}",
            boto3_client=logs_client,
            create_log_group=True,
            use_queues=False
        )
        logging.root.addHandler(cloudwatch_handler)
    except Exception:
        pass  # No credentials (e.g. CI) or AWS unavailable: use console logging only
    logging.root._cw_configured = True

def get_logger(name: str):
    """Get a logger - inherits CloudWatch handler from root."""
    return logging.getLogger(name)


cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return f"req_{uuid.uuid4().hex[:12]}"


# ---------------------------
# Request Context (Single JSON blob per request)
# ---------------------------
_request_context = {}

def start_request(request_id: str, session_id: str, user_query: str):
    """Start collecting events for a request."""
    _request_context[request_id] = {
        "request_id": request_id,
        "session_id": session_id,
        "user_query": user_query,
        "agent_flow": [],
    }

def add_event(request_id: str, **kwargs):
    """Add event data to the request context."""
    if request_id in _request_context:
        ctx = _request_context[request_id]
        
        # Track agent flow and per-agent metrics
        if "agent" in kwargs:
            agent_name = kwargs.pop("agent")
            ctx["agent_flow"].append(agent_name)
            
            # If agent metrics provided, store/accumulate them per-agent
            if "agent_metrics" in kwargs:
                metrics = kwargs.pop("agent_metrics")
                if "agents" not in ctx:
                    ctx["agents"] = {}
                
                if agent_name in ctx["agents"]:
                    # Accumulate metrics for agents called multiple times (e.g., orchestrator)
                    existing = ctx["agents"][agent_name]
                    existing["latency_sec"] = round(existing.get("latency_sec", 0) + metrics.get("latency_sec", 0), 3)
                    existing["tokens_in"] += metrics.get("tokens_in", 0)
                    existing["tokens_out"] += metrics.get("tokens_out", 0)
                    existing["cost_usd"] += metrics.get("cost_usd", 0)
                    existing["calls"] = existing.get("calls", 1) + 1
                else:
                    metrics["calls"] = 1
                    ctx["agents"][agent_name] = metrics
        
        # Only keep first values for these fields (don't overwrite)
        for key in ["orchestrator_decision", "rag_query", "db_query", "forecasting_payload"]:
            if key in kwargs:
                if key not in ctx:
                    ctx[key] = kwargs.pop(key)
                else:
                    kwargs.pop(key)
        
        # Update remaining fields
        ctx.update(kwargs)

def end_request(request_id: str, llm_response: str, status: str = "success"):
    """Log the complete request as one JSON blob and cleanup."""
    if request_id not in _request_context:
        return
    
    ctx = _request_context.pop(request_id)
    ctx["llm_response"] = llm_response
    ctx["status"] = status
    
    logger = get_logger("request")
    logger.info(json.dumps(ctx))


def log_metrics(latency_sec: float, tokens_in: int, tokens_out: int, cost_usd: float, steps: int, agent_name: str = None):
    """
    Send numerical metrics to CloudWatch Metrics.
    
    These are for dashboards and alarms - NOT for debugging text.
    """
    metric_data = [
        {"MetricName": "Latency", "Value": latency_sec, "Unit": "Seconds"},
        {"MetricName": "TokensIn", "Value": tokens_in, "Unit": "Count"},
        {"MetricName": "TokensOut", "Value": tokens_out, "Unit": "Count"},
        {"MetricName": "Cost", "Value": cost_usd * 1000000, "Unit": "Count"},
        {"MetricName": "Steps", "Value": steps, "Unit": "Count"},
        {"MetricName": "RequestCount", "Value": 1, "Unit": "Count"},
    ]
    
    # Add agent dimension if provided (for per-agent dashboards)
    if agent_name:
        for metric in metric_data:
            metric["Dimensions"] = [{"Name": "Agent", "Value": agent_name}]
    
    cloudwatch.put_metric_data(Namespace=METRICS_NAMESPACE, MetricData=metric_data)


def log_error_metric(agent_name: str = None):
    """Log error count to CloudWatch Metrics for alarms."""
    metric_data = [{"MetricName": "ErrorCount", "Value": 1, "Unit": "Count"}]
    
    if agent_name:
        metric_data[0]["Dimensions"] = [{"Name": "Agent", "Value": agent_name}]
    
    cloudwatch.put_metric_data(Namespace=METRICS_NAMESPACE, MetricData=metric_data)


def log_agent_metrics(agent_name: str, latency_sec: float, tokens_in: int = 0, tokens_out: int = 0):
    """
    Log per-agent metrics to CloudWatch with Agent dimension.
    
    Use this in each agent node to track individual agent performance.
    """
    cost_usd = estimate_cost(tokens_in, tokens_out)
    
    cloudwatch.put_metric_data(
        Namespace=METRICS_NAMESPACE,
        MetricData=[
            {"MetricName": "Latency", "Value": latency_sec, "Unit": "Seconds", 
             "Dimensions": [{"Name": "Agent", "Value": agent_name}]},
            {"MetricName": "TokensIn", "Value": tokens_in, "Unit": "Count",
             "Dimensions": [{"Name": "Agent", "Value": agent_name}]},
            {"MetricName": "TokensOut", "Value": tokens_out, "Unit": "Count",
             "Dimensions": [{"Name": "Agent", "Value": agent_name}]},
            {"MetricName": "Cost", "Value": cost_usd * 1000000, "Unit": "Count",
             "Dimensions": [{"Name": "Agent", "Value": agent_name}]},
            {"MetricName": "InvocationCount", "Value": 1, "Unit": "Count",
             "Dimensions": [{"Name": "Agent", "Value": agent_name}]},
        ]
    )


def log_request(logger, request_id: str, session_id: str, user_query: str, agent_name: str = None):
    """
    Log request details to CloudWatch Logs.
    
    Full user query logged (suitable for low-traffic systems).
    For high traffic, consider sampling.
    """
    log_data = {
        "event": "request_received",
        "request_id": request_id,
        "session_id": session_id,
        "user_query": user_query,
    }
    if agent_name:
        log_data["agent"] = agent_name
    
    logger.info(json.dumps(log_data))


def log_response(logger, request_id: str, session_id: str, llm_response: str, status: str = "success", agent_name: str = None):
    """
    Log response details to CloudWatch Logs.
    
    Full LLM response logged (suitable for low-traffic systems).
    For high traffic, consider sampling or truncating.
    """
    log_data = {
        "event": "response_sent",
        "request_id": request_id,
        "session_id": session_id,
        "llm_response": llm_response,
        "status": status,
    }
    if agent_name:
        log_data["agent"] = agent_name
    
    logger.info(json.dumps(log_data))


def log_error(logger, request_id: str, session_id: str, error_message: str, agent_name: str = None, include_trace: bool = True):
    """
    Log error details to CloudWatch Logs with optional stack trace.
    
    Also sends error count to CloudWatch Metrics for alarms.
    """
    log_data = {
        "event": "error",
        "request_id": request_id,
        "session_id": session_id,
        "error_message": error_message,
    }
    if agent_name:
        log_data["agent"] = agent_name
    if include_trace:
        log_data["stack_trace"] = traceback.format_exc()
    
    logger.error(json.dumps(log_data))
    
    # Also log to metrics for alerting
    log_error_metric(agent_name)


def log_rag_retrieval(logger, query: str, docs_with_scores: list, answer: str = None):
    """
    Save RAG retrieval data to S3 for offline RAGAS evaluation.
    
    Tier 3 Strategy:
    - Save full retrieval data to S3 (cheap storage)
    - Run weekly RAGAS evaluation on collected data
    - No flooding of CloudWatch Logs
    
    S3 Path: s3://{bucket}/rag_evaluation/{date}/{timestamp}.json
    """
    from pathlib import Path
    from datetime import datetime
    
    # Build full retrieval record for RAGAS
    retrieved_docs = []
    contexts = []
    for doc, score in docs_with_scores:
        content = doc.page_content
        contexts.append(content)
        retrieved_docs.append({
            "content": content,
            "score": round(float(score), 4),
            "source": Path(doc.metadata.get("source", "unknown")).name,
            "page": doc.metadata.get("page", "")
        })
    
    # RAGAS-compatible format
    ragas_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": query,
        "contexts": contexts,
        "answer": answer,
        "ground_truth": None,  # Fill manually for evaluation
        "metadata": {
            "num_docs_retrieved": len(docs_with_scores),
            "top_score": round(float(docs_with_scores[0][1]), 4) if docs_with_scores else None,
            "sources": [d["source"] for d in retrieved_docs],
            "retrieved_documents": retrieved_docs
        }
    }
    
    # Save to S3 (silent operation - no CloudWatch logging)
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        bucket = os.getenv("S3_BUCKET", "ticket-forecasting-lake")
        date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
        timestamp = datetime.utcnow().strftime("%H%M%S%f")
        s3_key = f"rag_evaluation/{date_prefix}/{timestamp}.json"
        
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(ragas_record),
            ContentType="application/json"
        )
        
    except Exception as e:
        logger.error(f"Failed to save RAG data to S3: {e}")

logger = get_logger("config")

# ---------------------------
# AWS Bedrock Configuration
# ---------------------------
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

# Guardrails Configuration
GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"

# Nova Pro pricing (per 1K tokens) - update if prices change
NOVA_PRO_INPUT_COST = 0.0008   # $0.0008 per 1K input tokens
NOVA_PRO_OUTPUT_COST = 0.0032  # $0.0032 per 1K output tokens


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost based on Nova Pro pricing."""
    input_cost = (input_tokens / 1000) * NOVA_PRO_INPUT_COST
    output_cost = (output_tokens / 1000) * NOVA_PRO_OUTPUT_COST
    return round(input_cost + output_cost, 6)


# Create LLM with or without guardrails (only once)
llm = None
if not hasattr(logging.root, '_llm_instance'):
    if GUARDRAIL_ID and ENABLE_GUARDRAILS:
        llm = ChatBedrock(
            model_id=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
            model_kwargs={"temperature": 0},
            guardrails={
                "guardrailIdentifier": GUARDRAIL_ID,
                "guardrailVersion": GUARDRAIL_VERSION,
                "trace": "enabled"
            }
        )
    else:
        llm = ChatBedrock(
            model_id=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
            model_kwargs={"temperature": 0}
        )
    logging.root._llm_instance = llm
else:
    llm = getattr(logging.root, '_llm_instance', None)
