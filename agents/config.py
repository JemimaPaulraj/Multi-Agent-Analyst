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

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logging.root.addHandler(watchtower.CloudWatchLogHandler(log_group="multi-agent-analyst"))

def get_logger(name: str):
    return logging.getLogger(name)


cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return f"req_{uuid.uuid4().hex[:12]}"


def log_metrics(latency_ms: int, tokens_in: int, tokens_out: int, cost_usd: float, steps: int, agent_name: str = None):
    """
    Send numerical metrics to CloudWatch Metrics.
    
    These are for dashboards and alarms - NOT for debugging text.
    """
    metric_data = [
        {"MetricName": "Latency", "Value": latency_ms, "Unit": "Milliseconds"},
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


def log_agent_metrics(agent_name: str, latency_ms: int, tokens_in: int = 0, tokens_out: int = 0):
    """
    Log per-agent metrics to CloudWatch with Agent dimension.
    
    Use this in each agent node to track individual agent performance.
    """
    cost_usd = estimate_cost(tokens_in, tokens_out)
    
    cloudwatch.put_metric_data(
        Namespace=METRICS_NAMESPACE,
        MetricData=[
            {"MetricName": "Latency", "Value": latency_ms, "Unit": "Milliseconds", 
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
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
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


# Create LLM with or without guardrails
if GUARDRAIL_ID and ENABLE_GUARDRAILS:
    logger.info(f"Guardrails ENABLED: {GUARDRAIL_ID} (v{GUARDRAIL_VERSION})")
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
    logger.info("Guardrails DISABLED")
    llm = ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"temperature": 0}
    )

logger.info(f"LLM initialized: {BEDROCK_MODEL_ID}")
