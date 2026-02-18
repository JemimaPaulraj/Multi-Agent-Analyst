"""
LLM Configuration with AWS Bedrock Guardrails and Token Tracking.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()

# ---------------------------
# Logging Setup (Console + CloudWatch)
# ---------------------------
import json
import watchtower

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logging.root.addHandler(watchtower.CloudWatchLogHandler(log_group="multi-agent-analyst"))

def get_logger(name: str):
    return logging.getLogger(name)

import boto3
cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)

def log_metrics(session, latency_ms, tokens_in, tokens_out, cost_usd, steps):
    """Send metrics to CloudWatch for dashboard."""
    cloudwatch.put_metric_data(
        Namespace="MultiAgentAnalyst",
        MetricData=[
            {"MetricName": "Latency", "Value": latency_ms, "Unit": "Milliseconds"},
            {"MetricName": "TokensIn", "Value": tokens_in, "Unit": "Count"},
            {"MetricName": "TokensOut", "Value": tokens_out, "Unit": "Count"},
            {"MetricName": "Cost", "Value": cost_usd * 1000000, "Unit": "Count"},  # Store as micro-dollars
            {"MetricName": "Steps", "Value": steps, "Unit": "Count"},
        ]
    )

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
