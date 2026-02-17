"""
LLM Configuration with AWS Bedrock Guardrails.
"""

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()

# AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

# Guardrails Configuration
GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"

# Create LLM with or without guardrails
if GUARDRAIL_ID and ENABLE_GUARDRAILS:
    # With guardrails
    print(f"[CONFIG] Guardrails ENABLED: {GUARDRAIL_ID} (v{GUARDRAIL_VERSION})")
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
    # Without guardrails - don't pass guardrails parameter at all
    print("[CONFIG] Guardrails DISABLED")
    llm = ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"temperature": 0}
    )

print(f"[CONFIG] LLM initialized: {BEDROCK_MODEL_ID}")
