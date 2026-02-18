"""
Forecasting Agent module.
Calls AWS SageMaker endpoint for predictions.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from langsmith import traceable
import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import ForecastPayload
from config import get_logger

logger = get_logger("forecasting")

# Timezone Configuration
NY_TZ = ZoneInfo("America/New_York")

# SageMaker Configuration
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "prophet-fastapi-endpoint-latest")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Session state
_state = {"runtime_client": None}


def get_sagemaker_runtime():
    """Get or create SageMaker runtime client."""
    if _state["runtime_client"] is None:
        _state["runtime_client"] = boto3.client(
            "sagemaker-runtime",
            region_name=AWS_REGION
        )
    return _state["runtime_client"]


def check_forecast_service():
    """Check if the SageMaker endpoint is available."""
    try:
        sagemaker = boto3.client("sagemaker", region_name=AWS_REGION)
        response = sagemaker.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT)
        status = response.get("EndpointStatus", "Unknown")
        
        if status == "InService":
            logger.info(f"SageMaker endpoint '{SAGEMAKER_ENDPOINT}' is InService")
            return True
        else:
            logger.warning(f"Endpoint status: {status}")
            return False
    except ClientError as e:
        logger.error(f"Cannot check endpoint: {e}")
        return False


@traceable(name="Forecasting Agent")
def forecasting_agent(payload: ForecastPayload) -> dict:
    """
    Forecasting Agent that calls SageMaker endpoint for predictions.
    
    Args:
        payload: ForecastPayload containing horizon_days and optional start_date
        
    Returns:
        Dictionary containing the forecast results
    """
    horizon_days = int(payload.horizon_days)
    
    if payload.start_date:
        start_date = payload.start_date
    else:
        start_date = datetime.now(NY_TZ).date().isoformat()
    
    request_payload = {
        "horizon_days": horizon_days,
        "start_date": start_date
    }
    
    logger.info(f"Calling endpoint={SAGEMAKER_ENDPOINT} | start_date={start_date} | horizon={horizon_days}")
    
    try:
        runtime = get_sagemaker_runtime()
        
        response = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(request_payload)
        )
        
        result = json.loads(response["Body"].read().decode("utf-8"))
        
        logger.info(f"Forecast returned {len(result.get('forecast', []))} predictions")
        
        return {
            "agent": "forecasting_agent",
            "payload_received": payload.model_dump(),
            "start_date": start_date,
            "forecast": result.get("forecast", []),
        }
    
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        
        logger.error(f"SageMaker error ({error_code}): {error_msg}")
        
        return {
            "agent": "forecasting_agent",
            "payload_received": payload.model_dump(),
            "forecast": [],
            "error": f"SageMaker error: {error_msg}"
        }
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "agent": "forecasting_agent",
            "payload_received": payload.model_dump(),
            "forecast": [],
            "error": f"Unexpected error: {str(e)}"
        }
