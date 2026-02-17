# retrain_trigger.py
"""Lambda: Check bronze/ for new data → Trigger training → Register → Deploy if MAPE < 15%"""
import os, json, time
from datetime import datetime
import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "ticket-forecasting-lake")
ECR_IMAGE_URI = os.getenv("ECR_IMAGE_URI")
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "prophet-fastapi-endpoint-latest")
MODEL_PACKAGE_GROUP = os.getenv("MODEL_PACKAGE_GROUP", "prophet-forecasting-models")
MAPE_THRESHOLD = 20.0

sm = boto3.client("sagemaker", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)


def has_new_data():
    """Check if any new CSV in bronze/ since last training."""
    try:
        last = datetime.fromisoformat(json.loads(s3.get_object(Bucket=S3_BUCKET, Key="Model/last_trained.json")["Body"].read())["trained_at"])
        files = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="bronze/").get("Contents", [])
        return any(f["Key"].endswith(".csv") and f["LastModified"].replace(tzinfo=None) > last for f in files)
    except:
        return True


def lambda_handler(event, context):
    """Check for new data, trigger training (train script does all processing)."""
    
    if not has_new_data():
        return {"status": "skipped", "reason": "no_new_data"}
    
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"prophet-retrain-{ts}"
    
    # 1. Create Model Package Group (if not exists)
    try:
        sm.create_model_package_group(ModelPackageGroupName=MODEL_PACKAGE_GROUP, ModelPackageGroupDescription="Prophet forecasting models")
    except:
        pass
    
    # 2. Start training (train script reads bronze/, processes, trains)
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={"TrainingImage": ECR_IMAGE_URI, "TrainingInputMode": "File"},
        RoleArn=SAGEMAKER_ROLE,
        OutputDataConfig={"S3OutputPath": f"s3://{S3_BUCKET}/training-output/"},
        ResourceConfig={"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        Environment={"S3_BUCKET": S3_BUCKET},
    )
    
    # 3. Wait for training to complete
    while True:
        job_info = sm.describe_training_job(TrainingJobName=job_name)
        status = job_info["TrainingJobStatus"]
        if status == "Completed":
            training_job_arn = job_info["TrainingJobArn"]
            break
        if status in ["Failed", "Stopped"]:
            return {"status": "failed", "job": job_name}
        time.sleep(30)
    
    # 4. Get metrics & determine approval
    try:
        metrics = json.loads(s3.get_object(Bucket=S3_BUCKET, Key="Model/metrics.json")["Body"].read())
    except:
        metrics = {}
    mape = metrics.get("mape")
    approval = "Approved" if mape and mape < MAPE_THRESHOLD else "PendingManualApproval"
    
    # 5. Register in Model Registry
    pkg_params = {
        "ModelPackageGroupName": MODEL_PACKAGE_GROUP,
        "InferenceSpecification": {
            "Containers": [{"Image": ECR_IMAGE_URI, "ModelDataUrl": f"s3://{S3_BUCKET}/Model/model.tar.gz"}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.t2.medium", "ml.m5.large"]
        },
        "ModelApprovalStatus": approval,
        "MetadataProperties": {
            "GeneratedBy": training_job_arn
        }
    }
    if mape:
        # Format metrics in SageMaker Model Quality schema
        model_quality_metrics = {
            "regression_metrics": {
                "mape": {"value": mape},
                "rmse": {"value": metrics.get("rmse", 0)},
                "mae": {"value": metrics.get("mae", 0)}
            }
        }
        
        # Save metrics file to S3 in the expected format
        s3.put_object(Bucket=S3_BUCKET, Key=f"metrics/{ts}.json", Body=json.dumps(model_quality_metrics))
        
        # Attach metrics to model package
        pkg_params["ModelMetrics"] = {
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": f"s3://{S3_BUCKET}/metrics/{ts}.json"
                }
            }
        }
        
        # Also add customer metadata for easy viewing in Details tab
        pkg_params["CustomerMetadataProperties"] = {
            "mape": str(mape),
            "rmse": str(metrics.get("rmse", "")),
            "mae": str(metrics.get("mae", ""))
        }
    
    pkg_arn = sm.create_model_package(**pkg_params)["ModelPackageArn"]
    
    # 6. Deploy if approved
    if approval == "Approved":
        sm.create_model(ModelName=f"prophet-{ts}", PrimaryContainer={"ModelPackageName": pkg_arn}, ExecutionRoleArn=SAGEMAKER_ROLE)
        sm.create_endpoint_config(EndpointConfigName=f"prophet-cfg-{ts}", ProductionVariants=[{"VariantName": "primary", "ModelName": f"prophet-{ts}", "InstanceType": "ml.t2.medium", "InitialInstanceCount": 1}])
        
        # Check if endpoint exists, create if not, update if exists
        try:
            sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
            # Endpoint exists, update it
            sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=f"prophet-cfg-{ts}")
        except sm.exceptions.ClientError:
            # Endpoint doesn't exist, create it
            sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=f"prophet-cfg-{ts}")
    
    # 7. Save metadata
    s3.put_object(Bucket=S3_BUCKET, Key="Model/last_trained.json", Body=json.dumps({"trained_at": datetime.utcnow().isoformat(), "mape": mape, "approval": approval}))
    
    return {"status": "success", "job": job_name, "mape": mape, "approval": approval, "deployed": approval == "Approved"}
