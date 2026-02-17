# SageMaker BYOC Deployment Guide (AWS Console)

This guide walks you through deploying the Prophet forecasting model to AWS SageMaker with a public URL endpoint using the AWS Console UI.

## Architecture

```
Client (forecasting.py)
        │
        │ HTTPS Request
        ▼
┌─────────────────────────────┐
│  API Gateway (Public URL)   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   SageMaker Endpoint        │  ← Runs your BYOC container (app.py)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   S3 Bucket                 │  ← model.tar.gz
└─────────────────────────────┘
```

---

## Part 1: Prepare Model Artifact and Push Container to ECR

### Step 1.1: Create Model Archive

Package your trained Prophet model into a tar.gz file:

```bash
cd model_folder

# Create model.tar.gz containing model.joblib
tar -czvf model.tar.gz -C ../model model.joblib
```

### Step 1.2: Upload Model to S3

1. Go to **AWS Console → S3**
2. Navigate to your bucket: `ticket-forecasting-lake`
3. Create folder `Model/` if it doesn't exist
4. Click **Upload** → Select `model.tar.gz` → Click **Upload**
5. Note the S3 URI: `s3://ticket-forecasting-lake/Model/model.tar.gz`

### Step 1.3: Create ECR Repository

1. Go to **AWS Console → ECR (Elastic Container Registry)**
2. Click **Create repository**
3. Configure:
   - **Repository name**: `ticket-forecasting-prophet`
   - **Image tag mutability**: Mutable
   - **Scan on push**: Enabled (optional)
4. Click **Create repository**
5. Note the **Repository URI**: `<account-id>.dkr.ecr.<region>.amazonaws.com/ticket-forecasting-prophet`

### Step 1.4: Build and Push Docker Image

From your local machine with Docker installed:

```bash
cd model_folder

# Set variables (replace with your values)
AWS_REGION="us-east-1"
ACCOUNT_ID="123456789012"  # Your AWS account ID
ECR_REPO="ticket-forecasting-prophet"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build image
docker build -t $IMAGE_URI .

# Push to ECR
docker push $IMAGE_URI
```

---

## Part 2: Create SageMaker Execution Role

### Step 2.1: Create IAM Role

1. Go to **AWS Console → IAM → Roles**
2. Click **Create role**
3. Select **AWS service** → **SageMaker** → Click **Next**
4. Add permissions (search and select):
   - `AmazonSageMakerFullAccess`
   - `AmazonS3ReadOnlyAccess`
5. Click **Next**
6. **Role name**: `SageMakerExecutionRole`
7. Click **Create role**
8. Click on the role → Copy the **ARN**: `arn:aws:iam::<account-id>:role/SageMakerExecutionRole`

---

## Part 3: Create SageMaker Model

### Step 3.1: Create Model

1. Go to **AWS Console → SageMaker → Models** (under Inference)
2. Click **Create model**
3. Configure:

   **Model settings:**
   - **Model name**: `ticket-forecasting-prophet-model`
   - **IAM role**: Select `SageMakerExecutionRole` (created in Part 2)

   **Container definition:**
   - **Container input options**: Select **Provide model artifacts and inference image location**
   - **Provide model artifacts and inference image location**: Single model
   - **Location of inference code image**: `<account-id>.dkr.ecr.<region>.amazonaws.com/ticket-forecasting-prophet:latest`
   - **Location of model artifacts**: `s3://ticket-forecasting-lake/Model/model.tar.gz`

4. Click **Create model**

---

## Part 4: Create Endpoint Configuration

### Step 4.1: Create Endpoint Config

1. Go to **AWS Console → SageMaker → Endpoint configurations**
2. Click **Create endpoint configuration**
3. Configure:
   - **Endpoint configuration name**: `ticket-forecasting-prophet-config`
4. Under **Production variants**, click **Add model**:
   - **Model name**: Select `ticket-forecasting-prophet-model`
   - **Variant name**: `primary`
   - **Instance type**: `ml.t2.medium` (cost-effective for Prophet)
   - **Initial instance count**: `1`
5. Click **Create endpoint configuration**

---

## Part 5: Create SageMaker Endpoint

### Step 5.1: Deploy Endpoint

1. Go to **AWS Console → SageMaker → Endpoints**
2. Click **Create endpoint**
3. Configure:
   - **Endpoint name**: `ticket-forecasting-prophet-endpoint`
   - **Attach endpoint configuration**: Select **Use an existing endpoint configuration**
   - **Endpoint configuration**: Select `ticket-forecasting-prophet-config`
4. Click **Create endpoint**
5. **Wait 5-10 minutes** for status to change from "Creating" to **"InService"**

### Step 5.2: Test Endpoint (Optional - via AWS CLI)

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name ticket-forecasting-prophet-endpoint \
  --content-type application/json \
  --body '{"horizon_days": 7, "start_date": "2024-01-15"}' \
  --region us-east-1 \
  output.json

cat output.json
```

---

## Part 6: Create API Gateway for Public URL

SageMaker endpoints are private by default (require AWS credentials). To make it publicly accessible, create an API Gateway.

### Step 6.1: Create Lambda Function (Proxy)

1. Go to **AWS Console → Lambda**
2. Click **Create function**
3. Configure:
   - **Function name**: `ticket-forecasting-proxy`
   - **Runtime**: Python 3.11
   - **Architecture**: x86_64
4. Click **Create function**

5. **Replace the code** with:

```python
import json
import boto3
import os

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "ticket-forecasting-prophet-endpoint")
runtime = boto3.client("sagemaker-runtime")

def lambda_handler(event, context):
    print(f"Event: {json.dumps(event)}")
    
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    
    # Health check
    if path == "/ping" and http_method == "GET":
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"status": "ok"})
        }
    
    # Prediction
    if path == "/invocations" and http_method == "POST":
        try:
            body = event.get("body", "{}")
            if isinstance(body, str):
                body = json.loads(body) if body else {}
            
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(body)
            )
            
            result = json.loads(response["Body"].read().decode())
            
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps(result)
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": str(e)})
            }
    
    # CORS preflight
    if http_method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": ""
        }
    
    return {
        "statusCode": 404,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found"})
    }
```

6. Click **Deploy**

7. **Configuration → General configuration → Edit**:
   - **Timeout**: 1 minute
   - **Memory**: 256 MB
   - Click **Save**

8. **Configuration → Environment variables → Edit**:
   - Add: `SAGEMAKER_ENDPOINT` = `ticket-forecasting-prophet-endpoint`
   - Click **Save**

### Step 6.2: Add Lambda Permissions for SageMaker

1. Go to **Lambda → Configuration → Permissions**
2. Click on the **Role name** link (opens IAM)
3. Click **Add permissions → Attach policies**
4. Search and select: `AmazonSageMakerFullAccess`
5. Click **Attach policies**

### Step 6.3: Create API Gateway

1. Go to **AWS Console → API Gateway**
2. Click **Create API**
3. Choose **REST API** → Click **Build**
4. Configure:
   - **API name**: `ticket-forecasting-api`
   - **Endpoint Type**: Regional
5. Click **Create API**

### Step 6.4: Create /ping Resource and Method

1. Click **Create resource**
   - **Resource name**: `ping`
   - **Resource path**: `/ping`
   - Click **Create resource**

2. With `/ping` selected, click **Create method**
   - **Method type**: GET
   - **Integration type**: Lambda Function
   - **Lambda proxy integration**: ✓ Enabled
   - **Lambda function**: `ticket-forecasting-proxy`
   - Click **Create method**

### Step 6.5: Create /invocations Resource and Method

1. Click on `/` (root), then **Create resource**
   - **Resource name**: `invocations`
   - **Resource path**: `/invocations`
   - Click **Create resource**

2. With `/invocations` selected, click **Create method**
   - **Method type**: POST
   - **Integration type**: Lambda Function
   - **Lambda proxy integration**: ✓ Enabled
   - **Lambda function**: `ticket-forecasting-proxy`
   - Click **Create method**

3. (Optional) Add OPTIONS method for CORS - repeat step 2 with OPTIONS method

### Step 6.6: Deploy API

1. Click **Deploy API**
2. Configure:
   - **Stage**: Create new stage
   - **Stage name**: `prod`
3. Click **Deploy**
4. **Copy the Invoke URL**: `https://<api-id>.execute-api.<region>.amazonaws.com/prod`

---

## Part 7: Configure Your Application

### Step 7.1: Update Environment Variables

Update your `.env` file:

```bash
# SageMaker Public Endpoint
FORECAST_API_URL=https://<api-id>.execute-api.us-east-1.amazonaws.com/prod

# Increase timeout for cold starts
FORECAST_REQUEST_TIMEOUT=60
```

### Step 7.2: Test the Public Endpoint

```bash
# Health check
curl https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/ping

# Prediction
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/invocations \
  -H "Content-Type: application/json" \
  -d '{"horizon_days": 7, "start_date": "2024-01-15"}'
```

---

## Troubleshooting

### SageMaker Endpoint Stuck in "Creating"
- Check CloudWatch Logs: `/aws/sagemaker/Endpoints/ticket-forecasting-prophet-endpoint`
- Verify container health check (`/ping`) returns 200
- Verify model.tar.gz is accessible and contains model.joblib

### API Gateway Returns 500
- Check Lambda CloudWatch Logs
- Verify Lambda has SageMaker invoke permissions
- Verify endpoint name in Lambda environment variable

### Timeout Errors
- SageMaker cold starts can take 30-60 seconds
- Increase Lambda timeout to 1-2 minutes
- Increase `FORECAST_REQUEST_TIMEOUT` in your app

### Container Health Check Fails
- Verify `/ping` endpoint works locally:
  ```bash
  docker run -p 8080:8080 <image-uri>
  curl http://localhost:8080/ping
  ```
- Check model file exists in tar.gz

---

## Cost Estimation

| Resource | Approximate Cost |
|----------|-----------------|
| SageMaker ml.t2.medium | ~$0.05/hour (~$36/month) |
| API Gateway | First 1M requests free |
| Lambda | First 1M requests free |
| ECR Storage | ~$0.10/GB/month |

**Tip**: Delete the endpoint when not in use to save costs.

---

## Cleanup

To avoid ongoing charges:

1. **SageMaker → Endpoints** → Delete `ticket-forecasting-prophet-endpoint`
2. **SageMaker → Endpoint configurations** → Delete `ticket-forecasting-prophet-config`
3. **SageMaker → Models** → Delete `ticket-forecasting-prophet-model`
4. **API Gateway** → Delete `ticket-forecasting-api`
5. **Lambda** → Delete `ticket-forecasting-proxy`
6. **ECR** → Delete `ticket-forecasting-prophet` repository
