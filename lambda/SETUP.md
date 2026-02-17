# Lambda Retraining Setup Guide

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  EventBridge    │────▶│  Lambda         │────▶│  SageMaker      │
│  (Weekly)       │     │  (Trigger)      │     │  Training Job   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  Update         │
                                                │  Endpoint       │
                                                └─────────────────┘
```

## Step 1: Create IAM Role for Lambda

1. Go to **IAM > Roles > Create Role**
2. Select **Lambda** as use case
3. Attach these policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3ReadOnlyAccess`
4. Name it: `lambda-sagemaker-retrain-role`
5. Copy the Role ARN

## Step 2: Create Lambda Function

1. Go to **Lambda > Create Function**
2. Settings:
   - Name: `prophet-retrain-trigger`
   - Runtime: `Python 3.11`
   - Role: Select the role from Step 1
3. Upload code:
   - Copy contents of `retrain_trigger.py`
4. Configuration:
   - Timeout: **15 minutes** (max for Lambda)
   - Memory: **256 MB**

## Step 3: Set Environment Variables

In Lambda > Configuration > Environment Variables:

| Key | Value |
|-----|-------|
| `AWS_REGION` | `us-east-1` |
| `S3_BUCKET` | `ticket-forecasting-lake` |
| `ECR_IMAGE_URI` | `<your-account-id>.dkr.ecr.us-east-1.amazonaws.com/prophet-forecast:latest` |
| `SAGEMAKER_ROLE` | `arn:aws:iam::<account-id>:role/SageMakerExecutionRole` |
| `ENDPOINT_NAME` | `prophet-fastapi-endpoint-latest` |

## Step 4: Create EventBridge Rule (Weekly Trigger)

1. Go to **EventBridge > Rules > Create Rule**
2. Settings:
   - Name: `weekly-prophet-retrain`
   - Schedule expression: `cron(0 9 ? * SUN *)` (Every Sunday 9 AM UTC)
3. Target:
   - Select **Lambda function**
   - Choose `prophet-retrain-trigger`
4. Create the rule

## Step 5: Test the Lambda

1. Go to Lambda > Test
2. Create test event with empty JSON: `{}`
3. Click **Test**
4. Check CloudWatch Logs for output

## Monitoring

- **CloudWatch Logs**: `/aws/lambda/prophet-retrain-trigger`
- **SageMaker Console**: Training Jobs
- **S3**: `s3://ticket-forecasting-lake/Model/last_trained.json`

## Cost Estimate

| Resource | Cost |
|----------|------|
| Lambda (weekly, ~10 min) | ~$0.01/month |
| SageMaker Training (ml.m5.large, ~5 min) | ~$0.50/run |
| **Total** | ~$2-3/month |
