# Lambda Retraining Setup Guide

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  EventBridge    │────▶│  Lambda         │────▶│  SageMaker      │
│  (Weekly)       │     │  (Trigger)      │     │  Training Job   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                        │
                               ┌────────────────────────┼────────────────────────┐
                               │                        │                        │
                               ▼                        ▼                        ▼
                    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
                    │  CloudWatch     │      │  Model Registry │      │  Update/Keep    │
                    │  Alarm (MAPE)   │      │  (Version)      │      │  Endpoint       │
                    └────────┬────────┘      └─────────────────┘      └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  SNS Alert      │
                    │  (Drift Alert)  │
                    └─────────────────┘
```

## Model Drift Detection Flow

```
New Model Trained
       │
       ▼
  MAPE Calculated
       │
       ├─── MAPE < 20% ───▶ Deploy New Model ───▶ Update Endpoint
       │
       └─── MAPE >= 20% ──▶ REJECT New Model
                                   │
                                   ├──▶ Keep Old Model Serving
                                   ├──▶ Publish CloudWatch Metric
                                   ├──▶ Trigger CloudWatch Alarm
                                   └──▶ Send SNS Alert (with old model name)
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
| `SNS_TOPIC_ARN` | `arn:aws:sns:us-east-1:<account-id>:model-drift-alerts` |

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

## Step 6: Create SNS Topic for Model Drift Alerts

### Via AWS Console:

1. Go to **SNS > Topics > Create Topic**
2. Settings:
   - Type: **Standard**
   - Name: `model-drift-alerts`
3. Create the topic and copy the ARN
4. Go to **Subscriptions > Create Subscription**:
   - Protocol: **Email** (or Slack/Teams via Lambda)
   - Endpoint: Your email address
5. Confirm the subscription via email

### Via AWS CLI:

```bash
# Create SNS topic
aws sns create-topic --name model-drift-alerts

# Subscribe email
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:<account-id>:model-drift-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com

# Subscribe Slack (via AWS Chatbot or Lambda)
# See: https://docs.aws.amazon.com/chatbot/latest/adminguide/slack-setup.html
```

## Step 7: Verify CloudWatch Alarm

After the first Lambda run, verify the alarm was created:

1. Go to **CloudWatch > Alarms**
2. Look for: `prophet-fastapi-endpoint-latest-mape-drift-alarm`
3. Verify settings:
   - Namespace: `MLOps/Forecasting`
   - Metric: `ModelMAPE`
   - Threshold: `> 20`
   - Actions: SNS topic

## Monitoring

- **CloudWatch Logs**: `/aws/lambda/prophet-retrain-trigger`
- **CloudWatch Alarms**: `prophet-fastapi-endpoint-latest-mape-drift-alarm`
- **CloudWatch Metrics**: `MLOps/Forecasting` namespace
  - `ModelMAPE` - MAPE value per model
  - `ModelDriftDetected` - Count of drift events
  - `DataDrift` - Count of data drift events
- **SageMaker Console**: Training Jobs, Model Registry
- **S3**: `s3://ticket-forecasting-lake/Model/last_trained.json`

## SNS Alert Format

When model exceeds MAPE threshold (needs manual approval), you'll receive an email:

```
Subject: Model Needs Manual Approval
Body: New model exceeded MAPE threshold. MAPE: 25.3%, Threshold: 20.0%. Please review and approve manually.
```

## Cost Estimate

| Resource | Cost |
|----------|------|
| Lambda (weekly, ~10 min) | ~$0.01/month |
| SageMaker Training (ml.m5.large, ~5 min) | ~$0.50/run |
| CloudWatch Alarms (1 alarm) | ~$0.10/month |
| SNS (1000 notifications) | ~$0.00 (free tier) |
| **Total** | ~$2-3/month |
