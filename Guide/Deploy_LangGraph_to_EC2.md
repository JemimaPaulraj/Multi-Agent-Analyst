# Deploy our Multi-Agent LangGraph APP on ECR

#-------------------------***STEPS TO LAUNCH AN EC2 INSTANCE***-----------------------------------------

#-------------***STEP 1. Create an IAM Role for EC2 to Access ECR and Cloud Watch***--------------------

1. Open **IAM** in the AWS Console
2. Left menu → **Roles** → **Create role**.
3. **Trusted entity type:** AWS service.
4. **Use case:** EC2 → **Next**.
5. Search for **CloudWatchFullAccess, AmazonBedrockReadOnly, AmazonS3FullAccess, ** → **Next**.
6. Create Role name **Allow-EC2-access-ECR-and-cloudWatch**  → **Create role**
7. Open the same role again to give permission to use ECR
8. **Add permissions:** Click **Create policy** (opens new tab).
   - **Policy editor** → JSON, paste:
   - **Next** → Policy name: `ECR-Pull-MultiAgentAnalyst-Policy` → **Create policy**.

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": "ecr:GetAuthorizationToken",
         "Resource": "*"
       },
       {
         "Effect": "Allow",
         "Action": [
           "ecr:BatchGetImage",
           "ecr:GetDownloadUrlForLayer"
         ],
         "Resource": "arn:aws:ecr:us-east-1:251323758798:repository/multi-agent-analyst-image"
       }
     ]
   }
   ```
# GetAuthorizationToken: The EC2 service is allowed to ask ECR: "Please give me a temporary login token so I can access your images."
# Permission to Download the actual Docker image layers.
9. **Add permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "arn:aws:sagemaker:us-east-1:251323758798:endpoint/prophet-fastapi-endpoint"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:DescribeEndpoint"
      ],
      "Resource": "*"
    }
  ]
}
```

#----------------***STEP 2. Create Security Group for EC2***--------------------------------------------

* A security group is the firewall for your EC2 instance. It controls which traffic is allowed in and out.

* By default, no inbound traffic is allowed. So:
- Without rules allowing port 22 → you can’t SSH into the instance.
- Without rules allowing port 8000 → nothing from the internet can reach your Multi-Agent Analyst API.

1. Open **EC2** in the AWS Console: https://console.aws.amazon.com/ec2/
2. Left menu → **Security groups** (under Network & Security) → **Create security group**.
3. **Name:** `multi-agent-analyst-security-group`.
4. **Description:** Allow SSH and API port 8000.
5. **VPC:** Default VPC (or the VPC where you will launch the instance).
6. **Inbound rules** – Add:

   | Type        | Protocol | Port range | Source        | Description   |
   |------------|----------|------------|---------------|---------------|
   | SSH        | TCP      | 22         | My IP         | SSH access    |
   | Custom TCP | TCP      | 8000       | 0.0.0.0/0     | API (or My IP)|

   - **My IP** = your current IP (Console can fill it).
   - Use **0.0.0.0/0** for 8000 only if you want the API reachable from the internet; otherwise use **My IP** or a VPN/CIDR.

7. **Outbound rules:** Leave default (all traffic to 0.0.0.0/0).
8. **Create security group**.

#-------------------***STEP 3. Create and Launch an EC2 Instance***------------------------------------

1. In **EC2** → **Instances** → **Launch instance**.
2. **Name:** `multi-agent-analyst` (or any name).
3. **AMI:** **Amazon Linux 2023** (or **Amazon Linux 2**). Both work; commands below assume Amazon Linux.
4. **Instance type:** e.g. **t3.small** or **t3.medium** (depending on load and cost).
5. **Key pair:**
   - Create new or use existing.
   - Download the `.pem` and keep it safe (needed for SSH).
6. **Network settings:**
   - **VPC:** Same as the security group (e.g. default).
   - **Subnet:** Any public subnet (e.g. default) if you want a public IP.
   - **Auto-assign public IP:** Enable.
   - **Firewall (security groups):** Select existing → choose **multi-agent-analyst-security-group**.
7. **Storage:** Default 8 GB gp3 is fine; increase if you need more.
8. **Advanced details** (expand):
   - **IAM instance profile:** Select **Allow-EC2-access-ECR-and-cloudWatch ** (the role you created in Part 1).
9. **Launch instance**.

Wait until **Instance state** is **Running**. Note the **Public IPv4 address** (e.g. `54.209.185.9`).

#-------------------***STEP 4. Connect and install Docker***------------------------------------------

1. **Connect to EC2 to use SSH**
   - In EC2 → Instances → select your instance → **Connect**.
   - Use **EC2 Instance Connect** (browser) or **SSH client** with your `.pem`:
     ```bash
     ssh -i "your-key.pem" ec2-user@<PUBLIC-IP>
     ```
2. **Install Docker** (Amazon Linux 2023):
   ```bash                              
   sudo dnf update -y                   # refresh package list and upgrade installed packages
   sudo dnf install -y docker           # Install docker
   sudo systemctl start docker          # Start the docker
   sudo systemctl enable docker         # enable automatic docker restart when you restart the instance
   sudo usermod -aG docker ec2-user     # Adds ec2-user to docker group, then you need not use sudo
   ```
3. **Log out and SSH back in** so the `docker` group applies:
   ```bash
   exit
   # Then SSH in again
   ```

-------------------***STEP 5. Deploy the App- Pull the Image from ECR***-----------------------------

1. **Login to ECR** (uses the IAM role; no keys needed)

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 251323758798.dkr.ecr.us-east-1.amazonaws.com
```

2. **Pull the Image** from the ECR

```bash
docker pull 251323758798.dkr.ecr.us-east-1.amazonaws.com/multi-agent-analyst-image:latest
```

3. **Create a new container** from the image and start it.
```bash
docker run -d \
  --name multi-agent-analyst \
  --restart unless-stopped \
  -p 8000:8000 \
  -e MULTI_AGENT_SECRET_ARN=multi-agent-analyst-secrets \
  -e AWS_REGION=us-east-1 \
  251323758798.dkr.ecr.us-east-1.amazonaws.com/multi-agent-analyst-image:latest
```
   (With `--network host`, the app listens on the host’s 8000; no need for `-p 8000:8000`.)

-------------------***STEP 5a. Allow EC2 to call SageMaker (for Forecasting agent)***-----------------------------

1. **IAM** → **Roles** → open the role attached to your EC2 instance (e.g. `Allow-EC2-access-ECR-and-cloudWatch`).
2. **Add permissions** → **Create inline policy** → **JSON**:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "arn:aws:sagemaker:us-east-1:251323758798:endpoint/prophet-fastapi-endpoint"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:DescribeEndpoint"
      ],
      "Resource": "*"
    }
  ]
}
```

3. **Next** → Policy name: e.g. `SageMaker-Invoke-MultiAgent` → **Create policy**.
4. Attach the policy to the EC2 role if you created it as a standalone policy; if you used inline, it’s already attached.

This lets the LangGraph app (running in the container on EC2) call your SageMaker endpoint for forecasts. No VPC peering or security group change is needed; invocation uses the AWS API.

--------------------***STEP 6. Keep the access keys in AWS Secret Manager***------------------------

1. **Create a secret** in Secrets Manager (e.g. name `multi-agent-analyst-secrets`):
   - Secret type: **Other type of secret**.
   - Key/value: add `OPENAI_API_KEY` = your OpenAI API key (and any other keys).
   - Or store as plaintext JSON: `{"OPENAI_API_KEY":"sk-...","AWS_REGION":"us-east-1"}`.

2. **Allow EC2 to access Secret Manager**
    - Go to IAM → Roles → click the role attached to your EC2 (e.g. Allow-EC2-access-ECR-and-cloudWatch).
    - Add permissions → Create inline policy → JSON.
```bash
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "secretsmanager:GetSecretValue",
    "Resource": "arn:aws:secretsmanager:us-east-1:251323758798:secret:multi-agent-analyst-secret-AmFwqO"
  }]
}
```
    - Save the policy (e.g. name it SecretsManager-Read-MultiAgent).

---------------------***STEP 7. Test***-------------------------------------------------------------
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What is NET_500?"}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What is forecast for next 2 days?"}'
```