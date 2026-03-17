# Deploy our Multi-Agent LangGraph APP on ECR

#-------------------------***STEPS TO Deploy AN SageMaker Endpoint***-----------------------------------------

#---------------------------------***STEP 1. Create ECR Repository***----------------------------------------

1. Go to **AWS Console → ECR (Elastic Container Registry)**
2. Click **Create repository**
3. Configure:
   - **Repository name**: `prophet-fastapi-inference`
   - **Image tag mutability**: Mutable
   - **Scan on push**: Enabled (optional)
4. Click **Create repository**
5. Note the **Repository URI**: `251323758798.dkr.ecr.us-east-1.amazonaws.com/prophet-fastapi-inference`

#-------------------------***STEP 2. Build and Push the Docker Image Inside ECR***----------------------------

<#
  * On Windows or Mac, a plain docker build can produce an image for the wrong OS/arch (e.g. Windows or arm64) OCI format.
  * EC2 needs linux/amd64. So, build the image using --platform linux/amd64 and output the image in Docker-format tar.
  * Take the image from the tar file and load it into Docker.
  * Tag the image with the repository name and the latest tag.
  * Push the image to ECR.
#>

*** Step 1 : AWS gives a temporary password to login into ECR**
```bash
aws ecr get-login-password --region us-east-1 |
docker login --username AWS --password-stdin 251323758798.dkr.ecr.us-east-1.amazonaws.com
```

*** Step 2 : build as a Docker-format tar (NOT OCI)**
```bash
docker buildx create --use
docker buildx build --platform linux/amd64 -t prophet-fastapi-inference:latest --output type=docker,dest=prophet.tar .
```

*** Step 3 : load into docker**
```bash
docker load -i .\prophet.tar
```

*** Step 4 : Tag + Push to ECR**
```bash
docker tag prophet-fastapi-inference:latest 251323758798.dkr.ecr.us-east-1.amazonaws.com/prophet-fastapi-inference:latest
docker push 251323758798.dkr.ecr.us-east-1.amazonaws.com/prophet-fastapi-inference:latest
```

#-------------------***STEP 3. Create a Permission for SageMaker container to access S3***--------------------

IAM → AmazonSageMakerServiceCatalogProductsUseRole → Permissions → Add inline policy → JSON:

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::ticket-forecasting-lake"
    },
    {
      "Sid": "GetModelTar",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::ticket-forecasting-lake/Model/model.tar.gz"
    }
  ]
}


#---------------------***STEP 4: Create model in SageMaker***-------------------------------------------------

Create the deployable model that references your ECR image and (if needed) S3 artifacts.

1. Go to **AWS Console → SageMaker → Inference → Deployable models**.
2. Click **Create model**.
3. Fill in:
   - **Model name** : `prophet-byoc-model`
   - **IAM Role** : Select `AmazonSageMakerServiceCatalogProductsUseRole`
   - **ECR image URI** : `251323758798.dkr.ecr.us-east-1.amazonaws.com/prophet-fastapi-inference:latest`
   - **S3 model data / artifact path** :`s3://ticket-forecasting-lake/Model/model.tar.gz`
4. Click **Create model**.

#---------------------***STEP 5: Create endpoint configuration***--------------------------------------------

Define how the endpoint will run (instance type, model, variant).

1. Go to **SageMaker → Inference → Endpoint configurations**.
2. Click **Create endpoint configuration**.
3. Set:
   - **Name:** `prophet-fastapi-config`
   - **Type of endpoint:** **Provisioned**
4. Under **Production variants**:
   - Click **Create production variant**.
   - **Select the model** you created in Step 4.
   - Use **Edit** (on the right) and set:
     - **Instance type:** `ml.t2.medium`
     - **Variant name:** `AllTraffic`
   - Click **Create endpoint configuration**.

#---------------------***STEP 6: Create the endpoint***-----------------------------------------------------

Create the live endpoint using the configuration from Step 5.

1. Go to **SageMaker → Inference → Endpoints**.
2. Click **Create endpoint**.
3. Set:
   - **Endpoint name:** `prophet-fastapi-endpoint`
   - **Use an existing endpoint configuration** → select **prophet-fastapi-config** (from Step 5).
4. Click **Create endpoint**.
5. Wait until **Status** is **InService** before calling the endpoint.

---

## CD: Deploy via GitHub Actions

The workflow **`.github/workflows/cd-sagemaker.yml`** automates:

1. **Build** the Prophet inference image from `model_folder/`
2. **Push** the image to ECR (`prophet-fastapi-inference` repo)
3. **Clean up** untagged images in that ECR repo
4. **Create** a new SageMaker model (pointing to the new image + S3 model data)
5. **Create** a new endpoint configuration (with that model, `ml.t2.medium`)
6. **Create or update** the endpoint `prophet-fastapi-endpoint` to use the new config

**Triggers:** After CI succeeds on `main`/`master`, or on release, or manually via **Run workflow**.

### Required GitHub secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key (used by existing CD) |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REGISTRY` | e.g. `<>.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY_SAGEMAKER` | ECR repo for Prophet image: `prophet-fastapi-inference` |
| `SAGEMAKER_EXECUTION_ROLE_ARN` | ARN of the role SageMaker uses to run the container (e.g. `AmazonSageMakerServiceCatalogProductsUseRole` or your custom role) |
| `SAGEMAKER_MODEL_DATA_URL` | S3 path to model artifact, e.g. `s3://ticket-forecasting-lake/Model/model.tar.gz` |

### Optional GitHub variable

| Variable | Default | Description |
|----------|---------|-------------|
| `SAGEMAKER_ENDPOINT_NAME` | `prophet-fastapi-endpoint` | Endpoint name to create/update |

### IAM permissions for the GitHub Actions user

The IAM user (or OIDC role) whose keys are in `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` needs:

- **ECR:** `ecr:GetAuthorizationToken`, plus for repo `prophet-fastapi-inference`: `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, `ecr:PutImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:ListImages`, `ecr:DescribeImages`, `ecr:BatchDeleteImage`
- **SageMaker:** `sagemaker:CreateModel`, `sagemaker:DescribeModel`, `sagemaker:CreateEndpointConfig`, `sagemaker:DescribeEndpointConfig`, `sagemaker:CreateEndpoint`, `sagemaker:DescribeEndpoint`, `sagemaker:UpdateEndpoint`

After adding the secrets (and optionally the variable), push to `main` or run **Actions → CD SageMaker → Run workflow**.
