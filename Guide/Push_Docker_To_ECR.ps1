# ----------------------------------------------------------#
#   LangGraph App Push to ECR- Powershell Script            #
# ----------------------------------------------------------#

set -e
REGION="us-east-1"
REGISTRY="251323758798.dkr.ecr.us-east-1.amazonaws.com"
REPO="multi-agent-analyst-image"
IMAGE="${REGISTRY}/${REPO}:latest"
TAR="multi-agent-analyst.tar"

<#
  * On Windows or Mac, a plain docker build can produce an image for the wrong OS/arch (e.g. Windows or arm64) OCI format.
  * EC2 needs linux/amd64. So, build the image using --platform linux/amd64 and output the image in Docker-format tar.
  * Take the image from the tar file and load it into Docker.
  * Tag the image with the repository name and the latest tag.
  * Push the image to ECR.
#>

# Step 0: ECR login
echo "Step 0: Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

# Step 1: Build the image for linux/amd64

echo "Step 1: Building image (linux/amd64)..."
docker buildx create --use 2>/dev/null || true
docker buildx build --platform linux/amd64 -t "${REPO}:latest" --output "type=docker,dest=${TAR}" .

# Step 2: Load into Docker (v2 schema2)
echo "Step 2: Loading image into Docker..."
docker load -i "$TAR"

# Step 3: Tag and push to ECR
echo "Step 3: Tagging and pushing to ECR..."
docker tag "${REPO}:latest" "$IMAGE"
docker push "$IMAGE"

echo "Done. Image: $IMAGE"
