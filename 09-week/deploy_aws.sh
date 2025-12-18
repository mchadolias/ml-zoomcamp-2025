#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# CONFIG — EDIT THESE
# -----------------------------------------------------------------------------

AWS_REGION="eu-north-1"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

ECR_REPO_NAME="hair-lambda"
IMAGE_TAG="latest"

# -----------------------------------------------------------------------------
# DERIVED VALUES (DO NOT EDIT)
# -----------------------------------------------------------------------------

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

# -----------------------------------------------------------------------------
# CHECKS
# -----------------------------------------------------------------------------

command -v aws >/dev/null || { echo "❌ aws CLI not found"; exit 1; }
command -v docker >/dev/null || { echo "❌ docker not found"; exit 1; }

echo "✅ AWS Account: ${ACCOUNT_ID}"
echo "✅ Region: ${AWS_REGION}"
echo "✅ Image URI: ${IMAGE_URI}"

# -----------------------------------------------------------------------------
# CREATE ECR REPOSITORY (if missing)
# -----------------------------------------------------------------------------

if ! aws ecr describe-repositories \
  --repository-names "${ECR_REPO_NAME}" \
  --region "${AWS_REGION}" >/dev/null 2>&1; then

  echo "📦 Creating ECR repository: ${ECR_REPO_NAME}"
  aws ecr create-repository \
    --repository-name "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" >/dev/null
else
  echo "📦 ECR repository exists"
fi

# -----------------------------------------------------------------------------
# AUTHENTICATE DOCKER TO ECR
# -----------------------------------------------------------------------------

echo "🔐 Logging in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URI}"

# -----------------------------------------------------------------------------
# BUILD IMAGE
# -----------------------------------------------------------------------------

echo "🐳 Building Docker image"
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .

# -----------------------------------------------------------------------------
# TAG & PUSH IMAGE
# -----------------------------------------------------------------------------

echo "🏷️  Tagging image"
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"

echo "🚀 Pushing image to ECR"
docker push "${IMAGE_URI}"


echo "✅ Deployment complete!"
