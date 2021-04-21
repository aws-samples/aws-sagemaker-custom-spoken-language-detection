#!/usr/bin/env bash

# Specify an algorithm name
algorithm_name=spoken-language-detection

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-east-1 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}

fullname=${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest
# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly

$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# build docker image
DOCKER_BUILDKIT=1 docker build -t ${algorithm_name}:pytorch-cpu -f Dockerfile.cpu --no-cache .
docker tag ${algorithm_name}:pytorch-cpu ${fullname}
docker push ${fullname}