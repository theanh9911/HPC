#!/bin/bash
# Docker Swarm Deployment Script
# cluster/deploy.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Print commands before executing them
set -x

# Variables
IMAGE_NAME="pneumonia-detection-api"
IMAGE_TAG="latest"
STACK_NAME="pneumonia-detection"
COMPOSE_FILE="../docker/docker-compose.yml"

# Check if Docker Swarm is initialized
if ! docker info | grep -q "Swarm: active"; then
    echo "Initializing Docker Swarm..."
    docker swarm init
    echo "Docker Swarm initialized."
else
    echo "Docker Swarm is already active."
fi

# Build the Docker image
echo "Building Docker image..."
cd .. && docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f docker/Dockerfile .

# Deploy the stack to Docker Swarm
echo "Deploying stack to Docker Swarm..."
docker stack deploy -c ${COMPOSE_FILE} ${STACK_NAME}

# Check deployment status
echo "Checking deployment status..."
docker stack services ${STACK_NAME}

echo "Deployment completed successfully!"
echo "Access the API at http://localhost:8000"
echo "Use the following command to check service logs:"
echo "  docker service logs ${STACK_NAME}_pneumonia-api"
echo "Use the following command to scale the service:"
echo "  docker service scale ${STACK_NAME}_pneumonia-api=5"
echo "Use the following command to remove the stack:"
echo "  docker stack rm ${STACK_NAME}"

# Wait for services to start
echo "Waiting for services to start..."
sleep 15

# Check service health
echo "Service health check:"
docker service ls --filter name=${STACK_NAME}