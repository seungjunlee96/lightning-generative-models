#!/bin/bash

# Define a variable for the Docker file name, passed as an argument
DOCKER_FILE_NAME=${1:-"Dockerfile"}
CONTAINER_NAME="lightning-generative-models"
IMAGE_NAME="lightning-generative-models"
IMAGE_TAG="latest"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Remove the existing Docker container
echo "Removing existing Docker container '$CONTAINER_NAME'..."
if docker rm $CONTAINER_NAME -f; then
    echo "Container '$CONTAINER_NAME' removed successfully."
else
    echo "Error removing container '$CONTAINER_NAME'."
    exit 1
fi

# Build the Docker image
echo "Building the Docker image from $DOCKER_FILE_NAME..."
if docker build -f $DOCKER_FILE_NAME -t $IMAGE_NAME:$IMAGE_TAG .; then
    echo "Docker image built successfully."
else
    echo "Error building Docker image."
    exit 1
fi

# Run the Docker container
echo "Running the Docker container..."
if docker run \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    -dit \
    --name $CONTAINER_NAME \
    -v /mnt:/mnt \
    -p 8889:8889 \
    $IMAGE_NAME:$IMAGE_TAG; then
    echo "Docker container is running."
else
    echo "Error running Docker container."
    exit 1
fi

# List running containers
echo "Listing running containers..."
docker ps

# Attach to the running container's shell
echo "Attaching to the running container's shell..."
docker exec -it $CONTAINER_NAME bash
