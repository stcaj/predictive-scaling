#!/bin/bash
set -e

# metadata
IMAGE_NAME="mlp-autoscaler:latest"
APP_GROUP="mlp"
YAML_DIR="."

# Ensure Minikube is running
if ! minikube status &>/dev/null; then
    echo "Minikube is not running. Please start it first."
    exit 1
fi

# Switch to Minikube's Docker
echo "Switching Docker CLI to Minikube’s Docker..."
eval $(minikube docker-env)

# Build image
echo "Building image inside Minikube: $IMAGE_NAME"
docker build --pull -t $IMAGE_NAME .

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -k $YAML_DIR

# Restart and wait for all deployments in app-group
echo "Restarting all $APP_GROUP deployments..."
deployments=$(kubectl get deployments -l app-group=$APP_GROUP -o name || true)
if [ -n "$deployments" ]; then
    for d in $deployments; do
        kubectl rollout restart $d
        kubectl rollout status $d --timeout=180s
    done
else
    echo "No $APP_GROUP deployments found."
fi

# Show info at the end
echo "Setup complete!"
