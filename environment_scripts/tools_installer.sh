#!/bin/bash
set -e

echo "=== Installing Tools ==="

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root: sudo $0"
  exit 1
fi

# Update system
echo "Updating system..."
apt-get update -y
apt-get upgrade -y

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get install -y docker.io
    systemctl enable docker
    systemctl start docker
    usermod -aG docker $USER
    echo "Docker installed. You may need to log out and log back in for group changes."
else
    echo "Docker is already installed."
fi

# Install kubectl
if ! command -v kubectl &> /dev/null; then
    echo "Installing kubectl..."
    RELEASE=$(curl -s https://dl.k8s.io/release/stable.txt) # Get latest stable version
    curl -LO "https://dl.k8s.io/release/v1.34.2/bin/linux/amd64/kubectl" # if different version needed, replace v1.34.2 with $RELEASE
    chmod +x kubectl
    mv kubectl /usr/local/bin/
else
    echo "kubectl is already installed."
fi

# Install minikube
if ! command -v minikube &> /dev/null; then
    echo "Installing Minikube..."
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    install minikube-linux-amd64 /usr/local/bin/minikube
    rm minikube-linux-amd64
else
    echo "Minikube is already installed."
fi

# Install Helm
if ! command -v helm &> /dev/null; then
    echo "Installing Helm..."
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
else
    echo "Helm is already installed."
fi

# Install Skaffold
if ! command -v skaffold &> /dev/null; then
    echo "Installing Skaffold..."
    latest=$(curl -s https://storage.googleapis.com/skaffold/releases/latest/stable.txt) # Get latest stable version
    curl -Lo skaffold "https://storage.googleapis.com/skaffold/releases/v2.17.0/skaffold-linux-amd64" # if different version needed, replace v2.17.0 with $latest
    install skaffold /usr/local/bin/
    rm skaffold
    echo "Skaffold installed."
else
    echo "Skaffold is already installed."
fi

echo "=== All tools installed successfully ==="
