#!/bin/bash
set -e

# Prevent running as root
if [ "$EUID" -eq 0 ]; then
  echo "Do not run this script with sudo or as root"
  echo "Run it as your normal user instead"
  exit 1
fi

# minikube size parameters - adjust as needed
CPUS=6
MEMORY=12288
DISK_SIZE="100g"

# Start Minikube with Docker driver
echo "=== Starting Minikube with ${CPUS} CPUs, ${MEMORY}MB RAM, ${DISK_SIZE} disk ==="
minikube start --driver=docker --cpus=$CPUS --memory=$MEMORY --disk-size=$DISK_SIZE


# Enable useful addons
echo "Enabling addons..."
minikube addons enable metrics-server
minikube addons enable storage-provisioner
minikube addons enable default-storageclass

echo "Minikube setup complete!"
echo "To use kubectl with Minikube, run: kubectl get nodes"


echo "=== Installing Prometheus ==="
# Add repo if not already added
if ! helm repo list | grep -q "prometheus-community"; then
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
fi
helm repo update

# Create monitoring namespace if missing
kubectl get ns monitoring >/dev/null 2>&1 || kubectl create namespace monitoring

# Install or upgrade Prometheus stack
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring

echo "Prometheus (kube-prometheus-stack) installed in namespace 'monitoring'"
echo "To check: helm list -n monitoring"