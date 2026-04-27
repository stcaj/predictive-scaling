#!/bin/bash
set -e

APP_GROUP="sequence"

echo "=== Shutting down deployments in app-group=$APP_GROUP ==="
echo "Scaling all replicas to 0..."
deployments=$(kubectl get deployments -l app-group=$APP_GROUP -o name || true)

if [ -n "$deployments" ]; then
    for d in $deployments; do
        kubectl scale $d --replicas=0 || true
    done
else
    echo "No deployments found with app-group=$APP_GROUP."
fi

echo "Shutdown complete!"
