#!/bin/bash
set -e

# MOUNT_DIR="/media/stefansko/C90D-6492" # - Example for specific USB drive
MOUNT_DIR="../logs"
LOG_LOCATION="/mnt/data/logs"

# Ensure USB logs mount is active (runs in background if not already running)
if ! pgrep -f "minikube mount $MOUNT_DIR:$LOG_LOCATION" >/dev/null; then
    echo "Starting Minikube mount..."
    nohup minikube mount $MOUNT_DIR:$LOG_LOCATION >/dev/null 2>&1 &
fi


# Wait until USB mount is accessible inside Minikube
echo "Waiting for USB mount to be accessible inside Minikube..."
for i in {1..30}; do
    if minikube ssh "ls $LOG_LOCATION" >/dev/null 2>&1; then
        echo "USB mount is ready."
        break
    fi
    echo "Waiting... ($i)"
    sleep 2
done