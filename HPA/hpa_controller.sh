#!/usr/bin/env bash
set -euo pipefail

# --------------------------------
# HPA Toggle Script
# Usage:
#   ./hpa-toggle.sh enable
#   ./hpa-toggle.sh disable
# --------------------------------

ACTION="${1:-}"
NAMESPACE="default"
HPA_FILE="hpa.yaml"

# Validate action
if [[ "${ACTION}" != "enable" && "${ACTION}" != "disable" ]]; then
  echo "Usage: $0 enable|disable"
  exit 1
fi

# Check for kubectl
command -v kubectl >/dev/null 2>&1 || { echo "[ERROR] kubectl not found in PATH"; exit 1; }

# Check if HPA file exists
if [[ ! -f "${HPA_FILE}" ]]; then
  echo "[ERROR] File not found: ${HPA_FILE}"
  exit 1
fi

# Execute action
echo "[INFO] Namespace: ${NAMESPACE}"
echo "[INFO] Action: ${ACTION}"
echo "[INFO] HPA file: ${HPA_FILE}"

# Enable or disable HPAs
if [[ "${ACTION}" == "enable" ]]; then
  kubectl apply -n "${NAMESPACE}" -f "${HPA_FILE}"
  echo "[DONE] All HPAs enabled."
else
  kubectl delete -n "${NAMESPACE}" -f "${HPA_FILE}" --ignore-not-found
  echo "[DONE] All HPAs disabled."
fi

# List current HPAs
echo
kubectl get hpa -n "${NAMESPACE}" || true
