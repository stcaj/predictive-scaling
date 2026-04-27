#!/bin/bash
set -e

# Mapovanie: deployment -> minimal replicas
declare -A MIN_REPLICAS=(
  ["adservice"]=1
  ["cartservice"]=2
  ["checkoutservice"]=1
  ["currencyservice"]=2
  ["emailservice"]=1
  ["frontend"]=2
  ["paymentservice"]=2
  ["productcatalogservice"]=2
  ["recommendationservice"]=2
  ["shippingservice"]=1
)

echo "=== Scaling microservices to their MINIMUM replicas ==="

for DEPLOY in "${!MIN_REPLICAS[@]}"; do
    REPLICAS=${MIN_REPLICAS[$DEPLOY]}
    echo "Scaling $DEPLOY to $REPLICAS replicas"
    kubectl scale deployment/$DEPLOY --replicas=$REPLICAS || true
done

echo "Scaling complete!"