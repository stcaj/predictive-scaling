# Predictive Autoscaling of Microservices Using Machine Learning Approaches with Incremental Learning

## Source Code

This repository contains the exact version of the implementation used for experimental evaluation in the diploma thesis.

## Description

This work investigates machine learning approaches for predictive autoscaling of microservices in Kubernetes. Two approaches are implemented:

- **Approach 1 – Multilayer Perceptron (MLP):** classification-based prediction of future CPU usage
- **Approach 2 – Sequence-based models:**
  - Long Short-Term Memory (LSTM)
  - Hybrid architecture combining LSTM, CNN, and GRU in parallel

## Setup

Each approach has its own independent setup instructions:

- **Approach 1 (MLP):** see `mlp/README.md`
- **Approach 2 (LSTM & Hybrid):** see `sequence_models/README.md`

Follow the instructions in the respective directories to install dependencies and run experiments.