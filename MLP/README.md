# Prediction of CPU usage using MLP Classifier

## Acknowledgment
This part of the thesis is based on [Neural Network Prediction of Virtual Machine Workloads](https://github.com/author/original-repo) by Anthony Kwan and Fei Pan.

## How to run

### 1. Environment setup
Follow instructions in:
`environment_scripts/README.md`

This includes:
- dependencies installation
- Minikube setup prerequisites
- Kubernetes setup prerequisites

---

### 2. Test application (workload source)

The autoscaler is evaluated using a third-party open-source microservice application.

Setup instructions:
`data/test_app_in_thesis/README.md`

---

### 3. Start autoscaler

Deploy autoscaler in Kubernetes:

```bash
./setup.sh