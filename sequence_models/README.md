# Prediction of CPU usage using RNN

## Acknowledgment
This part of the thisis was adapted from code presented in:
- Vidiečan, M. (2024). *Prediktívne škálovanie virtuálnych softvérových platforiem na báze umelej inteligencie*. Slovenská technická univ. v Bratislave FIIT UISI (FIIT)

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