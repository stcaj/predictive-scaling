# Autoscaling Datasets

This repository contains datasets collected during experiments evaluating different autoscaling strategies for microservices deployed in Kubernetes. The experiments include:

- static replica configuration
- Horizontal Pod Autoscaler (HPA)
- ML-based autoscaling (MLP and Hybrid LSTM-CNN-GRU models)

---

## Data Collection

During experiments, workload was generated using Locust to simulate user traffic under controlled conditions.

While the application was running, runtime metrics were collected using Prometheus and a custom metrics service.

Collected metrics include:
- timestamps
- CPU usage (absolute and percentage)
- memory usage
- request load and system-level indicators

---

## Scripts

### metrics_reader.py
Reads and extracts metrics from raw datasets.

```bash
python metrics_reader.py
```

### preprocess_dataset.py
Preprocesses raw metrics into structured datasets used for training and evaluation.

```bash
python preprocess_dataset.py
```

### synthesize_dataset.py
Generates synthetic datasets based on real workload traces.

```bash
python synthesize_dataset.py
```

### utils.py
Utility functions used across preprocessing and dataset handling.

## Datasets

### HPA
Metrics collected under Kubernetes Horizontal Pod Autoscaler.

### Hybrid
Data collected when using the Hybrid ML model (LSTM + CNN + GRU).

### MLP
Data collected when using the MLP-based autoscaling approach.

### static_week
Baseline dataset collected under static replica configuration (2 replicas per microservice).

### synthetic_year
Synthetic dataset generated from one week of real workload data (static_week).

### fastStorage
Bitbrains fastStorage workload dataset used as an external benchmark:
https://www.kaggle.com/datasets/gauravdhamane/gwa-bitbrains

### test_app_in_thesis
External microservices application used for experiments (see project documentation).