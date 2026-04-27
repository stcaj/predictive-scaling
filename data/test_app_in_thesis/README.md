## Web Application

This thesis uses an external open-source microservices application as a test workload for experimental evaluation in Kubernetes.

The application is not included in this repository and must be deployed separately.

Source:
https://github.com/GoogleCloudPlatform/microservices-demo

### app_modifications

Directory contains modifications used for microservice 'loadgenerator'

## Setup

1. Clone the repository:
```bash
git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
```

2. Deploy the application following the official instructions in the repository.

3. Verify that all services are running in the Kubernetes cluster before starting the autoscaler.