## Setup Environment

This section describes the environment setup required to run the experiments, including Kubernetes cluster initialization and storage configuration.

---

### 1. Install required tools and start cluster

Run the following scripts:

```bash
./tools_installer.sh
./minikube_setup.sh
```

### 2. Create persistent storage directory

Create a local directory that will be used for:

- model storage
- experiment outputs

Example:
```
mkdir -p /path/to/storage
```

### 3. Configure mount directory
Set variable:

MOUNT_DIR=/path/to/storage

### 4. Mount storage directory

Run:
```bash
./mount-dir.sh
```