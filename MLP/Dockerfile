FROM python:3.11-slim

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_NO_CACHE_DIR=1

# Working directory
WORKDIR /app

# Install dependencies separately
COPY requirements.txt /tmp/requirements.txt

# Install dependencies in a single layer
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy apps with modules
COPY my_modules ./my_modules
COPY main_fortune_teller.py .
# COPY main_multiStep_MLP.py .

# Default command (overridden by Kubernetes)
CMD ["python"]
