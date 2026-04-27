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
# COPY main_lstm.py .
# COPY main_hybrid.py .
COPY main_prediction.py .
COPY prometheus_collector.py .

# for multi training
# COPY main_lstm_multi.py .
# COPY main_prediction_multi.py .


# Default command (overridden by Kubernetes)
CMD ["python"]
