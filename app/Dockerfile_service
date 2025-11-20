# -------------------------------------
# Ticket Fraud Detection App Dockerfile
# -------------------------------------
 
# Use a maintained base image (buster is deprecated)
ARG BASE_IMAGE=python:3.10-slim-bullseye
FROM $BASE_IMAGE
 
# Set working directory
WORKDIR /app
 
# Copy app code
COPY . .
 
# Install system dependencies (for scipy, zlib, nltk, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
&& rm -rf /var/lib/apt/lists/*
 
# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m nltk.downloader punkt averaged_perceptron_tagger stopwords
 
# Expose Flask port
ENV SERVER_PORT=8000
EXPOSE 8000
 
# Set environment variables for Snowflake
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
 
# Run the batch service
CMD ["python3", "main_service.py"]