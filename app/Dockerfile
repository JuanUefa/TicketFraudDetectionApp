# ---------------------------------------------
# Dockerfile for Ticket Fraud Detection (Job)
# ---------------------------------------------
 
# Use a slim Python base image with Debian Bullseye (actively maintained)
ARG BASE_IMAGE=python:3.10-slim-bullseye
FROM $BASE_IMAGE
 
# Set the working directory in the container
WORKDIR /app
 
# Copy all application files into the container
COPY . .
 
# Install system dependencies needed by some Python packages (e.g., scipy, nltk)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
&& rm -rf /var/lib/apt/lists/*
 
# Upgrade pip and install all required Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m nltk.downloader punkt averaged_perceptron_tagger stopwords
 
# Optional: Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
 
# Default command for the container
# This will be overridden by spcs.yaml, but helps for local testing
CMD ["python3", "main.py"]