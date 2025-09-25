# Base image
ARG BASE_IMAGE=python:3.10-slim-buster
FROM $BASE_IMAGE
 
# Set working directory
WORKDIR /app
 
# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
 
# Copy the rest of the application
COPY . .
 
# Default command
CMD ["python", "main.py"]