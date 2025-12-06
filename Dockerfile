# Dockerfile
FROM python:3.9-slim

LABEL org.opencontainers.image.source="https://github.com/Abzwingten/lingui_analysis"


# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    libpq-dev \
    curl \
    ca-certificates \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install  --no-cache-dir -r requirements.txt





# Create working directory
WORKDIR /app

# Copy application code
COPY analyze_text.py .

# Create directory for output files
RUN mkdir -p /app/output

# Set entrypoint
ENTRYPOINT ["python", "./analyze_text.py"]
