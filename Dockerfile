# # Stage 1: Build the React frontend
# FROM node:20-slim AS frontend-builder

# # Set the working directory
# WORKDIR /app/frontend

# # Copy package.json and package-lock.json
# COPY frontend/package.json frontend/package-lock.json ./

# # Install dependencies
# RUN npm install

# # Copy the rest of the frontend application code
# COPY frontend/ ./

# # Build the frontend application
# RUN npm run build

# Stage 2: Build the Python backend
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Verify installation
RUN node -v && npm -v

# Create app directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies
# RUN npm install

# Copy the rest of the application code
COPY . .

# Set permissions and logs
RUN mkdir -p /app/logs && \
    chmod -R a+r /app && \
    chmod +x /app/start-local.sh && \
    find /app -type d -exec chmod a+x {} \;

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Entrypoint script
CMD ["/bin/bash", "./start-local.sh", "--backend"]
