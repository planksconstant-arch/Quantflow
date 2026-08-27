# Dockerfile for QuantFlow Institutional HFT & Swarm Platform
FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application repository
COPY . .

# Create directories for cached data and charts
RUN mkdir -p outputs/charts data/cache

# Expose Streamlit default port
EXPOSE 8501

# Default command launches the institutional trading terminal
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
