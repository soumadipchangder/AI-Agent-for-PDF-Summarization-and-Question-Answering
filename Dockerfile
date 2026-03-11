# ================================================================
# Hugging Face Spaces compatible Dockerfile
# HF Spaces only exposes a single port: 7860
# Architecture:
#   - Streamlit runs on port 7860 (publicly accessible via HF proxy)
#   - Flask runs on port 5001 (internal only, Streamlit talks to it via localhost)
# ================================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Create the uploads directory
RUN mkdir -p uploads

# HuggingFace Spaces requires the app to listen on port 7860
EXPOSE 7860

# Start Flask (internal on 5001) AND Streamlit (public on 7860)
# Flask runs in the background; Streamlit is the public-facing app
CMD ["sh", "-c", "PYTHONPATH=. python backend/server.py & streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true"]
