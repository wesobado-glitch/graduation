FROM python:3.11-slim

# Set environment variables to optimize Python performance and memory usage
# MALLOC_ARENA_MAX=2: Reduces glibc memory fragmentation (critical for Python+PyTorch in containers)
# OMP/MKL_NUM_THREADS=1: Limit CPU threads to reduce memory overhead
# HF_HOME/TRANSFORMERS_CACHE: Cache HuggingFace models in a known location inside the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_CACHE=/app/.hf_cache

# Set the working directory
WORKDIR /app

# Install system dependencies (build-essential for compiling source packages, libgomp1 for CPU PyTorch & FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to take advantage of Docker layer caching
COPY requirements.txt /app/

# Install python dependencies without caching to keep the image slim
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace embedding model at BUILD time so it's cached in the image.
# This avoids downloading at runtime, which causes a memory spike (download buffer + model load).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy the rest of the application code
COPY . /app/

# Pre-collect static files for Whitenoise to serve them efficiently in production
RUN python manage.py collectstatic --noinput

# Expose port 8000 (Railway will dynamically bind to this or its own PORT environment variable)
EXPOSE 8000

# Start Django with Gunicorn, automatically running migrations before booting the web server
CMD ["sh", "-c", "python manage.py migrate && gunicorn recommendation_v1.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload"]
