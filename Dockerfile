FROM python:3.11-slim

# Set environment variables to optimize Python performance and configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

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

# Copy the rest of the application code
COPY . /app/

# Pre-collect static files for Whitenoise to serve them efficiently in production
RUN python manage.py collectstatic --noinput

# Expose port 8000 (Railway will dynamically bind to this or its own PORT environment variable)
EXPOSE 8000

# Start Django with Gunicorn, automatically running migrations before booting the web server
CMD ["sh", "-c", "python manage.py migrate && gunicorn recommendation_v1.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload"]
