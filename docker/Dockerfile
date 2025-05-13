# Dockerfile for Pneumonia Detection API
# docker/Dockerfile

# Use TensorFlow GPU image as base
FROM tensorflow/tensorflow:2.12.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ ./api/

# Copy model files
COPY model/model.h5 ./model/model.h5
COPY model/model_config.json ./model/model_config.json

# Environment variables
ENV MODEL_PATH=/app/model/model.h5
ENV CONFIG_PATH=/app/model/model_config.json
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]