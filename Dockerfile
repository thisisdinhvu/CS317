FROM python:3.12.0

# Set working dir
WORKDIR /app

# Copy all project files (or minimum needed)
COPY api/main.py .
COPY saved_models ./saved_models
COPY logging/ ./logging
COPY requirements.txt .
# COPY wait-for-fluentd.sh .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create logging dir if not exists (to avoid runtime error)
RUN mkdir -p logging


# Expose API port
EXPOSE 8000

# Run API via Uvicorn from api.main
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
