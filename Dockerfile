FROM python:3.12.0

WORKDIR /app

COPY api/main.py .
COPY saved_models ./saved_models
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
