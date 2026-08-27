# Minimal image for serving the churn prediction API + demo frontend.
# Build:  docker build -t churn-api .
# Run:    docker run -p 8000:8000 churn-api
FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ api/
COPY src/ src/
COPY frontend/ frontend/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
