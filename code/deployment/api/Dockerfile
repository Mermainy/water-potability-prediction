FROM python:3.11-slim
WORKDIR /app
COPY code/deployment/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY code/schema.py code/schema.py
COPY code/features.py code/features.py
COPY code/deployment/api/main.py code/deployment/api/main.py
COPY models/model.joblib models/metrics.json models/
ENV PYTHONPATH=/app/code
EXPOSE 8000
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
