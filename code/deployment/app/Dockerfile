FROM python:3.11-slim
WORKDIR /app
COPY code/deployment/app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY code/schema.py /app/schema.py
COPY code/deployment/app/main.py /app/main.py
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
