FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY artifacts artifacts
RUN useradd --create-home apiuser
USER apiuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
