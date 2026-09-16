FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py web_app.py .

RUN pip install --no-cache-dir openai-whisper flask "torch==2.4.1+cu121" --extra-index-url https://download.pytorch.org/whl/cu121

EXPOSE 5000

CMD ["python", "web_app.py"]
