FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /models

VOLUME /models

COPY download_trained_models.py .

# Download and convert models, save into /models
CMD ["python3", "download_trained_models.py"]
