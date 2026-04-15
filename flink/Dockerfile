FROM flink:1.20-java11
USER root
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY streaming_fraud.py api.py ./
ENTRYPOINT ["python3", "streaming_fraud.py"]
CMD ["--kafka-brokers", "kafka:9092"]
