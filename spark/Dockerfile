FROM bitnami/spark:3.5
USER root
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY batch_features.py api.py ./
ENTRYPOINT ["spark-submit", "--master", "local[*]", "batch_features.py"]
CMD ["--csv", "/data/transactions.csv", "--output", "/data/spark_output"]
