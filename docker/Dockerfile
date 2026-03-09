FROM python:3.11-slim

LABEL maintainer="Gabriel Demetrios Lafis <gabriel.lafis@gmail.com>"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
