FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install poetry

COPY pyproject.toml poetry.lock* /app/

RUN pip install llama-cpp-python==0.3.2

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

COPY . /app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "./main.py"]