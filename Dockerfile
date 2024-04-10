#FROM python:3.10.6
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /subsai

COPY requirements.txt .

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends --fix-missing git gcc mono-mcs && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY ./src ./src
COPY ./assets ./assets

RUN pip install .

# ENTRYPOINT ["python", "src/subsai/webui.py", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]

