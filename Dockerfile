FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

WORKDIR /app

COPY selector.py /app/selector.py
COPY coverage_map.json.gz /app/coverage_map.json.gz

RUN chmod 0644 /app/selector.py /app/coverage_map.json.gz

# The scorer runs the container as uid 65534 with a read-only root FS.
USER nobody

# Reads /data/tests.json and /data/commits/*.patch, prints JSONL to stdout.
ENTRYPOINT ["python", "selector.py", "--data-dir", "/data"]
