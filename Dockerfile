FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

WORKDIR /app

RUN pip install --no-cache-dir \
    duckdb \
    python-dateutil \
    pytz \
    numpy \
    pandas \
    typing_extensions

COPY selector.py /app/selector.py
COPY coverage_map.pkl.gz /app/coverage_map.pkl.gz
COPY sqlglot_src.tar.gz /tmp/sqlglot_src.tar.gz

RUN mkdir -p /app/sqlglot_src && \
    tar -xzf /tmp/sqlglot_src.tar.gz -C /app/sqlglot_src && \
    rm -f /tmp/sqlglot_src.tar.gz && \
    python -m compileall -q /app/sqlglot_src && \
    chmod 0644 /app/selector.py /app/coverage_map.pkl.gz && \
    chmod -R a+rX /app/sqlglot_src

# The scorer runs the container as uid 65534 with a read-only root FS.
USER nobody

# Reads /data/tests.json and /data/commits/*.patch, prints JSONL to stdout.
ENTRYPOINT ["python", "selector.py", "--data-dir", "/data"]
