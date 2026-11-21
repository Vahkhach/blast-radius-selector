FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    SQLGLOT_SRC=/opt/sqlglot

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tobymao/sqlglot.git /opt/sqlglot \
    && git -C /opt/sqlglot checkout eb11aec4be8cb2122afe713a4816662c3f0fb2fa

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements.lock \
    && python -m pip install --no-cache-dir --no-deps /opt/sqlglot \
    && rm -f /tmp/requirements.lock

WORKDIR /app
COPY --chown=nobody:nogroup selector.py /app/selector.py
COPY --chown=nobody:nogroup coverage_map.pkl.gz /app/coverage_map.pkl.gz

RUN find /opt/sqlglot -type d -name __pycache__ -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/sqlglot

USER 65534:65534

ENTRYPOINT ["python", "selector.py", "--data-dir", "/data"]
