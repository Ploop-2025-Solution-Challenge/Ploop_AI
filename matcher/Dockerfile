FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Debian trixie 기준: libgl1 사용
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
  && rm -rf /var/lib/apt/lists/*

# requirements만 먼저 복사 → 설치 → 검증
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python - <<'PY'
import sys
print("PY:", sys.version)
import pandas
print("OK pandas:", pandas.__version__)
PY

# 앱 복사
COPY . .

EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]