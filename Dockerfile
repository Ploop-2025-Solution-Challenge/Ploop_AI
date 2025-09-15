# 1. Python 기반 slim 이미지 사용
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# 4. requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt \
 && useradd -m appuser

# 5. 전체 코드 복사 (모델 포함)
COPY . .
USER appuser

# 6. FastAPI 서버 포트 설정 (8000)
EXPOSE 8000

# app:app = <모듈>:<애플리케이션 객체>
# CMD ["gunicorn","app:app","-b","0.0.0.0:8000","--workers","2"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# main:app 이었는데 app.py이므로 app:app으로 변경
