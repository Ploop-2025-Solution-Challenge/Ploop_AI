import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import tempfile
import base64
from matcher import (
    process_new_user,
    init_pinecone,
    get_mysql_connection,
    # setup_scheduler,            # ⬅️ 스케줄러 사용 안 함 (주석)
    weekly_matching_process,       # 매칭을 POST에서만 실행
)
# import threading               # ⬅️ 스케줄러 스레드 사용 안 함 (주석)
from datetime import datetime


# FastAPI 앱 생성
app = FastAPI(title="쓰레기 감지 API", description="YOLO 모델을 사용한 쓰레기 객체 감지 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 결과물 저장 디렉토리 설정
RESULTS_DIR = "detection_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pydantic 모델 - URL을 통한 이미지 요청
class ImageUrlRequest(BaseModel):
    image_url: str

# Pydantic 모델 정의
class NewUserRequest(BaseModel):
    user_id: int

# -----------------------------
# 앱 시작 시 스케줄러 동작 제거
# -----------------------------
# @app.on_event("startup")
# async def startup_event():
#     # Pinecone 초기화 (필요 시)
#     init_pinecone()
#
#     # 스케줄러를 별도 스레드로 실행 (사용 안 함)
#     scheduler_thread = threading.Thread(target=setup_scheduler)
#     scheduler_thread.daemon = True
#     scheduler_thread.start()


@app.post("/api/new_user")
async def handle_new_user(request: NewUserRequest):
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    try:
        mysql_conn = get_mysql_connection()
        pinecone_idx = init_pinecone()
        process_new_user(request.user_id, mysql_conn, pinecone_idx)
        mysql_conn.close()
        return {"status": "success", "message": "User processed successfully"}
    except Exception as e:
        # 에러는 fail 형태로 반환해도 되지만 기존 엔드포인트는 유지
        raise HTTPException(status_code=500, detail=str(e))


# ✅ 매칭 알고리즘 수동 실행 (POST로만 트리거)
@app.post("/api/run_matching")
async def run_matching():
    """
    POST로 호출 시에만 매칭 알고리즘(weekly_matching_process)을 즉시 실행합니다.
    성공 시: {"status":"success","processed_at":"..."}
    실패 시: {"status":"fail","error":"...","processed_at":"..."}
    """
    try:
        mysql_conn = get_mysql_connection()
        pinecone_idx = init_pinecone()
        try:
            weekly_matching_process(connection=mysql_conn, pinecone_index=pinecone_idx)
        finally:
            try:
                mysql_conn.close()
            except Exception:
                pass

        return {
            "status": "success",
            "processed_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "fail",
            "error": str(e),
            "processed_at": datetime.now().isoformat()
        }


@app.get("/")
async def root():
    """
    API 사용 안내 페이지
    """
    return {
        "message": "쓰레기 감지 API에 오신 것을 환영합니다!",
        "endpoints": {
            "/detect/url": "이미지 URL을 통한 객체 감지",
            "/detect/upload": "이미지 파일 업로드를 통한 객체 감지",
            "/api/new_user": "신규 유저 임베딩 반영",
            "/api/run_matching": "POST로 매칭 알고리즘 즉시 실행"  # ⬅️ POST로만 동작
        }
    }

if __name__ == "__main__":
    # 서버는 평시엔 스케줄러 없이 대기,
    # 매칭은 /api/run_matching 호출 시에만 수행됩니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)
