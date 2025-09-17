#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI 서버
- /route/compute : 기존 경로 탐색(google_route.py)
- /detect        : GPT-기반 쓰레기(6종) 개수 탐지
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

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
    # process_new_user, # ⬅️ process_new_user 사용 안 함 (주석)
    init_pinecone,
    get_mysql_connection,
    setup_scheduler,            # ⬅️ 스케줄러 사용 안 함 (주석)
    weekly_matching_process,       # 매칭을 POST에서만 실행
)
# import threading               # ⬅️ 스케줄러 스레드 사용 안 함 (주석)
from datetime import datetime

# 비즈니스 로직 모듈 (경로 탐색)
from google_route import handle_compute, invalid_json_response

# GPT기반 탐지 모듈
from trash_detection import TrashDetector, DetectionError
import threading
app = FastAPI(title="Trash Routing & Detection Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 개발 편의. 배포 시 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detector는 서버 기동 시 1회 로드하여 재사용
detector = TrashDetector(model="gpt-4o")  # 필요시 gpt-4o-mini 등으로 변경 가능

# Pydantic 모델 - URL을 통한 이미지 요청
class ImageUrlRequest(BaseModel):
    image_url: str

# Pydantic 모델 정의
class NewUserRequest(BaseModel):
    user_id: int

@app.on_event("startup")
async def startup_event():
    # Pinecone 초기화 (필요 시)
    init_pinecone()

    # 스케줄러를 별도 스레드로 실행 
    scheduler_thread = threading.Thread(target=setup_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

# @app.post("/api/new_user")
# async def handle_new_user(request: NewUserRequest):
#     if not request.user_id:
#         raise HTTPException(status_code=400, detail="user_id required")

#     try:
#         mysql_conn = get_mysql_connection()
#         pinecone_idx = init_pinecone()
#         process_new_user(request.user_id, mysql_conn, pinecone_idx)
#         mysql_conn.close()
#         return {"status": "success", "message": "User processed successfully"}
#     except Exception as e:
#         # 에러는 fail 형태로 반환해도 되지만 기존 엔드포인트는 유지
#         raise HTTPException(status_code=500, detail=str(e))


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


# ------------------------- 경로 탐색 -------------------------
@app.post("/route/compute")
async def compute_route(req: Request):
    """
    통신(HTTP) 레이어만 담당.
    - 요청 JSON 파싱
    - 비즈니스 로직 모듈(google_route.handle_compute) 호출
    - JSONResponse 반환
    """
    try:
        payload = await req.json()
    except Exception:
        # JSON 파싱 실패 시 통일된 에러 응답
        return invalid_json_response()

    return handle_compute(payload)

# ------------------------- GPT 탐지 -------------------------
@app.post("/detect")
async def detect_endpoint(req: Request):
    """
    요청(JSON):
      { "image": "<base64 string>" }

    응답(JSON) 성공:
      {
        "success": true,
        "results": {
          "can": 0,
          "plastic_bottle": 0,
          "bottle_cap": 0,
          "paper_cup": 0,
          "plastic_bag": 0,
          "trash_bin": 0
        }
      }

    응답(JSON) 실패:
      { "success": false, "msg": "<이유>" }
    """
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"success": False, "msg": "invalid JSON body"}, status_code=200)

    if not isinstance(body, dict):
        return JSONResponse({"success": False, "msg": "body must be a JSON object"}, status_code=200)

    image_b64 = body.get("image")
    if not image_b64 or not isinstance(image_b64, str):
        return JSONResponse({"success": False, "msg": "field 'image' (base64 string) is required"}, status_code=200)

    try:
        result = detector.detect_counts_from_base64(image_b64=image_b64)
        return JSONResponse(result, status_code=200)
    except DetectionError as e:
        return JSONResponse({"success": False, "msg": str(e)}, status_code=200)
    except Exception as e:
        return JSONResponse({"success": False, "msg": f"internal error: {e}"}, status_code=200)


if __name__ == "__main__":
    # 로컬 실행
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
