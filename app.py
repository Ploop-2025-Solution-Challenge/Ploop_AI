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

# 비즈니스 로직 모듈 (경로 탐색)
from google_route import handle_compute, invalid_json_response

# GPT기반 탐지 모듈
from trash_detection import TrashDetector, DetectionError

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
