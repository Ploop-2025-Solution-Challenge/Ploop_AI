# -*- coding: utf-8 -*-

import os
import re
import base64
from typing import Dict, Any

from pydantic import BaseModel, Field

# OpenAI Python SDK (Responses API)
# pip install openai>=1.40
from openai import OpenAI

# ------------------ 고정 프롬프트 ------------------
SYSTEM_PROMPT = """You are a highly advanced AI specializing in waste detection within images.

Your primary mission is to meticulously scan the provided image and identify specific types of trash, as well as trash bins. You must accurately count the quantities of the following six objects: can, plastic bottle, bottle cap, paper cup, plastic bag, and trash bin.

Every object must be accounted for. If an object from the list is not visible in the image, you must report its count as 0. Do not detect any other objects besides the ones specified.

Your final output must be delivered ONLY in the JSON format provided below. Do not include any additional text, comments, or explanations.

Example Output:

{
  "results": {
    "can": 2,
    "plastic_bottle": 0,
    "bottle_cap": 5,
    "paper_cup": 1,
    "plastic_bag": 0,
    "trash_bin": 1
  }
}
"""

USER_PROMPT = "Please analyze this image and return ONLY the JSON in the specified format."

# ------------------ 파싱용 스키마 ------------------
class Results(BaseModel):
    can: int
    plastic_bottle: int
    bottle_cap: int
    paper_cup: int
    plastic_bag: int
    trash_bin: int

class WasteResults(BaseModel):
    results: Results


    

# ------------------ 유틸 ------------------
def _strip_data_uri_prefix(b64: str) -> str:
    """
    data:image/png;base64,XXXX 같은 접두사가 있으면 제거
    """
    return re.sub(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", "", b64)

# ------------------ 예외 ------------------
class DetectionError(Exception):
    """탐지 파이프라인에서 사용자에게 보여줄 에러 메시지"""
    pass

# ------------------ Detector ------------------
class TrashDetector:
    """
    GPT(Visual) 기반 쓰레기 탐지기
    - 6개 항목의 '개수'만 반환
    """
    def __init__(self, model: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        ################# GPT API KEY 값 넣는 곳!!#################
        # api_key=""
        
        if not api_key:
            raise DetectionError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def detect_counts_from_base64(self, image_b64: str) -> Dict[str, Any]:
        """
        입력: base64 이미지 문자열
        출력(JSON dict):
          { "success": True, "results": {...6개 항목...} }
        실패:
          { "success": False, "msg": ... }  (app.py에서 래핑)
        """
        if not image_b64 or not isinstance(image_b64, str):
            raise DetectionError("image(base64) is empty")

        # data URL 접두사 제거(있을 수도, 없을 수도 있음)
        raw = _strip_data_uri_prefix(image_b64)

        # base64 유효성 간단 확인
        try:
            # decode만 해보고 바로 버림 — 전송은 data URL로
            base64.b64decode(raw, validate=True)
        except Exception:
            raise DetectionError("invalid base64 image")

        data_url = f"data:image/jpeg;base64,{raw}"  # MIME은 jpeg로 통일(대부분 문제없음)

        # OpenAI Responses API (구조화 파싱)
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": SYSTEM_PROMPT}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": USER_PROMPT},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    },
                ],
                text_format=WasteResults,  # Pydantic으로 파싱
            )
        except Exception as e:
            raise DetectionError(f"gpt api request failed: {e}")

        try:
            parsed: WasteResults = response.output_parsed
            r = parsed.results  # Results 모델 인스턴스

            results = {
                "can": int(r.can),
                "plastic_bottle": int(r.plastic_bottle),
                "bottle_cap": int(r.bottle_cap),
                "paper_cup": int(r.paper_cup),
                "plastic_bag": int(r.plastic_bag),
                "trash_bin": int(r.trash_bin),
            }
        except Exception as e:
            raise DetectionError(f"failed to parse gpt output: {e}")

        return {"success": True, "results": results}
