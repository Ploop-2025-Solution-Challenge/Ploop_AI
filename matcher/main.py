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
from matcher import process_new_user, init_pinecone, get_mysql_connection, setup_scheduler
# from detector import detect_trash
import threading


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

# 애플리케이션 시작 시 스케줄러 실행
@app.on_event("startup")
async def startup_event():
    # Pinecone 초기화
    init_pinecone()

    # 스케줄러를 별도 스레드로 실행
    scheduler_thread = threading.Thread(target=setup_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()



# @app.post("/detect/upload")
# async def detect_from_upload(file: UploadFile = File(...)):
#     """
#     업로드된 이미지 파일에서 객체 감지를 수행합니다.
#     """
#     try:
#         # Get the file extension from the uploaded file
#         file_extension = os.path.splitext(file.filename)[1]

#         # Create a temporary file with the correct extension
#         with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
#             temp_file_path = temp_file.name

#             # Save the uploaded file content
#             content = await file.read()
#             temp_file.write(content)

#         try:
#             # 객체 감지 수행
#             json_result, result_plot, _ = detect_trash(temp_file_path)

#             # 결과 이미지 저장
#             result_filename = f"{RESULTS_DIR}/result_{file.filename}"
#             cv2.imwrite(result_filename, result_plot)

#             # 이미지를 Base64로 인코딩
#             _, buffer = cv2.imencode('.jpg', result_plot)
#             img_base64 = base64.b64encode(buffer).decode('utf-8')

#             return {
#                 "detection_results": json_result,
#                 "result_image_base64": f"data:image/jpeg;base64,{img_base64}"
#             }

#         finally:
#             # 임시 파일 삭제
#             os.unlink(temp_file_path)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"객체 감지 중 오류 발생: {str(e)}")



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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    API 사용 안내 페이지
    """
    return {
        "message": "쓰레기 감지 API에 오신 것을 환영합니다!",
        "endpoints": {
            "/detect/url": "이미지 URL을 통한 객체 감지",
            "/detect/upload": "이미지 파일 업로드를 통한 객체 감지"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)