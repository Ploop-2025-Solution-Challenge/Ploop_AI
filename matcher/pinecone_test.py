# test_pinecone_upsert.py
import os
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import pinecone

# 1) 환경변수 로드 (.env에 아래 3개가 있어야 함)
# PINECONE_API_KEY=...
# PINECONE_ENVIRONMENT=us-east-1
# PINECONE_INDEX=member-embeddings
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY","API키를 입력하세요")
ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "member-embeddings")

assert API_KEY, "PINECONE_API_KEY 가 비어 있습니다"
assert INDEX_NAME, "PINECONE_INDEX 가 비어 있습니다"

# 2) Pinecone 초기화 및 인덱스 객체 만들기
pinecone.init(api_key=API_KEY, environment=ENV)
index = pinecone.Index(INDEX_NAME)

# 3) 768차원 임의 유저 벡터 생성 (cosine metric -> 정규화 권장)
user_id = "test-user-1"
np.random.seed(42)
vec = np.random.randn(768).astype(np.float32)
vec = vec / np.linalg.norm(vec)  # L2 normalize for cosine

metadata = {
    "name": "Dummy User",
    "role": "tester",
    "updated_at": datetime.now().isoformat()
}

# 4) upsert
print(f"[UPsert] id={user_id}, dim={len(vec)} to index='{INDEX_NAME}'")
index.upsert(vectors=[(user_id, vec.tolist(), metadata)])

# 5) 약간 대기 (가끔 일관 조회를 위해 필요)
time.sleep(1.5)

# 6) fetch로 존재 확인
fetch_res = index.fetch(ids=[user_id])
print("\n[Fetch] result keys:", list(fetch_res.vectors.keys()))
if user_id in fetch_res.vectors:
    fetched = fetch_res.vectors[user_id]
    print(f"  - dim={len(fetched['values'])}, metadata={fetched.get('metadata')}")

# 7) query로 유사도 확인 (자기 자신으로 질의 → score 가 가장 높아야 함)
query_res = index.query(
    vector=vec.tolist(),
    top_k=3,
    include_values=False,
    include_metadata=True
)
print("\n[Query] top_k=3")
for m in query_res.matches:
    print(f"  - id={m['id']}, score={m['score']:.4f}, metadata={m.get('metadata')}")

print("\nDone.")
