# pinecone_v3_test.py
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX", "member-embeddings")

# v3 클라이언트 초기화
pc = Pinecone(api_key=api_key)

# 인덱스 연결
index = pc.Index(index_name)
print("✅ Connected to index:", index_name)

# 1. 벡터 생성
vec = np.random.randn(768).astype(np.float32)
vec /= np.linalg.norm(vec)
test_id = "test-vec-1"
metadata = {"source": "pinecone_v3_test.py", "updated_at": datetime.now().isoformat()}

# 2. 업서트
index.upsert(vectors=[{"id": test_id, "values": vec.tolist(), "metadata": metadata}])
print("⬆️ Upsert ok")

# 3. Fetch
fetched = index.fetch(ids=[test_id])
print("📦 Fetch:", fetched)

# 4. Query
res = index.query(vector=vec.tolist(), top_k=3, include_metadata=True)
print("🔎 Query result:")
for m in res["matches"]:
    print(f" - id={m['id']}, score={m['score']:.4f}, metadata={m.get('metadata')}")
