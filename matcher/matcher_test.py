import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from datetime import datetime
import mysql.connector
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("matcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger("matcher")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "member-embeddings")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "member_matching_db")


def init_pinecone():
    """Pinecone v3 서비스 초기화 및 인덱스 연결"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise


def get_mysql_connection():
    """MySQL 데이터베이스 연결 생성"""
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            port=3306,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
        )
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to MySQL: {str(e)}")
        raise


def load_user_data(conn):
    """user + user_location_preferences 데이터를 통합"""
    query = """
    SELECT u.id AS user_id,
           u.region,
           u.motivation,
           u.difficulty,
           p.preference
    FROM user u
    LEFT JOIN user_location_preferences p
      ON u.id = p.user_id
    ORDER BY u.id;
    """
    df = pd.read_sql(query, conn)

    if df.empty:
        logger.warning("⚠️ No user data found.")
        return pd.DataFrame()

    # user별 preference 묶기
    user_data = (
        df.groupby("user_id", dropna=False)
          .agg(
              region=("region", "first"),
              motivation=("motivation", "first"),
              difficulty=("difficulty", "first"),
              preference=("preference", lambda s: [v for v in s if pd.notnull(v)])
          )
          .reset_index()
          .sort_values("user_id")
    )

    return user_data


def prepare_text_for_embedding(row):
    """유저 데이터로부터 임베딩용 텍스트 생성"""
    prefs = ", ".join([str(p) for p in row["preference"]]) if row["preference"] else "없음"
    text = (
        f"지역: {row['region'] if pd.notnull(row['region']) else '없음'}, "
        f"동기: {row['motivation'] if pd.notnull(row['motivation']) else '없음'}, "
        f"난이도: {row['difficulty'] if pd.notnull(row['difficulty']) else '없음'}, "
        f"선호지역: {prefs}"
    )
    return text


def save_embeddings_to_pinecone(user_ids, embeddings, pinecone_index):
    """생성된 임베딩을 Pinecone에 저장"""
    vectors_to_upsert = []
    for i, embedding in enumerate(embeddings):
        user_id = str(user_ids[i])
        vectors_to_upsert.append(
            {
                "id": user_id,
                "values": embedding.tolist(),
                "metadata": {"updated_at": datetime.now().isoformat()},
            }
        )

    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i: i + batch_size]
        pinecone_index.upsert(vectors=batch)

    logger.info(f"Saved {len(user_ids)} embeddings to Pinecone")


def run_matching():
    """매칭 테스트 실행"""
    logger.info("🚀 매칭 프로세스 시작")

    conn = get_mysql_connection()
    pinecone_index = init_pinecone()

    try:
        user_data = load_user_data(conn)
        if user_data.empty:
            return

        # embedding text 생성
        user_data["embedding_text"] = user_data.apply(prepare_text_for_embedding, axis=1)

        # 모델 임베딩 생성
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        embeddings = model.encode(user_data["embedding_text"].tolist())

        # Pinecone 저장
        save_embeddings_to_pinecone(user_data["user_id"].tolist(), embeddings, pinecone_index)

        # 유사도 행렬 계산
        sim_matrix = cosine_similarity(embeddings)

        logger.info("✅ 매칭 프로세스 완료")
        print("\n=== Sample Similarity Matrix ===")
        print(pd.DataFrame(sim_matrix, index=user_data["user_id"], columns=user_data["user_id"]))

    finally:
        conn.close()


if __name__ == "__main__":
    run_matching()
