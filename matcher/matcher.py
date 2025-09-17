import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import schedule
import time
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


# ---------------------------
# Pinecone / MySQL 초기화
# ---------------------------
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


# ---------------------------
# MySQL 로딩 (실제 스키마 기준)
# ---------------------------
def load_user_data(connection):
    """
    user + user_location_preferences 데이터를 통합해
    user별로 preference 리스트를 만든 DataFrame을 반환.
    NULL 값도 보존.
    """
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
    df = pd.read_sql(query, connection)

    if df.empty:
        logger.warning("⚠️ No user data found.")
        return pd.DataFrame()

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


def get_single_user(user_id: int, connection):
    """단일 유저(user + preferences) 로드 (NULL 허용, preference는 리스트)"""
    query = """
    SELECT u.id AS user_id,
           u.region,
           u.motivation,
           u.difficulty,
           p.preference
      FROM user u
 LEFT JOIN user_location_preferences p
        ON u.id = p.user_id
     WHERE u.id = %s
     ORDER BY u.id;
    """
    df = pd.read_sql(query, connection, params=[user_id])
    if df.empty:
        return None

    prefs = [v for v in df["preference"] if pd.notnull(v)]
    row = df.iloc[0]
    return {
        "user_id": int(row["user_id"]),
        "region": row["region"],
        "motivation": row["motivation"],
        "difficulty": row["difficulty"],
        "preference": prefs,
    }


# ---------------------------
# 임베딩 텍스트 & Pinecone 업서트
# ---------------------------
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


def save_embeddings_to_pinecone(ids, embeddings, pinecone_index):
    """생성된 임베딩을 Pinecone에 저장 (v3 dict 포맷)"""
    vectors_to_upsert = []
    for i, emb in enumerate(embeddings):
        vectors_to_upsert.append(
            {
                "id": str(ids[i]),
                "values": emb.tolist(),
                "metadata": {"updated_at": datetime.now().isoformat()},
            }
        )
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        pinecone_index.upsert(vectors=vectors_to_upsert[i:i+batch_size])

    logger.info(f"Saved {len(ids)} embeddings to Pinecone")


# ---------------------------
# 매칭 / 저장 유틸
# ---------------------------
def create_pair_matches(similarity_matrix, user_df):
    """
    각 유저를 최대 1명과만 매칭하는 알고리즘 (그리디, disjoint pairs).
    반환: pairs(list of tuples[(i_idx, j_idx, score), ...])
    """
    num = similarity_matrix.shape[0]
    all_pairs = []
    for i in range(num):
        for j in range(i + 1, num):
            all_pairs.append((i, j, float(similarity_matrix[i, j])))
    all_pairs.sort(key=lambda x: x[2], reverse=True)

    used = set()
    pairs = []

    for i, j, score in all_pairs:
        if i not in used and j not in used:
            pairs.append((i, j, score))
            used.add(i)
            used.add(j)

    # 매칭되지 않은 유저는 더 이상 고려하지 않음(요구사항)
    return pairs


def save_matches_to_db(pairs, user_df, connection):
    """
    1:1 매칭 결과를 team 테이블에 저장.
    - 같은 주에 이미 팀이 있는 유저는 제외
    - (user_id_1, user_id_2) 정규화 후 중복 저장 방지
    """
    cursor = connection.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    year, week, _ = datetime.now().isocalendar()
    week_str = f"{year}-W{week:02d}"

    cursor.execute(
        """
        SELECT user_id_1, user_id_2
          FROM team
         WHERE week = %s
        """,
        (week_str,),
    )
    rows = cursor.fetchall()
    already_paired = set()
    for (u1, u2) in rows:
        if u1 is not None:
            already_paired.add(int(u1))
        if u2 is not None:
            already_paired.add(int(u2))

    inserted_pairs = set()
    n = 0

    for i_idx, j_idx, _score in pairs:
        uid1 = int(user_df.iloc[i_idx]["user_id"])
        uid2 = int(user_df.iloc[j_idx]["user_id"])
        if uid1 in already_paired or uid2 in already_paired:
            continue

        a, b = (uid1, uid2) if uid1 < uid2 else (uid2, uid1)
        if (a, b) in inserted_pairs:
            continue

        cursor.execute(
            """
            INSERT INTO team (created_at, user_id_1, user_id_2, week)
            VALUES (%s, %s, %s, %s)
            """,
            (created_at, a, b, week_str),
        )
        inserted_pairs.add((a, b))
        already_paired.add(a)
        already_paired.add(b)
        n += 1

    connection.commit()
    cursor.close()
    logger.info(f"Saved {n} teams to database (week={week_str})")


# ---------------------------
# 주간(또는 즉시) 매칭 프로세스
# ---------------------------
def weekly_matching_process(connection=None, pinecone_index=None):
    """일정과 무관하게 호출하면 즉시 실행됨"""
    logger.info(f"매칭 프로세스 시작: {datetime.now()}")

    close_conn = False
    if connection is None:
        connection = get_mysql_connection()
        close_conn = True
    if pinecone_index is None:
        pinecone_index = init_pinecone()

    try:
        user_df = load_user_data(connection)
        if user_df.empty:
            logger.warning("No users to process.")
            return

        user_df["embedding_text"] = user_df.apply(prepare_text_for_embedding, axis=1)

        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        embeddings = model.encode(user_df["embedding_text"].tolist())

        save_embeddings_to_pinecone(user_df["user_id"].tolist(), embeddings, pinecone_index)

        sim = cosine_similarity(embeddings)
        pairs = create_pair_matches(sim, user_df)

        save_matches_to_db(pairs, user_df, connection)

        logger.info(f"매칭 프로세스 완료: {datetime.now()}")
    except Exception as e:
        logger.error(f"매칭 프로세스 오류: {str(e)}")
        raise
    finally:
        if close_conn and connection:
            connection.close()


# ---------------------------
# 신규 유저 등록 처리
# ---------------------------
def process_new_user(new_user_id, mysql_connection=None, pinecone_index=None):
    """
    요구사항에 따라 waiting(대기열) 사용 없음.
    - 신규 유저 임베딩 생성 및 Pinecone에 업서트만 수행
    - 즉시 매칭 시도/대기열 편입 없음
    """
    logger.info(f"새 유저 처리 시작: {new_user_id}, {datetime.now()}")

    close_conn = False
    if mysql_connection is None:
        mysql_connection = get_mysql_connection()
        close_conn = True
    if pinecone_index is None:
        pinecone_index = init_pinecone()

    try:
        # 새 유저 데이터 로드
        new_user = get_single_user(new_user_id, mysql_connection)
        if not new_user:
            raise ValueError(f"user {new_user_id} not found")

        # 새 유저 임베딩 생성 & 업서트
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        emb_text = prepare_text_for_embedding(new_user)
        new_emb = model.encode([emb_text])[0]

        pinecone_index.upsert(
            vectors=[{
                "id": str(new_user_id),
                "values": new_emb.tolist(),
                "metadata": {"updated_at": datetime.now().isoformat()}
            }]
        )

        logger.info(f"새 유저 {new_user_id} 임베딩 저장 완료 (waiting 미사용)")
        logger.info(f"새 유저 처리 완료: {new_user_id}, {datetime.now()}")
    except Exception as e:
        logger.error(f"새 유저 처리 오류: {str(e)}")
        raise
    finally:
        if close_conn and mysql_connection:
            mysql_connection.close()


# ---------------------------
# 스케줄러 (원하면 사용)
# ---------------------------
def setup_scheduler():
    logger.info("스케줄러 설정 시작")
    schedule.every().monday.at("02:00").do(weekly_matching_process)
    logger.info("매주 월요일 02:00에 매칭 실행 예약됨")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    # 스케줄 없이 즉시 실행도 가능:
    weekly_matching_process()
