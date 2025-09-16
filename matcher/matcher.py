import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pinecone
import schedule
import time
from datetime import datetime
import mysql.connector
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("matcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("matcher")


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "member_embeddings")


MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "member_matching_db")


def init_pinecone():
    """Pinecone 서비스 초기화 및 인덱스 연결"""
    try:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

        return pinecone.Index(PINECONE_INDEX)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise


def get_mysql_connection():
    """MySQL 데이터베이스 연결 생성"""
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to MySQL: {str(e)}")
        raise


def prepare_text_for_embedding(row):
    """멤버 데이터로부터 임베딩용 텍스트 생성"""
    text = f"위치: {row['location']}, 선호지역: {row['preferred_location']}, "
    text += f"동기: {row['motivation']}, 관심사: {row['interests']}"
    return text


def save_embeddings_to_pinecone(member_ids, embeddings, pinecone_index):
    """생성된 임베딩을 Pinecone에 저장"""
    vectors_to_upsert = []

    for i, embedding in enumerate(embeddings):
        member_id = str(member_ids[i])
        vectors_to_upsert.append((member_id, embedding.tolist(), {"updated_at": datetime.now().isoformat()}))


    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        pinecone_index.upsert(vectors=batch)

    logger.info(f"Saved {len(member_ids)} embeddings to Pinecone")


def get_user_data(user_id, connection):
    """특정 사용자의 데이터를 MySQL에서 가져옴"""
    cursor = connection.cursor(dictionary=True)
    user_query = """
    SELECT m.*, l.preferred_location, mo.motivation, m.interests 
    FROM members m 
    JOIN member_location_preference l ON m.member_id = l.member_id 
    JOIN member_motivation mo ON m.member_id = mo.member_id 
    WHERE m.member_id = %s
    """
    cursor.execute(user_query, (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    return user_data


def create_pair_matches(similarity_matrix, member_data):
    """각 멤버마다 정확히 2명과 매칭하는 알고리즘"""
    num_members = similarity_matrix.shape[0]
    available_members = set(range(num_members))
    matches = {}
    waiting_list = []


    all_pairs = []
    for i in range(num_members):
        for j in range(i+1, num_members):
            all_pairs.append((i, j, similarity_matrix[i][j]))

    all_pairs.sort(key=lambda x: x[2], reverse=True)


    match_count = {i: 0 for i in range(num_members)}


    for i, j, score in all_pairs:
        if i in available_members and j in available_members and match_count[i] < 2 and match_count[j] < 2:
            if i not in matches:
                matches[i] = []
            matches[i].append((j, score))
            match_count[i] += 1

            if j not in matches:
                matches[j] = []
            matches[j].append((i, score))
            match_count[j] += 1

            if match_count[i] == 2:
                available_members.remove(i)
            if match_count[j] == 2:
                available_members.remove(j)

    for member_idx in range(num_members):
        if match_count[member_idx] < 2:
            member_id = member_data.iloc[member_idx]['member_id']
            waiting_list.append(member_id)

    return matches, waiting_list


def save_matches_to_db(matches, member_data, connection):
    """매칭 결과를 MySQL에 저장"""
    cursor = connection.cursor()


    cursor.execute("UPDATE member_matches SET status = 'inactive' WHERE status = 'active'")

    match_count = 0
    for idx, match_list in matches.items():
        member_id = member_data.iloc[idx]['member_id']

        for match_idx, score in match_list:
            matched_member_id = member_data.iloc[match_idx]['member_id']

            query = """
            INSERT INTO member_matches 
            (member_id, matched_member_id, similarity_score, status) 
            VALUES (%s, %s, %s, 'active')
            """
            cursor.execute(query, (member_id, matched_member_id, float(score)))
            match_count += 1

    connection.commit()
    cursor.close()
    logger.info(f"Saved {match_count} matches to database")


def update_waiting_queue(waiting_list, member_data, connection):
    """대기열 정보를 MySQL에 업데이트"""
    cursor = connection.cursor()

    cursor.execute("UPDATE waiting_queue SET status = 'processed' WHERE status = 'waiting'")

    for member_id in waiting_list:
        query = """
        INSERT INTO waiting_queue (member_id, status) 
        VALUES (%s, 'waiting')
        ON DUPLICATE KEY UPDATE status = 'waiting', joined_at = CURRENT_TIMESTAMP
        """
        cursor.execute(query, (member_id,))

    connection.commit()
    cursor.close()
    logger.info(f"Updated waiting queue with {len(waiting_list)} members")

def weekly_matching_process(connection=None, pinecone_index=None):
    """일주일에 한 번 실행되는 전체 매칭 프로세스"""
    logger.info(f"주간 매칭 프로세스 시작: {datetime.now()}")

    close_connection = False
    if connection is None:
        connection = get_mysql_connection()
        close_connection = True

    if pinecone_index is None:
        pinecone_index = init_pinecone()

    try:
        members_df = pd.read_sql("SELECT * FROM members WHERE status = 'active'", connection)
        location_prefs_df = pd.read_sql("SELECT * FROM member_location_preference", connection)
        motivation_df = pd.read_sql("SELECT * FROM member_motivation", connection)

        waiting_queue_df = pd.read_sql("SELECT * FROM waiting_queue WHERE status = 'waiting'", connection)

        member_data = pd.merge(members_df, location_prefs_df, on='member_id')
        member_data = pd.merge(member_data, motivation_df, on='member_id')

        logger.info(f"Processing {len(member_data)} members for matching")

        member_data['embedding_text'] = member_data.apply(prepare_text_for_embedding, axis=1)

        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        embeddings = model.encode(member_data['embedding_text'].tolist())

        # 임베딩을 Pinecone에 저장
        save_embeddings_to_pinecone(member_data['member_id'].tolist(), embeddings, pinecone_index)

        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(embeddings)

        # 매칭 알고리즘 실행
        matches, waiting_list = create_pair_matches(similarity_matrix, member_data)

        # 매칭 결과 저장
        save_matches_to_db(matches, member_data, connection)

        # 대기열 업데이트
        update_waiting_queue(waiting_list, member_data, connection)

        logger.info(f"주간 매칭 프로세스 완료: {datetime.now()}")
    except Exception as e:
        logger.error(f"주간 매칭 프로세스 오류: {str(e)}")
        raise
    finally:
        if close_connection and connection:
            connection.close()

# 새 유저 등록 처리 함수
def process_new_user(new_user_id, mysql_connection=None, pinecone_index=None):
    """새로운 유저 등록 시 처리 로직"""
    logger.info(f"새 유저 처리 시작: {new_user_id}, {datetime.now()}")

    # 연결이 전달되지 않은 경우 새로 생성
    close_connection = False
    if mysql_connection is None:
        mysql_connection = get_mysql_connection()
        close_connection = True

    if pinecone_index is None:
        pinecone_index = init_pinecone()

    try:
        cursor = mysql_connection.cursor(dictionary=True)

        # 대기열 확인
        cursor.execute("SELECT member_id FROM waiting_queue WHERE status = 'waiting' ORDER BY joined_at ASC")
        waiting_members = cursor.fetchall()

        if not waiting_members:
            # 대기열이 비어있으면 새 유저를 대기열에 추가
            # 새 유저의 임베딩 생성 및 Pinecone에 저장
            new_user_data = get_user_data(new_user_id, mysql_connection)
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            embedding_text = prepare_text_for_embedding(new_user_data)
            new_user_embedding = model.encode([embedding_text])[0]

            # Pinecone에 임베딩 저장
            pinecone_index.upsert([(str(new_user_id), new_user_embedding.tolist(),
                                    {"updated_at": datetime.now().isoformat()})])

            # 대기열에 추가
            query = "INSERT INTO waiting_queue (member_id, status) VALUES (%s, 'waiting')"
            cursor.execute(query, (new_user_id,))
            mysql_connection.commit()

            logger.info(f"새 유저 {new_user_id}가 대기열에 추가되었습니다.")
        else:
            # 대기열에 사람이 있으면 가장 오래 기다린 사람과 매칭
            waiting_member_id = waiting_members[0]['member_id']

            # 새 유저 데이터 가져오기
            new_user_data = get_user_data(new_user_id, mysql_connection)

            # 새 유저의 임베딩 생성
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            embedding_text = prepare_text_for_embedding(new_user_data)
            new_user_embedding = model.encode([embedding_text])[0]

            # Pinecone에 새 유저 임베딩 저장
            pinecone_index.upsert([(str(new_user_id), new_user_embedding.tolist(),
                                    {"updated_at": datetime.now().isoformat()})])

            # Pinecone에서 대기 유저의 임베딩 가져오기
            query_response = pinecone_index.query(
                vector=new_user_embedding.tolist(),
                filter={"id": str(waiting_member_id)},
                top_k=1,
                include_values=True
            )

            # 유사도 계산
            if query_response.matches:
                similarity = query_response.matches[0].score
            else:
                # Pinecone에 대기 유저의 임베딩이 없는 경우
                waiting_user_data = get_user_data(waiting_member_id, mysql_connection)
                waiting_embedding_text = prepare_text_for_embedding(waiting_user_data)
                waiting_user_embedding = model.encode([waiting_embedding_text])[0]

                # Pinecone에 대기 유저 임베딩 저장
                pinecone_index.upsert([(str(waiting_member_id), waiting_user_embedding.tolist(),
                                        {"updated_at": datetime.now().isoformat()})])

                # NumPy로 코사인 유사도 계산
                similarity = np.dot(new_user_embedding, waiting_user_embedding) / (
                        np.linalg.norm(new_user_embedding) * np.linalg.norm(waiting_user_embedding))

            # 매칭 정보 저장
            match_query = """
            INSERT INTO member_matches 
            (member_id, matched_member_id, similarity_score, status) 
            VALUES (%s, %s, %s, 'active'), (%s, %s, %s, 'active')
            """
            cursor.execute(match_query,
                           (new_user_id, waiting_member_id, float(similarity),
                            waiting_member_id, new_user_id, float(similarity)))

            # 대기열에서 매칭된 유저 제거
            cursor.execute("UPDATE waiting_queue SET status = 'matched' WHERE member_id = %s", (waiting_member_id,))
            mysql_connection.commit()

            logger.info(f"새 유저 {new_user_id}가 대기 유저 {waiting_member_id}와 매칭되었습니다. 유사도: {similarity}")

        cursor.close()
        logger.info(f"새 유저 처리 완료: {new_user_id}, {datetime.now()}")
    except Exception as e:
        logger.error(f"새 유저 처리 오류: {str(e)}")
        raise
    finally:
        if close_connection and mysql_connection:
            mysql_connection.close()


def setup_scheduler():
    """일주일에 한 번 매칭을 실행하는 스케줄러 설정"""
    logger.info("스케줄러 설정 시작")

    schedule.every().monday.at("02:00").do(weekly_matching_process)
    logger.info("매주 월요일 02:00에 매칭 실행 예약됨")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    logger.info("매처 모듈 테스트 시작")
    try:
        pinecone_idx = init_pinecone()
        mysql_conn = get_mysql_connection()
        weekly_matching_process(mysql_conn, pinecone_idx)
        mysql_conn.close()
    except Exception as e:
        logger.error(f"테스트 실행 오류: {str(e)}")