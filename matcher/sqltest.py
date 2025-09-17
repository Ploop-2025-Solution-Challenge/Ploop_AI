import mysql.connector
import os
from dotenv import load_dotenv
import pandas as pd

# .env 로드
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=3306,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )

def test_fetch_user_data():
    conn = get_mysql_connection()
    try:
        # user 全부 + preference (없으면 NULL) 포함
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
            print("⚠️ No data in tables.")
            return

        # 핵심: NA 그룹을 버리지 않도록 dropna=False
        # user_id 기준으로 모으고, user의 단일 속성은 first로 집계
        out = (
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

        # preference가 전부 NULL이었던 유저는 위 lambda로 []가 됨
        print("\n===== User Data with Preferences (NULL 허용) =====")
        print(out.to_string(index=False))

    finally:
        conn.close()

if __name__ == "__main__":
    test_fetch_user_data()
