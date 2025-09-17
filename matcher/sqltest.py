import mysql.connector
import os
from dotenv import load_dotenv
import pandas as pd

# .env 파일 로드
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

def test_fetch_tables():
    conn = get_mysql_connection()
    try:
        tables = [
            "members",
            "member_location_preference",
            "member_motivation",
            "member_matches",
            "waiting_queue"
        ]

        for table in tables:
            print(f"\n===== {table} =====")
            try:
                df = pd.read_sql(f"SELECT * FROM {table} LIMIT 10", conn)
                if df.empty:
                    print("(No data)")
                else:
                    print(df)
            except Exception as e:
                print(f"Error reading table {table}: {e}")

    finally:
        conn.close()

if __name__ == "__main__":
    test_fetch_tables()
