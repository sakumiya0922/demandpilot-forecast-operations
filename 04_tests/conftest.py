"""実DBテスト用の隔離スキーマ。業務テーブルは変更しない。"""
import os
from pathlib import Path
import uuid
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

@pytest.fixture
def database():
    # 本番テーブルは使わず、各テスト専用の空スキーマを作って終了時に削除する。
    base = os.environ["DATABASE_URL"]
    schema = "test_" + uuid.uuid4().hex
    with psycopg.connect(base, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    dsn = make_conninfo(base, options=f"-c search_path={schema}")
    try:
        with psycopg.connect(dsn) as connection:
            connection.execute((Path(__file__).parents[1] / "01_data/preparation/schema.sql").read_text())
        yield dsn
    finally:
        with psycopg.connect(base, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))

