import os
from collections.abc import Generator
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def _build_database_url() -> URL:
    raw_url = os.getenv("DB_URL")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")

    if not raw_url or not username or not password:
        raise RuntimeError(
            "DB_URL, DB_USERNAME, and DB_PASSWORD must be configured."
        )

    # The shared environment uses Spring's JDBC URL format. SQLAlchemy needs
    # the same address without the leading "jdbc:" prefix.
    parsed = urlsplit(raw_url.removeprefix("jdbc:"))
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise RuntimeError("DB_URL must be a PostgreSQL JDBC or PostgreSQL URL.")
    if not parsed.hostname or not parsed.path.lstrip("/"):
        raise RuntimeError("DB_URL must include a host and database name.")

    return URL.create(
        drivername="postgresql+psycopg",
        username=username,
        password=password,
        host=parsed.hostname,
        port=parsed.port,
        database=parsed.path.lstrip("/"),
        query=dict(parse_qsl(parsed.query)),
    )


engine: Engine = create_engine(
    _build_database_url(),
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session


def check_database_connection() -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
