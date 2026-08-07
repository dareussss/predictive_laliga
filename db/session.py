"""Engine si sesiuni SQLAlchemy."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import config
from db.models import Base

engine: Engine = create_engine(config.DB_URL, future=True)

if config.DB_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, future=True, expire_on_commit=False)


def init_db() -> None:
    """Creeaza tabelele lipsa."""
    Base.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
    """Sesiune tranzactionala: commit la iesire, rollback la eroare."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
