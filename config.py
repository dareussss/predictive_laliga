"""Configurare centralizata a proiectului."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for _directory in (DATA_DIR, RAW_DIR, PROCESSED_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def _resolve_db_url(url: str) -> str:
    """Rezolva caile SQLite relative fata de radacina proiectului."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        relative = url.removeprefix("sqlite:///")
        return f"sqlite:///{(ROOT / relative).as_posix()}"
    return url


DB_URL = _resolve_db_url(os.getenv("DB_URL", "sqlite:///data/football.db"))

FOOTBALL_DATA_UK_BASE = "https://www.football-data.co.uk/mmz4281"

FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"
FOOTBALL_DATA_ORG_TOKEN = os.getenv("FOOTBALL_DATA_ORG_TOKEN", "")
FOOTBALL_DATA_ORG_COMPETITION = "PD"

LEAGUE_CODE = "SP1"
LEAGUE_NAME = "La Liga"
LEAGUE_COUNTRY = "Spain"

FIRST_SEASON = 2005
LAST_SEASON = 2025
