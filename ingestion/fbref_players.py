"""Ingestion pentru statisticile de jucator FBref.

    python -m ingestion.fbref_players
    python -m ingestion.fbref_players --force

Datele provin din FBref, prin arhiva publica worldfootballR_data colectata de
Jason Zivkovic (github.com/JaseZiv/worldfootballR_data). Arhiva e inghetata, deci
ultimul sezon complet disponibil este 2024-25.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyreadr
import requests
from sqlalchemy import select

import config
from db.models import Player, PlayerSeasonStats
from db.session import get_session, init_db

log = logging.getLogger(__name__)

RELEASE_BASE = (
    "https://github.com/JaseZiv/worldfootballR_data/releases/download"
    "/fb_big5_advanced_season_stats"
)
STANDARD_FILE = "big5_player_standard.rds"
COMPETITION = "La Liga"

COLUMN_MAP: dict[str, str] = {
    "Pos": "position",
    "MP_Playing": "matches",
    "Starts_Playing": "starts",
    "Min_Playing": "minutes",
    "Gls": "goals",
    "Ast": "assists",
    "PK": "penalties",
    "PKatt": "penalties_attempted",
    "CrdY": "yellow",
    "CrdR": "red",
    "xG_Expected": "xg",
    "npxG_Expected": "npxg",
    "xAG_Expected": "xag",
    "PrgC_Progression": "progressive_carries",
    "PrgP_Progression": "progressive_passes",
}

INT_FIELDS = frozenset(
    {
        "matches",
        "starts",
        "minutes",
        "goals",
        "assists",
        "penalties",
        "penalties_attempted",
        "yellow",
        "red",
        "progressive_carries",
        "progressive_passes",
    }
)


def download(force: bool = False) -> Path:
    """Descarca arhiva de statistici, reutilizand cache-ul local."""
    destination = config.RAW_DIR / STANDARD_FILE

    if destination.exists() and not force:
        log.info("cache hit: %s", destination.name)
        return destination

    url = f"{RELEASE_BASE}/{STANDARD_FILE}"
    log.info("descarc %s", url)
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def season_label(season_end_year: int) -> str:
    """2025 (anul de final) -> '2024-25' (formatul nostru)."""
    return f"{season_end_year - 1}-{season_end_year % 100:02d}"


def derive_age(season_end_year: pd.Series, born: pd.Series) -> pd.Series:
    """Varsta aproximativa la inceputul sezonului, derivata din anul nasterii.

    Coloana Age din sursa are formate diferite intre sezoane; Born e consistent in
    toate cele 17. Incertitudinea ramasa e de pana la un an.
    """
    return season_end_year - born - 1


def parse(path: Path) -> pd.DataFrame:
    """Citeste fisierul RDS si pastreaza doar La Liga, normalizat."""
    raw = next(iter(pyreadr.read_r(str(path)).values()))
    raw = raw[raw["Comp"] == COMPETITION].copy()

    season_end_year = raw["Season_End_Year"].astype(int)

    parsed = pd.DataFrame(index=raw.index)
    parsed["season"] = season_end_year.map(season_label)
    parsed["squad"] = raw["Squad"].astype(str).str.strip()
    parsed["name"] = raw["Player"].astype(str).str.strip()
    parsed["nationality"] = raw["Nation"].astype(str).str.strip().str[:8]
    parsed["born"] = pd.to_numeric(raw["Born"], errors="coerce")
    parsed["fbref_url"] = raw["Url"].astype(str).str.strip() if "Url" in raw else None
    parsed["age"] = derive_age(season_end_year, parsed["born"])

    for source, field in COLUMN_MAP.items():
        if source == "Pos":
            parsed[field] = raw[source].astype(str).str.strip()
        elif source in raw.columns:
            parsed[field] = pd.to_numeric(raw[source], errors="coerce")
        else:
            parsed[field] = None

    return parsed.dropna(subset=["name", "squad", "season"])


def _values(row: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for field in COLUMN_MAP.values():
        value = getattr(row, field)
        if pd.isna(value):
            values[field] = None
        elif field in INT_FIELDS:
            values[field] = int(value)
        else:
            values[field] = value
    return values


def _player_key(row: object) -> str:
    """Identitatea unui jucator: URL-ul FBref daca exista, altfel nume plus an."""
    url = getattr(row, "fbref_url", None)
    if isinstance(url, str) and url.startswith("http"):
        return url
    born = getattr(row, "born", None)
    return f"{row.name_}|{int(born) if pd.notna(born) else '?'}"


def load(force: bool = False) -> dict[str, int]:
    """Incarca statisticile de jucator in baza de date, idempotent."""
    init_db()
    frame = parse(download(force=force))
    frame = frame.rename(columns={"name": "name_"})

    inserted = updated = 0

    with get_session() as session:
        by_url = {
            player.fbref_url: player
            for player in session.scalars(select(Player))
            if player.fbref_url
        }
        by_fallback = {
            f"{player.name}|{player.born if player.born else '?'}": player
            for player in session.scalars(select(Player))
        }
        existing = {
            (stats.player_id, stats.season, stats.squad): stats
            for stats in session.scalars(select(PlayerSeasonStats))
        }

        for row in frame.itertuples(index=False):
            key = _player_key(row)
            player = by_url.get(key) or by_fallback.get(key)

            if player is None:
                player = Player(
                    fbref_url=key if key.startswith("http") else None,
                    name=row.name_,
                    born=int(row.born) if pd.notna(row.born) else None,
                    nationality=row.nationality or None,
                )
                session.add(player)
                session.flush()
                if player.fbref_url:
                    by_url[player.fbref_url] = player
                by_fallback[f"{player.name}|{player.born if player.born else '?'}"] = player

            values = _values(row)
            values["age"] = row.age if pd.notna(row.age) else None

            stats = existing.get((player.id, row.season, row.squad))
            if stats is None:
                session.add(
                    PlayerSeasonStats(
                        player_id=player.id,
                        season=row.season,
                        squad=row.squad,
                        **values,
                    )
                )
                inserted += 1
            else:
                for field, value in values.items():
                    setattr(stats, field, value)
                updated += 1

    return {"inserted": inserted, "updated": updated, "rows": len(frame)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Incarca statistici FBref de jucator.")
    parser.add_argument("--force", action="store_true", help="Re-descarca, ignora cache-ul.")
    args = parser.parse_args()

    report = load(force=args.force)
    print(
        f"\nGata: {report['rows']} randuri La Liga | "
        f"{report['inserted']} inserate | {report['updated']} actualizate"
    )


if __name__ == "__main__":
    main()
