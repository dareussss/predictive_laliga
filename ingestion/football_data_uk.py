"""Ingestion pentru CSV-urile istorice de la football-data.co.uk.

    python -m ingestion.football_data_uk
    python -m ingestion.football_data_uk --seasons 2023 2024 2025
    python -m ingestion.football_data_uk --force
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import pandas as pd
import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from db.models import League, Match, Team
from db.session import get_session, init_db

log = logging.getLogger(__name__)

COLUMN_MAP: dict[str, str] = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    "AvgH": "avg_home",
    "AvgD": "avg_draw",
    "AvgA": "avg_away",
    "AvgCH": "avg_close_home",
    "AvgCD": "avg_close_draw",
    "AvgCA": "avg_close_away",
}

INT_FIELDS = frozenset(
    {
        "home_goals",
        "away_goals",
        "ht_home_goals",
        "ht_away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_corners",
        "away_corners",
        "home_fouls",
        "away_fouls",
        "home_yellow",
        "away_yellow",
        "home_red",
        "away_red",
    }
)


class Season(NamedTuple):
    """Un sezon identificat prin anul de start: Season(2025) este 2025-26."""

    start_year: int

    @property
    def code(self) -> str:
        """Codul din URL-ul football-data.co.uk: 2526."""
        return f"{self.start_year % 100:02d}{(self.start_year + 1) % 100:02d}"

    @property
    def label(self) -> str:
        """Eticheta stocata in baza de date: 2025-26."""
        return f"{self.start_year}-{(self.start_year + 1) % 100:02d}"


class LoadReport(NamedTuple):
    seasons: int
    inserted: int
    updated: int


def download(season: Season, force: bool = False) -> Path:
    """Descarca CSV-ul sezonului in data/raw/, reutilizand cache-ul local."""
    destination = config.RAW_DIR / f"{config.LEAGUE_CODE}_{season.code}.csv"

    if destination.exists() and not force:
        log.info("cache hit: %s", destination.name)
        return destination

    url = f"{config.FOOTBALL_DATA_UK_BASE}/{season.code}/{config.LEAGUE_CODE}.csv"
    log.info("descarc %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def parse(path: Path, season: Season) -> pd.DataFrame:
    """Citeste CSV-ul si il normalizeaza la schema interna."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    raw = raw[raw["HomeTeam"].astype(str).str.strip().ne("")]

    parsed = pd.DataFrame(index=raw.index)
    parsed["season"] = season.label
    parsed["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed").dt.date
    parsed["home_team"] = raw["HomeTeam"].astype(str).str.strip()
    parsed["away_team"] = raw["AwayTeam"].astype(str).str.strip()

    for source, field in COLUMN_MAP.items():
        if source not in raw.columns:
            parsed[field] = None
        elif field == "result":
            parsed[field] = raw[source].astype(str).str.strip()
        else:
            parsed[field] = pd.to_numeric(raw[source], errors="coerce")

    return parsed.dropna(subset=["home_goals", "away_goals"])


def _match_values(row: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in COLUMN_MAP.values():
        value = getattr(row, field)
        if pd.isna(value):
            values[field] = None
        elif field in INT_FIELDS:
            values[field] = int(value)
        else:
            values[field] = value
    return values


def _get_league(session: Session) -> League:
    league = session.scalar(select(League).where(League.code == config.LEAGUE_CODE))
    if league is None:
        league = League(
            code=config.LEAGUE_CODE,
            name=config.LEAGUE_NAME,
            country=config.LEAGUE_COUNTRY,
        )
        session.add(league)
        session.flush()
    return league


def _resolve_team(session: Session, index: dict[str, Team], name: str) -> Team:
    team = index.get(name)
    if team is None:
        team = Team(name=name)
        session.add(team)
        session.flush()
        index[name] = team
    return team


def _existing_matches(session: Session, season: Season) -> dict[tuple[int, int], Match]:
    matches = session.scalars(select(Match).where(Match.season == season.label))
    return {(match.home_team_id, match.away_team_id): match for match in matches}


def load(seasons: Iterable[Season], force: bool = False) -> LoadReport:
    """Descarca, parseaza si incarca sezoanele date."""
    init_db()
    inserted = updated = processed = 0

    with get_session() as session:
        league = _get_league(session)
        teams = {team.name: team for team in session.scalars(select(Team))}

        for season in seasons:
            frame = parse(download(season, force=force), season)
            log.info("%s: %d meciuri", season.label, len(frame))
            processed += 1

            existing = _existing_matches(session, season)

            for row in frame.itertuples(index=False):
                home = _resolve_team(session, teams, row.home_team)
                away = _resolve_team(session, teams, row.away_team)
                values = _match_values(row)

                match = existing.get((home.id, away.id))
                if match is None:
                    session.add(
                        Match(
                            league_id=league.id,
                            season=season.label,
                            date=row.date,
                            home_team_id=home.id,
                            away_team_id=away.id,
                            **values,
                        )
                    )
                    inserted += 1
                else:
                    match.date = row.date
                    for field, value in values.items():
                        setattr(match, field, value)
                    updated += 1

    return LoadReport(seasons=processed, inserted=inserted, updated=updated)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Incarca sezoane La Liga in baza de date.")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(config.FIRST_SEASON, config.LAST_SEASON + 1)),
        help="Anii de start ai sezoanelor (2023 inseamna sezonul 2023-24).",
    )
    parser.add_argument("--force", action="store_true", help="Re-descarca, ignora cache-ul.")
    args = parser.parse_args()

    report = load([Season(year) for year in args.seasons], force=args.force)
    print(
        f"\nGata: {report.seasons} sezoane | "
        f"{report.inserted} meciuri inserate | {report.updated} actualizate"
    )


if __name__ == "__main__":
    main()
