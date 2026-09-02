"""Ingestion pentru calendarul viitor de la football-data.org.

    python -m ingestion.football_data_org
    python -m ingestion.football_data_org --season 2026

Sursa istorica are doar meciuri jucate; calendarul viitor vine de aici.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, NamedTuple

import pandas as pd
import requests
from sqlalchemy import select

import config
from db.models import League, Match, ScorerSeason, Team
from db.session import get_session, init_db
from ingestion.football_data_uk import Season
from ingestion.team_names import to_canonical

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30
SCORER_LIMIT = 500


class FixtureReport(NamedTuple):
    fetched: int
    inserted: int
    updated: int


def fetch_fixtures(season_start_year: int) -> list[dict[str, Any]]:
    """Descarca toate meciurile unui sezon de la football-data.org."""
    if not config.FOOTBALL_DATA_ORG_TOKEN:
        raise RuntimeError("FOOTBALL_DATA_ORG_TOKEN lipseste din .env")

    url = (
        f"{config.FOOTBALL_DATA_ORG_BASE}"
        f"/competitions/{config.FOOTBALL_DATA_ORG_COMPETITION}/matches"
    )
    log.info("descarc calendarul sezonului %s", season_start_year)
    response = requests.get(
        url,
        headers={"X-Auth-Token": config.FOOTBALL_DATA_ORG_TOKEN},
        params={"season": season_start_year},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("matches", [])


def to_rows(fixtures: list[dict[str, Any]], season: Season) -> pd.DataFrame:
    """Normalizeaza raspunsul API la schema interna."""
    rows = []
    for fixture in fixtures:
        score = fixture.get("score", {}).get("fullTime", {})
        rows.append(
            {
                "season": season.label,
                "date": pd.to_datetime(fixture["utcDate"]).date(),
                "matchday": fixture.get("matchday"),
                "status": fixture.get("status"),
                "home_team": to_canonical(fixture["homeTeam"]["name"]),
                "away_team": to_canonical(fixture["awayTeam"]["name"]),
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
            }
        )
    return pd.DataFrame(rows)


def _result_from(home_goals: int | None, away_goals: int | None) -> str | None:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def load(season_start_year: int) -> FixtureReport:
    """Descarca si incarca calendarul, actualizand meciurile deja cunoscute."""
    init_db()
    season = Season(season_start_year)
    fixtures = fetch_fixtures(season_start_year)
    frame = to_rows(fixtures, season)

    inserted = updated = 0

    with get_session() as session:
        league = session.scalar(select(League).where(League.code == config.LEAGUE_CODE))
        if league is None:
            league = League(
                code=config.LEAGUE_CODE,
                name=config.LEAGUE_NAME,
                country=config.LEAGUE_COUNTRY,
            )
            session.add(league)
            session.flush()

        teams = {team.name: team for team in session.scalars(select(Team))}
        existing = {
            (match.home_team_id, match.away_team_id): match
            for match in session.scalars(select(Match).where(Match.season == season.label))
        }

        for row in frame.itertuples(index=False):
            for name in (row.home_team, row.away_team):
                if name not in teams:
                    team = Team(name=name)
                    session.add(team)
                    session.flush()
                    teams[name] = team

            home, away = teams[row.home_team], teams[row.away_team]
            values = {
                "date": row.date,
                "matchday": row.matchday,
                "status": row.status,
                "home_goals": row.home_goals,
                "away_goals": row.away_goals,
                "result": _result_from(row.home_goals, row.away_goals),
            }

            match = existing.get((home.id, away.id))
            if match is None:
                session.add(
                    Match(
                        league_id=league.id,
                        season=season.label,
                        home_team_id=home.id,
                        away_team_id=away.id,
                        **values,
                    )
                )
                inserted += 1
            else:
                for field, value in values.items():
                    setattr(match, field, value)
                updated += 1

    return FixtureReport(fetched=len(frame), inserted=inserted, updated=updated)


def fetch_scorers(season_start_year: int) -> list[dict[str, Any]]:
    """Descarca marcatorii unui sezon."""
    if not config.FOOTBALL_DATA_ORG_TOKEN:
        raise RuntimeError("FOOTBALL_DATA_ORG_TOKEN lipseste din .env")

    url = (
        f"{config.FOOTBALL_DATA_ORG_BASE}"
        f"/competitions/{config.FOOTBALL_DATA_ORG_COMPETITION}/scorers"
    )
    log.info("descarc marcatorii sezonului %s", season_start_year)
    response = requests.get(
        url,
        headers={"X-Auth-Token": config.FOOTBALL_DATA_ORG_TOKEN},
        params={"season": season_start_year, "limit": SCORER_LIMIT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("scorers", [])


def load_scorers(season_start_year: int) -> FixtureReport:
    """Incarca marcatorii sezonului, idempotent pe (sezon, jucator, echipa)."""
    init_db()
    season = Season(season_start_year)
    scorers = fetch_scorers(season_start_year)

    inserted = updated = 0

    with get_session() as session:
        existing = {
            (row.player_name, row.team): row
            for row in session.scalars(
                select(ScorerSeason).where(ScorerSeason.season == season.label)
            )
        }

        for entry in scorers:
            player = entry["player"]
            team = to_canonical(entry["team"]["name"])
            name = player["name"]

            values = {
                "date_of_birth": player.get("dateOfBirth"),
                "nationality": player.get("nationality"),
                "section": player.get("section"),
                "played_matches": entry.get("playedMatches"),
                "goals": entry.get("goals"),
                "assists": entry.get("assists"),
                "penalties": entry.get("penalties"),
            }

            row = existing.get((name, team))
            if row is None:
                session.add(
                    ScorerSeason(season=season.label, player_name=name, team=team, **values)
                )
                inserted += 1
            else:
                for field, value in values.items():
                    setattr(row, field, value)
                updated += 1

    return FixtureReport(fetched=len(scorers), inserted=inserted, updated=updated)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Incarca date La Liga de la football-data.org.")
    parser.add_argument("--season", type=int, default=config.LAST_SEASON + 1)
    parser.add_argument(
        "--scorers",
        type=int,
        metavar="AN",
        help="Incarca marcatorii sezonului dat in loc de calendar (ex: 2025).",
    )
    args = parser.parse_args()

    if args.scorers is not None:
        report = load_scorers(args.scorers)
        print(
            f"\nGata: {report.fetched} marcatori primiti | "
            f"{report.inserted} inserati | {report.updated} actualizati"
        )
        return

    report = load(args.season)
    print(
        f"\nGata: {report.fetched} meciuri primite | "
        f"{report.inserted} inserate | {report.updated} actualizate"
    )


if __name__ == "__main__":
    main()
