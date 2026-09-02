"""Interogari refolosite in tot proiectul."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from db.models import Match, Player, PlayerSeasonStats, Team
from db.session import engine


def _base_select():
    home = Team.__table__.alias("home")
    away = Team.__table__.alias("away")
    statement = (
        select(
            Match.season,
            Match.date,
            Match.matchday,
            home.c.name.label("home_team"),
            away.c.name.label("away_team"),
            Match.home_goals,
            Match.away_goals,
            Match.result,
            Match.home_shots,
            Match.away_shots,
            Match.avg_close_home,
            Match.avg_close_draw,
            Match.avg_close_away,
        )
        .join(home, Match.home_team_id == home.c.id)
        .join(away, Match.away_team_id == away.c.id)
        .order_by(Match.date)
    )
    return statement


def _read(statement) -> pd.DataFrame:
    with engine.connect() as connection:
        frame = pd.read_sql(statement, connection)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_matches() -> pd.DataFrame:
    """Meciurile jucate, cu numele echipelor rezolvate, ordonate cronologic."""
    return _read(_base_select().where(Match.home_goals.is_not(None)))


def load_fixtures(season: str | None = None) -> pd.DataFrame:
    """Meciurile programate dar nejucate inca."""
    statement = _base_select().where(Match.home_goals.is_(None))
    if season is not None:
        statement = statement.where(Match.season == season)
    return _read(statement)


def load_player_seasons(season: str | None = None) -> pd.DataFrame:
    """Statisticile de jucator pe sezon, cu identitatea jucatorului rezolvata."""
    statement = (
        select(
            PlayerSeasonStats.player_id,
            PlayerSeasonStats.season,
            PlayerSeasonStats.squad,
            Player.name,
            Player.born,
            Player.nationality,
            PlayerSeasonStats.position,
            PlayerSeasonStats.age,
            PlayerSeasonStats.matches,
            PlayerSeasonStats.starts,
            PlayerSeasonStats.minutes,
            PlayerSeasonStats.goals,
            PlayerSeasonStats.assists,
            PlayerSeasonStats.penalties,
            PlayerSeasonStats.penalties_attempted,
            PlayerSeasonStats.xg,
            PlayerSeasonStats.npxg,
            PlayerSeasonStats.xag,
            PlayerSeasonStats.progressive_carries,
            PlayerSeasonStats.progressive_passes,
        )
        .join(Player, PlayerSeasonStats.player_id == Player.id)
        .order_by(PlayerSeasonStats.season, Player.name)
    )
    if season is not None:
        statement = statement.where(PlayerSeasonStats.season == season)

    with engine.connect() as connection:
        return pd.read_sql(statement, connection)
