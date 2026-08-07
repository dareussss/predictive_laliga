"""Interogari refolosite in tot proiectul."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from db.models import Match, Team
from db.session import engine


def load_matches() -> pd.DataFrame:
    """Toate meciurile jucate, cu numele echipelor rezolvate, ordonate cronologic."""
    home = Team.__table__.alias("home")
    away = Team.__table__.alias("away")

    statement = (
        select(
            Match.season,
            Match.date,
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

    with engine.connect() as connection:
        matches = pd.read_sql(statement, connection)

    matches["date"] = pd.to_datetime(matches["date"])
    return matches
