"""Incarcarea si memorarea datelor folosite de dashboard.

Toate operatiile costisitoare -- antrenarea modelului, simularea sezonului, estimarea
curbei de varsta -- trec pe aici, ca sa fie calculate o singura data pe sesiune.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.theme import Palette, palette_for
from db.queries import load_fixtures, load_matches, load_player_seasons
from models.form_forecast import build_model, forecast
from models.player_forecast import estimate_age_curve, validate
from models.player_index import aggregate_seasons, build_index
from models.scoring_race import (
    build_weights,
    calibrate_appearance_shrinkage,
    load_scorers,
    simulate,
    summarise,
)


def active_palette() -> Palette:
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = "light"
    return palette_for(base)


@st.cache_data(show_spinner=False)
def cached_matches() -> pd.DataFrame:
    return load_matches()


@st.cache_data(show_spinner=False)
def cached_fixtures() -> pd.DataFrame:
    return load_fixtures()


@st.cache_data(show_spinner=False)
def cached_players() -> pd.DataFrame:
    return load_player_seasons()


@st.cache_resource(show_spinner="Antrenez modelul Dixon-Coles...")
def cached_model(xi: float):
    return build_model(cached_matches(), cached_fixtures(), xi=xi)


@st.cache_data(show_spinner="Calculez forma...")
def cached_form(xi: float, horizon: int) -> pd.DataFrame:
    model, promoted = cached_model(xi)
    fixtures = cached_fixtures()
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))

    rows = []
    for team in teams:
        form = forecast(model, fixtures, team, horizon)
        rows.append(
            {
                "echipa": team,
                "puncte": form.expected_points,
                "min7": form.probability_of_at_least(7),
                "maxim": float(form.points_distribution[-1]),
                "acasa": sum(1 for fixture in form.fixtures if fixture.at_home),
                "estimat": team in promoted,
            }
        )
    return pd.DataFrame(rows).sort_values("puncte", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_team_fixtures(xi: float, horizon: int, team: str) -> pd.DataFrame:
    model, _ = cached_model(xi)
    form = forecast(model, cached_fixtures(), team, horizon)
    return pd.DataFrame(
        [
            {
                "etapa": fixture.matchday,
                "data": fixture.date,
                "acasa": fixture.at_home,
                "adversar": fixture.opponent,
                "victorie": fixture.win,
                "egal": fixture.draw,
                "infrangere": fixture.loss,
                "goluri_marcate": fixture.goals_for,
                "goluri_primite": fixture.goals_against,
                "puncte": fixture.expected_points,
            }
            for fixture in form.fixtures
        ]
    )


@st.cache_data(show_spinner="Simulez sezonul...")
def cached_race(xi: float, n_sims: int) -> pd.DataFrame:
    model, _ = cached_model(xi)
    fixtures = cached_fixtures()
    scorers = load_scorers()
    shrinkage = calibrate_appearance_shrinkage(cached_players())
    teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
    weights = build_weights(scorers, shrinkage, teams)
    totals = simulate(model, fixtures, weights, n_sims=n_sims)
    return pd.DataFrame([result.__dict__ for result in summarise(totals, weights)])


@st.cache_data(show_spinner="Calculez indexul...")
def cached_index(season: str, min_minutes: int) -> pd.DataFrame:
    index, _ = build_index(cached_players(), season=season, min_minutes=min_minutes)
    return index


@st.cache_data(show_spinner="Estimez curba de varsta...")
def cached_age_curve(metric: str):
    seasons = aggregate_seasons(cached_players(), metric=metric)
    curve = estimate_age_curve(seasons)
    return curve.curve, curve.peak_age, curve.n_transitions, validate(seasons)


def table_view(frame: pd.DataFrame, label: str = "Vezi datele ca tabel") -> None:
    with st.expander(label):
        st.dataframe(frame, width="stretch", hide_index=True)
