"""Cursa pentru titlul de golgheter, prin simulare Monte Carlo.

    python -m models.scoring_race
    python -m models.scoring_race --sims 20000 --limit 25

Loturile folosite sunt cele de la finalul sezonului 2025-26, iar echipele promovate
n-au jucatori in datele de La Liga. Ambele limitari sunt detaliate in README.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select

from db.models import ScorerSeason
from db.session import engine
from models.dixon_coles import DEFAULT_XI, DixonColes
from models.form_forecast import build_model
from models.player_index import build_transitions, fit_shrinkage, primary_position

DEFAULT_SIMS = 10_000
DEFAULT_BASE_SEASON = "2025-26"


@dataclass(frozen=True)
class RaceResult:
    """Rezultatul simularii pentru un jucator."""

    player: str
    team: str
    base_goals: int
    base_matches: int
    rate: float
    expected_goals: float
    title_probability: float
    p20_plus: float
    p25_plus: float


def load_scorers(season: str = DEFAULT_BASE_SEASON) -> pd.DataFrame:
    """Marcatorii sezonului de referinta."""
    statement = select(
        ScorerSeason.player_name,
        ScorerSeason.team,
        ScorerSeason.played_matches,
        ScorerSeason.goals,
        ScorerSeason.section,
    ).where(ScorerSeason.season == season)

    with engine.connect() as connection:
        frame = pd.read_sql(statement, connection)

    frame["goals"] = frame["goals"].fillna(0).astype(int)
    frame["played_matches"] = frame["played_matches"].fillna(0).astype(int)
    return frame[frame["played_matches"] > 0]


def calibrate_appearance_shrinkage(players: pd.DataFrame) -> float:
    """Cat trebuie contractata rata de goluri pe aparitie, calibrat pe istoricul FBref.

    Pe scara aparitiilor in loc de minute, pentru ca sursa recenta nu are minute.
    """
    frame = players[(players["matches"] > 0) & (players["minutes"] > 0)].copy()
    frame["group"] = frame["position"].map(primary_position)

    seasons = (
        frame.groupby(["player_id", "season"])
        .agg(
            group=("group", "first"),
            minutes=("minutes", "sum"),
            exposure=("matches", "sum"),
            value=("goals", "sum"),
        )
        .reset_index()
    )
    seasons["rate"] = seasons["value"] / seasons["exposure"]

    transitions = build_transitions(seasons)
    mean = seasons["value"].sum() / seasons["exposure"].sum()
    return fit_shrinkage(transitions, mean)


def build_weights(
    scorers: pd.DataFrame,
    shrinkage: float,
    teams_in_league: set[str],
) -> pd.DataFrame:
    """Golurile asteptate de fiecare jucator pe un sezon intreg, si ponderea in echipa."""
    frame = scorers[scorers["team"].isin(teams_in_league)].copy()
    mean_rate = frame["goals"].sum() / frame["played_matches"].sum()

    frame["rate"] = (frame["goals"] + mean_rate * shrinkage) / (
        frame["played_matches"] + shrinkage
    )
    frame["expected_goals"] = frame["rate"] * frame["played_matches"]

    totals = frame.groupby("team")["expected_goals"].transform("sum")
    frame["share"] = frame["expected_goals"] / totals
    return frame.reset_index(drop=True)


def simulate_team_goals(
    model: DixonColes,
    fixtures: pd.DataFrame,
    n_sims: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    """Golurile marcate de fiecare echipa pe tot sezonul, in fiecare simulare."""
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    index = {team: position for position, team in enumerate(teams)}
    totals = np.zeros((n_sims, len(teams)), dtype=np.int32)

    for fixture in fixtures.itertuples(index=False):
        if not (model.known(fixture.home_team) and model.known(fixture.away_team)):
            continue

        matrix = model.score_matrix(fixture.home_team, fixture.away_team)
        flat = matrix.ravel()
        drawn = rng.choice(len(flat), size=n_sims, p=flat / flat.sum())
        home_goals, away_goals = np.unravel_index(drawn, matrix.shape)

        totals[:, index[fixture.home_team]] += home_goals.astype(np.int32)
        totals[:, index[fixture.away_team]] += away_goals.astype(np.int32)

    return totals, index


def simulate(
    model: DixonColes,
    fixtures: pd.DataFrame,
    weights: pd.DataFrame,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 7,
) -> np.ndarray:
    """Simuleaza sezonul de n_sims ori si intoarce golurile fiecarui jucator.

    Incertitudinea are trei surse: golurile echipei (matricea Dixon-Coles), partea
    care ii revine jucatorului (extrasa Dirichlet, nu fixata la estimare) si varianta
    proprie (Poisson). Returneaza o matrice (n_sims, n_jucatori).
    """
    rng = np.random.default_rng(seed)
    players = weights.reset_index(drop=True)
    team_totals, team_index = simulate_team_goals(model, fixtures, n_sims, rng)

    goals = np.zeros((n_sims, len(players)), dtype=np.int32)

    for team, block in players.groupby("team"):
        if team not in team_index:
            continue
        columns = block.index.to_numpy()
        concentration = block["expected_goals"].to_numpy()
        if concentration.sum() <= 0:
            continue

        shares = rng.dirichlet(np.maximum(concentration, 1e-3), size=n_sims)
        expected = team_totals[:, team_index[team]][:, None] * shares
        goals[:, columns] = rng.poisson(expected).astype(np.int32)

    return goals


def summarise(totals: np.ndarray, weights: pd.DataFrame) -> list[RaceResult]:
    """Transforma simularile in probabilitati per jucator."""
    best = totals.max(axis=1, keepdims=True)
    winners = (totals == best) & (totals > 0)
    title_probability = (winners / np.maximum(winners.sum(axis=1, keepdims=True), 1)).mean(axis=0)

    expected = totals.mean(axis=0)
    p20 = (totals >= 20).mean(axis=0)
    p25 = (totals >= 25).mean(axis=0)

    results = [
        RaceResult(
            player=row.player_name,
            team=row.team,
            base_goals=int(row.goals),
            base_matches=int(row.played_matches),
            rate=float(row.rate),
            expected_goals=float(expected[position]),
            title_probability=float(title_probability[position]),
            p20_plus=float(p20[position]),
            p25_plus=float(p25[position]),
        )
        for position, row in enumerate(weights.itertuples(index=False))
    ]
    return sorted(results, key=lambda result: result.title_probability, reverse=True)


def main() -> None:
    from db.queries import load_fixtures, load_matches, load_player_seasons

    parser = argparse.ArgumentParser(description="Cursa pentru golgheter, prin simulare.")
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--season", default=None, help="Sezonul de calendar, ex. 2026-27.")
    parser.add_argument("--base-season", default=DEFAULT_BASE_SEASON)
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    matches = load_matches()
    fixtures = load_fixtures(args.season)
    if fixtures.empty:
        raise SystemExit("Niciun meci programat. Ruleaza ingestion.football_data_org.")

    model, promoted = build_model(matches, fixtures, xi=args.xi)

    scorers = load_scorers(args.base_season)
    if scorers.empty:
        raise SystemExit(
            f"Niciun marcator pentru {args.base_season}. "
            f"Ruleaza ingestion.football_data_org --scorers 2025."
        )

    shrinkage = calibrate_appearance_shrinkage(load_player_seasons())
    teams_in_league = set(fixtures["home_team"]) | set(fixtures["away_team"])
    weights = build_weights(scorers, shrinkage, teams_in_league)

    print(f"Calendar        : {len(fixtures)} meciuri")
    print(f"Marcatori {args.base_season} : {len(scorers)}  ->  {len(weights)} in ligile de anul viitor")
    print(f"Contractie      : {shrinkage:.1f} aparitii")
    print(f"Simulari        : {args.sims:,}")
    if promoted:
        print(f"Fara jucatori   : {', '.join(sorted(promoted))}")

    totals = simulate(model, fixtures, weights, n_sims=args.sims)
    results = summarise(totals, weights)

    print(f"\n{'=' * 94}\nCURSA PENTRU GOLGHETER\n{'=' * 94}")
    print(
        f"{'#':>3} {'jucator':<26}{'echipa':<14}{'baza':>10}{'rata':>8}"
        f"{'goluri est.':>13}{'titlu':>9}{'20+':>8}{'25+':>8}"
    )
    print("-" * 94)
    for position, result in enumerate(results[: args.limit], start=1):
        base = f"{result.base_goals}g/{result.base_matches}m"
        print(
            f"{position:>3} {result.player[:25]:<26}{result.team[:13]:<14}{base:>10}"
            f"{result.rate:>8.3f}{result.expected_goals:>13.1f}"
            f"{result.title_probability:>9.1%}{result.p20_plus:>8.1%}{result.p25_plus:>8.1%}"
        )

    covered = sum(result.title_probability for result in results[: args.limit])
    print(f"\n  Primii {args.limit} acopera {covered:.1%} din probabilitatea totala.")


if __name__ == "__main__":
    main()
