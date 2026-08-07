"""Verificari de calitate a datelor.

    python -m pipeline.quality_checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from db.queries import load_matches
from db.session import init_db

EXPECTED_MATCHES_PER_SEASON = 380
EXPECTED_TEAMS_PER_SEASON = 20
MAX_PLAUSIBLE_GOALS = 12


def _teams_per_season(matches: pd.DataFrame) -> pd.Series:
    stacked = pd.concat(
        [
            matches[["season", "home_team"]].rename(columns={"home_team": "team"}),
            matches[["season", "away_team"]].rename(columns={"away_team": "team"}),
        ]
    )
    return stacked.groupby("season")["team"].nunique()


def _derived_result(matches: pd.DataFrame) -> np.ndarray:
    return np.select(
        [matches["home_goals"] > matches["away_goals"], matches["home_goals"] < matches["away_goals"]],
        ["H", "A"],
        default="D",
    )


def run_checks(matches: pd.DataFrame) -> list[str]:
    """Returneaza problemele gasite; lista goala inseamna date sanatoase."""
    problems: list[str] = []

    per_season = matches.groupby("season").size()
    wrong_count = per_season[per_season != EXPECTED_MATCHES_PER_SEASON]
    if not wrong_count.empty:
        problems.append(f"Sezoane cu numar gresit de meciuri:\n{wrong_count.to_string()}")

    per_season_teams = _teams_per_season(matches)
    wrong_teams = per_season_teams[per_season_teams != EXPECTED_TEAMS_PER_SEASON]
    if not wrong_teams.empty:
        problems.append(f"Sezoane fara {EXPECTED_TEAMS_PER_SEASON} de echipe:\n{wrong_teams.to_string()}")

    if matches["home_team"].eq(matches["away_team"]).any():
        problems.append("Exista meciuri in care gazda este si oaspete.")

    mismatched = int((_derived_result(matches) != matches["result"]).sum())
    if mismatched:
        problems.append(f"{mismatched} meciuri unde `result` nu corespunde scorului.")

    goals = matches[["home_goals", "away_goals"]]
    if (goals < 0).any().any():
        problems.append("Goluri negative.")
    if (goals > MAX_PLAUSIBLE_GOALS).any().any():
        problems.append(f"Scoruri implauzibile (peste {MAX_PLAUSIBLE_GOALS} goluri).")

    duplicates = int(matches.duplicated(subset=["season", "home_team", "away_team"]).sum())
    if duplicates:
        problems.append(f"{duplicates} meciuri duplicate pe (sezon, gazda, oaspete).")

    return problems


def _section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def report(matches: pd.DataFrame) -> None:
    """Afiseaza un rezumat descriptiv al datelor."""
    teams = set(matches["home_team"]) | set(matches["away_team"])

    _section("REZUMAT DATE")
    print(f"Meciuri:      {len(matches):,}")
    print(f"Sezoane:      {matches['season'].nunique()}  ({matches['season'].min()} -> {matches['season'].max()})")
    print(f"Echipe unice: {len(teams)}")
    print(f"Perioada:     {matches['date'].min()} -> {matches['date'].max()}")

    _section("AVANTAJUL TERENULUI PROPRIU")
    outcomes = matches["result"].value_counts(normalize=True)
    print(f"Victorie gazda:   {outcomes.get('H', 0):.1%}")
    print(f"Egal:             {outcomes.get('D', 0):.1%}")
    print(f"Victorie oaspete: {outcomes.get('A', 0):.1%}")
    print(f"\nGoluri/meci gazda:   {matches['home_goals'].mean():.2f}")
    print(f"Goluri/meci oaspete: {matches['away_goals'].mean():.2f}")
    print(f"Total goluri/meci:   {(matches['home_goals'] + matches['away_goals']).mean():.2f}")

    _section("CELE MAI FRECVENTE SCORURI")
    scorelines = (
        matches["home_goals"].astype(int).astype(str) + "-" + matches["away_goals"].astype(int).astype(str)
    )
    for scoreline, share in scorelines.value_counts(normalize=True).head(8).items():
        print(f"  {scoreline}   {share:6.2%}")

    _section("ACOPERIREA COTELOR DE INCHIDERE")
    coverage = matches.groupby("season")["avg_close_home"].apply(lambda column: column.notna().mean())
    complete = coverage[coverage > 0.9]
    print(f"Sezoane cu cote de inchidere complete: {len(complete)}")
    if not complete.empty:
        print(f"  {complete.index.min()} -> {complete.index.max()}")


def main() -> None:
    init_db()
    matches = load_matches()

    report(matches)

    _section("VERIFICARI DE CALITATE")
    problems = run_checks(matches)
    for problem in problems:
        print(f"  [X] {problem}")
    if problems:
        raise SystemExit(1)
    print("  [OK] Toate verificarile au trecut.")


if __name__ == "__main__":
    main()
