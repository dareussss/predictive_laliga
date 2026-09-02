"""Verificari de calitate pentru statisticile de jucator.

    python -m pipeline.player_checks

Invariantii de mai jos sunt afirmatii despre cum trebuie sa arate datele, nu despre
cum arata.
"""

from __future__ import annotations

import pandas as pd

from db.queries import load_player_seasons
from db.session import init_db

XG_FIRST_SEASON = "2017-18"
MIN_PLAUSIBLE_AGE = 14
MAX_PLAUSIBLE_AGE = 45
MINUTES_PER_MATCH_TOLERANCE = 95

MAX_MISSING_AGE_SHARE = 0.005


def run_checks(players: pd.DataFrame) -> list[str]:
    """Returneaza problemele gasite; lista goala inseamna date sanatoase."""
    problems: list[str] = []

    duplicates = int(players.duplicated(subset=["player_id", "season", "squad"]).sum())
    if duplicates:
        problems.append(f"{duplicates} randuri duplicate pe (player_id, sezon, echipa).")

    missing_age = int(players["age"].isna().sum())
    if missing_age / len(players) > MAX_MISSING_AGE_SHARE:
        problems.append(
            f"{missing_age} randuri fara varsta "
            f"({missing_age / len(players):.1%}, peste pragul de {MAX_MISSING_AGE_SHARE:.1%})."
        )

    ages = players["age"].dropna()
    outside = ages[(ages < MIN_PLAUSIBLE_AGE) | (ages > MAX_PLAUSIBLE_AGE)]
    if not outside.empty:
        problems.append(f"{len(outside)} varste in afara intervalului plauzibil.")

    over_played = players[players["minutes"] > players["matches"] * MINUTES_PER_MATCH_TOLERANCE]
    if not over_played.empty:
        problems.append(f"{len(over_played)} jucatori cu mai multe minute decat permit meciurile.")

    more_starts = players[players["starts"] > players["matches"]]
    if not more_starts.empty:
        problems.append(f"{len(more_starts)} jucatori cu mai multe titularizari decat meciuri.")

    bad_penalties = players[players["penalties"] > players["penalties_attempted"]]
    if not bad_penalties.empty:
        problems.append(f"{len(bad_penalties)} jucatori cu mai multe penalty-uri marcate decat executate.")

    fewer_goals = players[players["goals"] < players["penalties"]]
    if not fewer_goals.empty:
        problems.append(f"{len(fewer_goals)} jucatori cu mai putine goluri decat penalty-uri marcate.")

    negatives = players[["minutes", "goals", "assists", "matches"]].lt(0).any(axis=1)
    if negatives.any():
        problems.append(f"{int(negatives.sum())} randuri cu valori negative.")

    early = players[players["season"] < XG_FIRST_SEASON]
    if early["xg"].notna().any():
        problems.append(f"xG prezent inainte de {XG_FIRST_SEASON}, cand sursa nu il are.")

    late = players[players["season"] >= XG_FIRST_SEASON]
    coverage = late["xg"].notna().mean() if len(late) else 1.0
    if coverage < 0.95:
        problems.append(f"xG acoperit doar {coverage:.1%} dupa {XG_FIRST_SEASON}.")

    return problems


def _section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def report(players: pd.DataFrame) -> None:
    """Rezumat descriptiv al datelor de jucator."""
    _section("REZUMAT")
    print(f"Randuri jucator-sezon : {len(players):,}")
    print(f"Jucatori unici        : {players['name'].nunique():,}")
    print(f"Sezoane               : {players['season'].nunique()}  "
          f"({players['season'].min()} -> {players['season'].max()})")
    print(f"Minute totale         : {players['minutes'].sum():,.0f}")
    print(f"Goluri totale         : {players['goals'].sum():,.0f}")

    _section("ACOPERIRE PE SEZON")
    summary = players.groupby("season").agg(
        jucatori=("name", "size"),
        echipe=("squad", "nunique"),
        cu_xg=("xg", lambda column: column.notna().mean()),
        varsta_mediana=("age", "median"),
    )
    print(f"{'sezon':<10}{'jucatori':>10}{'echipe':>9}{'cu xG':>9}{'varsta med.':>13}")
    print("-" * 51)
    for season, row in summary.iterrows():
        print(
            f"{season:<10}{int(row.jucatori):>10}{int(row.echipe):>9}"
            f"{row.cu_xg:>9.0%}{row.varsta_mediana:>13.0f}"
        )

    _section("DISTRIBUTIA PE POZITII (ultimul sezon complet)")
    latest = players[players["season"] == "2024-25"]
    positions = latest["position"].value_counts().head(8)
    for position, count in positions.items():
        print(f"  {position:<12}{count:>5}")

    _section("MINUTE: CATI JUCATORI AU ESANTION UTILIZABIL")
    for threshold in (0, 450, 900, 1800):
        eligible = (latest["minutes"] >= threshold).sum()
        print(f"  peste {threshold:>5} minute: {eligible:>4} jucatori")


def main() -> None:
    init_db()
    players = load_player_seasons()
    if players.empty:
        raise SystemExit("Nicio statistica de jucator. Ruleaza ingestion.fbref_players.")

    report(players)

    _section("VERIFICARI DE CALITATE")
    problems = run_checks(players)
    for problem in problems:
        print(f"  [X] {problem}")
    if problems:
        raise SystemExit(1)
    print("  [OK] Toate verificarile au trecut.")


if __name__ == "__main__":
    main()
