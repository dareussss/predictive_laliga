"""Index de performanta ofensiva, ajustat pentru esantion si pozitie.

    python -m models.player_index
    python -m models.player_index --season 2023-24 --by-position
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

MINUTES_PER_MATCH = 90
DEFAULT_SEASON = "2024-25"
DEFAULT_MIN_MINUTES = 450
XG_FIRST_SEASON = "2017-18"
MIN_TRANSITION_MINUTES = 180

POSITION_GROUPS = ("GK", "DF", "MF", "FW")


@dataclass(frozen=True)
class Prior:
    """Media unui grup de jucatori si cate reprize de 90 cantareste ea."""

    mean: float
    shrinkage_90s: float
    n_players: int
    n_transitions: int


def primary_position(position: object) -> str:
    """FBref scrie 'FW,MF'; pentru grupare pastram doar pozitia principala."""
    if not isinstance(position, str) or not position:
        return "MF"
    head = position.split(",")[0].strip().upper()
    return head if head in POSITION_GROUPS else "MF"


def season_order(season: str) -> int:
    """'2024-25' -> 2024, ca sa putem verifica daca doua sezoane sunt consecutive."""
    return int(season.split("-")[0])


METRICS = {
    "contribution": lambda frame: frame["npxg"] + frame["xag"],
    "goals_assists": lambda frame: frame["goals"] + frame["assists"],
    "goals": lambda frame: frame["goals"],
}


def aggregate_seasons(players: pd.DataFrame, metric: str = "contribution") -> pd.DataFrame:
    """Un rand per jucator si sezon, insumand cluburile in caz de transfer."""
    if metric not in METRICS:
        raise ValueError(f"Metrica necunoscuta: {metric!r}. Alege din {sorted(METRICS)}.")

    frame = players[players["minutes"] > 0].copy()
    frame["group"] = frame["position"].map(primary_position)
    frame["value"] = METRICS[metric](frame)
    frame = frame.dropna(subset=["value", "minutes"])

    grouped = (
        frame.groupby(["player_id", "season"])
        .agg(
            name=("name", "first"),
            squad=("squad", "last"),
            group=("group", "first"),
            age=("age", "max"),
            minutes=("minutes", "sum"),
            value=("value", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
        )
        .reset_index()
    )
    grouped["exposure"] = grouped["minutes"] / MINUTES_PER_MATCH
    grouped["rate"] = grouped["value"] / grouped["exposure"]
    return grouped


def build_transitions(seasons: pd.DataFrame) -> pd.DataFrame:
    """Perechi (sezon, sezonul imediat urmator) pentru acelasi jucator."""
    frame = seasons.copy()
    frame["order"] = frame["season"].map(season_order)

    nxt = frame.copy()
    nxt["order"] = nxt["order"] - 1

    merged = frame.merge(
        nxt[["player_id", "order", "rate", "exposure"]],
        on=["player_id", "order"],
        suffixes=("", "_next"),
    )
    return merged[
        (merged["minutes"] >= MIN_TRANSITION_MINUTES)
        & (merged["exposure_next"] * MINUTES_PER_MATCH >= MIN_TRANSITION_MINUTES)
    ]


def fit_shrinkage(transitions: pd.DataFrame, mean: float) -> float:
    """Constanta de contractie care minimizeaza eroarea de predictie a sezonului urmator.

    Exprimata in reprize de 90 de minute: la o expunere egala cu k, rata proprie si
    media grupului cantaresc la fel.
    """
    value = transitions["value"].to_numpy()
    exposure = transitions["exposure"].to_numpy()
    rate_next = transitions["rate_next"].to_numpy()
    weight = transitions["exposure_next"].to_numpy()

    def error(k: float) -> float:
        predicted = (value + mean * k) / (exposure + k)
        return float(np.sum(weight * (predicted - rate_next) ** 2))

    result = minimize_scalar(error, bounds=(0.01, 500.0), method="bounded")
    return float(result.x)


def fit_priors(seasons: pd.DataFrame) -> dict[str, Prior]:
    """Media si constanta de contractie, estimate separat pe fiecare pozitie."""
    transitions = build_transitions(seasons)
    priors: dict[str, Prior] = {}

    for group, block in seasons.groupby("group"):
        mean = block["value"].sum() / block["exposure"].sum()
        group_transitions = transitions[transitions["group"] == group]

        if len(group_transitions) < 30:
            shrinkage = float(block["exposure"].median())
        else:
            shrinkage = fit_shrinkage(group_transitions, mean)

        priors[group] = Prior(
            mean=float(mean),
            shrinkage_90s=shrinkage,
            n_players=len(block),
            n_transitions=len(group_transitions),
        )

    return priors


def build_index(
    players: pd.DataFrame,
    season: str = DEFAULT_SEASON,
    min_minutes: int = DEFAULT_MIN_MINUTES,
) -> tuple[pd.DataFrame, dict[str, Prior]]:
    """Indexul unui sezon, cu contractia calibrata pe tot istoricul disponibil."""
    if season < XG_FIRST_SEASON:
        raise ValueError(f"xG exista abia din {XG_FIRST_SEASON}; {season} nu poate fi indexat.")

    seasons = aggregate_seasons(players)
    priors = fit_priors(seasons)

    current = seasons[seasons["season"] == season].copy()
    if current.empty:
        raise ValueError(f"Niciun jucator cu date complete in {season}.")

    constants = current["group"].map(lambda group: priors[group].shrinkage_90s)
    means = current["group"].map(lambda group: priors[group].mean)

    current["index_per90"] = (current["value"] + means * constants) / (
        current["exposure"] + constants
    )
    current["shrinkage"] = 1.0 - current["exposure"] / (current["exposure"] + constants)

    current["z_score"] = current.groupby("group")["index_per90"].transform(
        lambda column: (column - column.mean()) / column.std(ddof=0)
    )

    eligible = current[current["minutes"] >= min_minutes]
    return eligible.sort_values("index_per90", ascending=False).reset_index(drop=True), priors


def _print_priors(priors: dict[str, Prior]) -> None:
    print(f"\n{'pozitie':<9}{'jucatori':>10}{'tranzitii':>11}{'medie/90':>11}{'contractie k (90-uri)':>24}")
    print("-" * 65)
    for group in POSITION_GROUPS:
        prior = priors.get(group)
        if prior is None:
            continue
        print(
            f"{group:<9}{prior.n_players:>10}{prior.n_transitions:>11}"
            f"{prior.mean:>11.3f}{prior.shrinkage_90s:>24.1f}"
        )


def _print_top(index: pd.DataFrame, group: str | None, limit: int) -> None:
    subset = index if group is None else index[index["group"] == group]
    subset = subset.head(limit)
    title = "TOATE POZITIILE" if group is None else f"POZITIA {group}"

    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")
    print(
        f"{'#':>3} {'jucator':<24}{'echipa':<17}{'poz':<5}{'min':>6}{'G+A':>6}"
        f"{'brut':>8}{'index':>8}{'z':>7}{'contractie':>12}"
    )
    print("-" * 92)
    for position, row in enumerate(subset.itertuples(index=False), start=1):
        print(
            f"{position:>3} {row.name[:23]:<24}{row.squad[:16]:<17}{row.group:<5}"
            f"{int(row.minutes):>6}{int(row.goals + row.assists):>6}"
            f"{row.rate:>8.2f}{row.index_per90:>8.2f}{row.z_score:>7.2f}{row.shrinkage:>11.0%}"
        )


def main() -> None:
    from db.queries import load_player_seasons

    parser = argparse.ArgumentParser(description="Index de performanta ofensiva.")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--min-minutes", type=int, default=DEFAULT_MIN_MINUTES)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--by-position", action="store_true", help="Cate un top pe fiecare pozitie.")
    args = parser.parse_args()

    players = load_player_seasons()
    index, priors = build_index(players, season=args.season, min_minutes=args.min_minutes)

    print(f"Sezon: {args.season}   |   prag afisare: {args.min_minutes} minute")
    print(f"Jucatori indexati: {len(index)}")
    _print_priors(priors)

    if args.by_position:
        for group in POSITION_GROUPS:
            _print_top(index, group, args.limit)
    else:
        _print_top(index, None, args.limit)


if __name__ == "__main__":
    main()
