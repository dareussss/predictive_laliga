"""Curba de varsta si forecast individual pentru jucatori.

    python -m models.player_forecast
    python -m models.player_forecast --metric contribution --season 2024-25

Curba e estimata prin metoda diferentelor si iese cu varful la 26 de ani. Validarea
arata insa ca aplicarea ei inrautateste predictiile, asa ca ajustarea e dezactivata
implicit. Detaliile sunt in README.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.player_index import aggregate_seasons, build_transitions, fit_priors

DEFAULT_METRIC = "goals_assists"
DEFAULT_SEASON = "2024-25"
DEFAULT_SPLIT = "2022-23"
MIN_AGE = 18
MAX_AGE = 37
SMOOTH_WINDOW = 3


@dataclass(frozen=True)
class AgeCurve:
    """Cum se schimba productia unui jucator de la o varsta la urmatoarea."""

    deltas: pd.Series
    curve: pd.Series
    peak_age: int
    n_transitions: int

    def delta_at(self, age: float) -> float:
        """Schimbarea asteptata in anul urmator, in unitati relative la media pozitiei."""
        if not np.isfinite(age):
            return 0.0
        return float(self.deltas.get(int(round(age)), 0.0))


def position_means(seasons: pd.DataFrame) -> pd.Series:
    """Productia medie pe 90 de minute, pentru fiecare grup de pozitii."""
    return seasons.groupby("group").apply(
        lambda block: block["value"].sum() / block["exposure"].sum(), include_groups=False
    )


def estimate_age_curve(seasons: pd.DataFrame) -> AgeCurve:
    """Estimeaza efectul varstei prin diferente pereche, jucator cu jucator."""
    transitions = build_transitions(seasons)
    transitions = transitions[transitions["group"] != "GK"].dropna(subset=["age"])

    means = position_means(seasons)
    relative_now = transitions["rate"] / transitions["group"].map(means)
    relative_next = transitions["rate_next"] / transitions["group"].map(means)

    frame = pd.DataFrame(
        {
            "age": transitions["age"].round().astype(int),
            "delta": relative_next - relative_now,
            "weight": np.minimum(transitions["exposure"], transitions["exposure_next"]),
        }
    )
    frame = frame[(frame["age"] >= MIN_AGE) & (frame["age"] <= MAX_AGE)]

    grouped = frame.groupby("age")
    deltas = grouped.apply(
        lambda block: np.average(block["delta"], weights=block["weight"]),
        include_groups=False,
    )
    counts = grouped.size()

    ages = pd.RangeIndex(MIN_AGE, MAX_AGE + 1)
    deltas = deltas.reindex(ages).interpolate().fillna(0.0)
    smoothed = deltas.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()

    curve = smoothed.cumsum().shift(1).fillna(0.0)
    curve = curve - curve.max()

    return AgeCurve(
        deltas=smoothed,
        curve=curve,
        peak_age=int(curve.idxmax()),
        n_transitions=int(counts.sum()),
    )


def forecast(
    seasons: pd.DataFrame,
    season: str,
    priors: dict,
    age_curve: AgeCurve | None = None,
) -> pd.DataFrame:
    """Proiecteaza rata fiecarui jucator pentru sezonul urmator."""
    current = seasons[seasons["season"] == season].copy()
    if current.empty:
        raise ValueError(f"Niciun jucator in {season}.")

    means = position_means(seasons)
    constants = current["group"].map(lambda group: priors[group].shrinkage_90s)
    group_means = current["group"].map(lambda group: priors[group].mean)

    current["shrunk_rate"] = (current["value"] + group_means * constants) / (
        current["exposure"] + constants
    )

    if age_curve is None:
        current["age_effect"] = 0.0
    else:
        scale = current["group"].map(means)
        current["age_effect"] = current["age"].map(age_curve.delta_at) * scale

    current["projected_rate"] = (current["shrunk_rate"] + current["age_effect"]).clip(lower=0.0)
    return current


def _weighted_rmse(predicted: np.ndarray, actual: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weight * (predicted - actual) ** 2) / weight.sum()))


def validate(seasons: pd.DataFrame, split_season: str = DEFAULT_SPLIT) -> pd.DataFrame:
    """Compara predictia cu si fara ajustare de varsta, pe sezoane nefolosite la estimare."""
    train = seasons[seasons["season"] < split_season]
    if train.empty:
        raise ValueError("Perioada de antrenare e goala.")

    priors = fit_priors(train)
    age_curve = estimate_age_curve(train)
    means = position_means(train)

    transitions = build_transitions(seasons)
    test = transitions[
        (transitions["season"] >= split_season) & (transitions["group"] != "GK")
    ].dropna(subset=["age"])

    constants = test["group"].map(lambda group: priors[group].shrinkage_90s)
    group_means = test["group"].map(lambda group: priors[group].mean)

    shrunk = (test["value"] + group_means * constants) / (test["exposure"] + constants)
    age_effect = test["age"].map(age_curve.delta_at) * test["group"].map(means)

    actual = test["rate_next"].to_numpy()
    weight = test["exposure_next"].to_numpy()

    return pd.DataFrame(
        [
            {
                "model": "Media pozitiei",
                "rmse": _weighted_rmse(group_means.to_numpy(), actual, weight),
            },
            {
                "model": "Sezonul precedent, brut",
                "rmse": _weighted_rmse(test["rate"].to_numpy(), actual, weight),
            },
            {
                "model": "Contractat spre medie",
                "rmse": _weighted_rmse(shrunk.to_numpy(), actual, weight),
            },
            {
                "model": "Contractat + varsta",
                "rmse": _weighted_rmse(
                    (shrunk + age_effect).clip(lower=0.0).to_numpy(), actual, weight
                ),
            },
        ]
    ).assign(n=len(test))


def _print_curve(age_curve: AgeCurve) -> None:
    print(f"\n{'=' * 70}\nCURBA DE VARSTA  (varf la {age_curve.peak_age} ani, "
          f"{age_curve.n_transitions:,} tranzitii)\n{'=' * 70}")
    print(f"{'varsta':<8}{'schimbare/an':>15}{'fata de varf':>15}   ")
    print("-" * 70)

    peak = age_curve.curve.max()
    for age in age_curve.curve.index:
        relative = age_curve.curve[age] - peak
        bar_length = int(round((relative + 0.6) / 0.6 * 22))
        bar = "#" * max(bar_length, 0)
        marker = "  <- varf" if age == age_curve.peak_age else ""
        print(
            f"{age:<8}{age_curve.deltas[age]:>+15.3f}{relative:>15.3f}   {bar}{marker}"
        )


def main() -> None:
    from db.queries import load_player_seasons

    parser = argparse.ArgumentParser(description="Curba de varsta si forecast de jucator.")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="goals_assists | contribution | goals")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument(
        "--with-age",
        action="store_true",
        help="Aplica ajustarea de varsta. Validarea arata ca inrautateste predictiile.",
    )
    args = parser.parse_args()

    players = load_player_seasons()
    seasons = aggregate_seasons(players, metric=args.metric)

    print(f"Metrica: {args.metric}")
    print(f"Sezoane: {seasons['season'].nunique()} ({seasons['season'].min()} -> {seasons['season'].max()})")

    age_curve = estimate_age_curve(seasons)
    _print_curve(age_curve)

    print(f"\n{'=' * 70}\nVALIDARE  (curba estimata pe sezoane < {args.split})\n{'=' * 70}")
    results = validate(seasons, split_season=args.split)
    print(f"{'model':<28}{'RMSE':>10}{'n':>8}")
    print("-" * 46)
    for row in results.itertuples(index=False):
        print(f"{row.model:<28}{row.rmse:>10.4f}{row.n:>8,}")

    priors = fit_priors(seasons)
    projected = forecast(seasons, args.season, priors, age_curve if args.with_age else None)
    projected = projected[projected["minutes"] >= 900]

    applied = "cu ajustare de varsta" if args.with_age else "fara ajustare de varsta"
    print(f"\n{'=' * 82}\nPROIECTIE PENTRU SEZONUL DUPA {args.season}  ({applied})\n{'=' * 82}")
    print(f"{'jucator':<24}{'echipa':<17}{'poz':<5}{'varsta':>7}{'acum':>8}{'contractat':>12}{'varsta':>9}{'proiectat':>11}")
    print("-" * 82)
    for row in projected.nlargest(args.limit, "projected_rate").itertuples(index=False):
        print(
            f"{row.name[:23]:<24}{row.squad[:16]:<17}{row.group:<5}{row.age:>7.0f}"
            f"{row.rate:>8.2f}{row.shrunk_rate:>12.2f}{row.age_effect:>+9.3f}{row.projected_rate:>11.2f}"
        )


if __name__ == "__main__":
    main()
