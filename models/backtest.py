"""Backtesting walk-forward pentru Dixon-Coles.

    python -m models.backtest
    python -m models.backtest --xi 0.0005 0.0018 0.003 --from-season 2023-24
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from db.queries import load_matches
from models.dixon_coles import DEFAULT_XI, DixonColes

OUTCOMES = ("H", "D", "A")
ODDS_COLUMNS = ["avg_close_home", "avg_close_draw", "avg_close_away"]
EPSILON = 1e-15
DEFAULT_FROM_SEASON = "2019-20"
DEFAULT_REFIT_DAYS = 7
DEFAULT_HISTORY_YEARS = 8.0


@dataclass(frozen=True)
class Metrics:
    """Cat de bune sunt niste probabilitati fata de ce s-a intamplat efectiv."""

    name: str
    n: int
    log_loss: float
    brier: float
    accuracy: float


def outcome_index(results: pd.Series) -> np.ndarray:
    """Codifica H/D/A ca 0/1/2."""
    mapping = {outcome: position for position, outcome in enumerate(OUTCOMES)}
    return results.map(mapping).to_numpy(dtype=int)


def implied_probabilities(odds: pd.DataFrame) -> np.ndarray:
    """Probabilitatile pietei, cu marja casei eliminata proportional."""
    raw = 1.0 / odds.to_numpy(dtype=float)
    return raw / raw.sum(axis=1, keepdims=True)


def evaluate(name: str, probabilities: np.ndarray, actual: np.ndarray) -> Metrics:
    """Log-loss, Brier si acuratete pentru un set de probabilitati."""
    probabilities = np.clip(probabilities, EPSILON, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    count = len(actual)
    rows = np.arange(count)

    log_loss = float(-np.log(probabilities[rows, actual]).mean())

    one_hot = np.zeros_like(probabilities)
    one_hot[rows, actual] = 1.0
    brier = float(((probabilities - one_hot) ** 2).sum(axis=1).mean())

    accuracy = float((probabilities.argmax(axis=1) == actual).mean())

    return Metrics(name=name, n=count, log_loss=log_loss, brier=brier, accuracy=accuracy)


def base_rates(matches: pd.DataFrame) -> np.ndarray:
    """Frecventele istorice H/D/A, folosite ca model naiv."""
    shares = matches["result"].value_counts(normalize=True)
    return np.array([shares.get(outcome, 0.0) for outcome in OUTCOMES])


def walk_forward(
    matches: pd.DataFrame,
    xi: float,
    from_season: str = DEFAULT_FROM_SEASON,
    refit_days: int = DEFAULT_REFIT_DAYS,
    history_years: float | None = DEFAULT_HISTORY_YEARS,
    use_rho: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Prezice fiecare meci din perioada de test folosind doar trecutul lui.

    Reantrenat cel mult o data la refit_days zile, mereu pe meciuri strict
    anterioare datei prezise.
    """
    test = matches[matches["season"] >= from_season]
    model = DixonColes(xi=xi, history_years=history_years, use_rho=use_rho)

    predictions: list[dict] = []
    skipped = 0
    last_fit: pd.Timestamp | None = None
    refits = 0

    for match_date, same_day in test.groupby("date", sort=True):
        if last_fit is None or (match_date - last_fit).days >= refit_days:
            model.fit(matches, as_of=match_date)
            last_fit = match_date
            refits += 1
            if verbose and refits % 25 == 0:
                print(f"  {match_date.date()}  refit #{refits}")

        for match in same_day.itertuples(index=False):
            try:
                prediction = model.predict(match.home_team, match.away_team)
            except KeyError:
                skipped += 1
                continue

            predictions.append(
                {
                    "season": match.season,
                    "date": match_date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "result": match.result,
                    "model_home": prediction.home_win,
                    "model_draw": prediction.draw,
                    "model_away": prediction.away_win,
                    "avg_close_home": match.avg_close_home,
                    "avg_close_draw": match.avg_close_draw,
                    "avg_close_away": match.avg_close_away,
                }
            )

    if verbose:
        print(f"  {refits} reantrenari, {skipped} meciuri sarite (echipa necunoscuta)")

    return pd.DataFrame(predictions)


def compare(predictions: pd.DataFrame, prior: pd.DataFrame) -> list[Metrics]:
    """Compara modelul cu un baseline naiv si cu cotele de inchidere."""
    graded = predictions.dropna(subset=ODDS_COLUMNS)
    actual = outcome_index(graded["result"])

    model_probabilities = graded[["model_home", "model_draw", "model_away"]].to_numpy()
    market_probabilities = implied_probabilities(graded[ODDS_COLUMNS])
    naive_probabilities = np.tile(base_rates(prior), (len(graded), 1))

    return [
        evaluate("Baseline (frecvente istorice)", naive_probabilities, actual),
        evaluate("Dixon-Coles", model_probabilities, actual),
        evaluate("Cote de inchidere", market_probabilities, actual),
    ]


def calibration(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Cat de des se intampla lucrurile carora modelul le da o anumita probabilitate."""
    stacked = pd.DataFrame(
        {
            "probability": pd.concat(
                [predictions[f"model_{outcome}"] for outcome in ("home", "draw", "away")],
                ignore_index=True,
            ),
            "happened": pd.concat(
                [(predictions["result"] == code).astype(float) for code in OUTCOMES],
                ignore_index=True,
            ),
        }
    )

    stacked["bucket"] = pd.cut(stacked["probability"], np.linspace(0.0, 1.0, bins + 1))
    grouped = stacked.groupby("bucket", observed=True).agg(
        n=("happened", "size"),
        predicted=("probability", "mean"),
        observed=("happened", "mean"),
    )
    return grouped.reset_index()


def _print_metrics(results: list[Metrics]) -> None:
    print(f"\n{'model':<32}{'n':>7}{'log-loss':>11}{'Brier':>9}{'acuratete':>11}")
    print("-" * 70)
    for metrics in results:
        print(
            f"{metrics.name:<32}{metrics.n:>7,}{metrics.log_loss:>11.4f}"
            f"{metrics.brier:>9.4f}{metrics.accuracy:>11.1%}"
        )


def _print_calibration(table: pd.DataFrame) -> None:
    print(f"\n{'interval':<16}{'n':>7}{'prezis':>10}{'observat':>11}{'eroare':>10}")
    print("-" * 54)
    for row in table.itertuples(index=False):
        error = row.observed - row.predicted
        print(
            f"{str(row.bucket):<16}{row.n:>7,}{row.predicted:>10.1%}"
            f"{row.observed:>11.1%}{error:>+10.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtesting walk-forward pentru Dixon-Coles.")
    parser.add_argument("--xi", type=float, nargs="+", default=[DEFAULT_XI])
    parser.add_argument("--from-season", default=DEFAULT_FROM_SEASON)
    parser.add_argument("--refit-days", type=int, default=DEFAULT_REFIT_DAYS)
    parser.add_argument("--history-years", type=float, default=DEFAULT_HISTORY_YEARS)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--no-rho", action="store_true", help="Dezactiveaza corectia Dixon-Coles.")
    args = parser.parse_args()

    matches = load_matches()
    prior = matches[matches["season"] < args.from_season]

    print(f"Perioada de test: din {args.from_season} ({(matches['season'] >= args.from_season).sum():,} meciuri)")
    print(f"Baseline calculat pe: {len(prior):,} meciuri anterioare")

    best: tuple[float, pd.DataFrame] | None = None

    for xi in args.xi:
        print(f"\n{'=' * 70}\nxi = {xi}\n{'=' * 70}")
        started = time.perf_counter()
        predictions = walk_forward(
            matches,
            xi=xi,
            from_season=args.from_season,
            refit_days=args.refit_days,
            history_years=args.history_years,
            use_rho=not args.no_rho,
            verbose=True,
        )
        elapsed = time.perf_counter() - started

        results = compare(predictions, prior)
        _print_metrics(results)
        print(f"\n  ({elapsed:.0f}s)")

        model_log_loss = results[1].log_loss
        if best is None or model_log_loss < best[0]:
            best = (model_log_loss, predictions)

    if args.calibration and best is not None:
        print(f"\n{'=' * 70}\nCALIBRARE (cel mai bun xi)\n{'=' * 70}")
        _print_calibration(calibration(best[1]))


if __name__ == "__main__":
    main()
