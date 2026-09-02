"""Cat de bune sunt, in medie, echipele nou-promovate in La Liga.

    python -m models.promoted_prior
    python -m models.promoted_prior --refresh

Modelul principal nu poate estima o echipa absenta ani de zile din prima liga. In loc
sa o tratam ca fiind de nivel mediu, ii dam ratingul tipic al unei promovate, masurat
din istoric.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import config
from models.dixon_coles import DixonColes

CACHE_PATH = config.PROCESSED_DIR / "promoted_prior.json"


@dataclass(frozen=True)
class PromotedPrior:
    """Ratingul tipic al unei echipe in primul sezon dupa promovare."""

    attack: float
    defence: float
    attack_std: float
    defence_std: float
    n_teams: int
    n_seasons: int


def season_squads(matches: pd.DataFrame) -> dict[str, set[str]]:
    """Echipele care au jucat in fiecare sezon."""
    return {
        season: set(group["home_team"]) | set(group["away_team"])
        for season, group in matches.groupby("season")
    }


def newcomers(matches: pd.DataFrame) -> dict[str, set[str]]:
    """Pentru fiecare sezon, echipele absente din sezonul precedent."""
    squads = season_squads(matches)
    seasons = sorted(squads)
    return {
        season: squads[season] - squads[previous]
        for previous, season in zip(seasons, seasons[1:])
    }


def estimate(matches: pd.DataFrame) -> PromotedPrior:
    """Potriveste modelul sezon cu sezon si masoara ratingul promovatelor."""
    promoted = newcomers(matches)
    attacks: list[float] = []
    defences: list[float] = []
    seasons_used = 0

    for season, teams in promoted.items():
        if not teams:
            continue

        season_matches = matches[matches["season"] == season]
        model = DixonColes(xi=0.0).fit(season_matches)

        for team in teams:
            if team in model.attack:
                attacks.append(model.attack[team])
                defences.append(model.defence[team])
        seasons_used += 1

    if not attacks:
        raise ValueError("Nicio promovare identificata.")

    return PromotedPrior(
        attack=float(np.mean(attacks)),
        defence=float(np.mean(defences)),
        attack_std=float(np.std(attacks, ddof=1)),
        defence_std=float(np.std(defences, ddof=1)),
        n_teams=len(attacks),
        n_seasons=seasons_used,
    )


def load(matches: pd.DataFrame | None = None, refresh: bool = False) -> PromotedPrior:
    """Citeste prior-ul din cache, recalculandu-l daca lipseste sau daca se cere."""
    if CACHE_PATH.exists() and not refresh:
        return PromotedPrior(**json.loads(CACHE_PATH.read_text(encoding="utf-8")))

    if matches is None:
        from db.queries import load_matches

        matches = load_matches()

    prior = estimate(matches)
    CACHE_PATH.write_text(json.dumps(asdict(prior), indent=2), encoding="utf-8")
    return prior


def main() -> None:
    from db.queries import load_matches

    parser = argparse.ArgumentParser(description="Estimeaza ratingul tipic al promovatelor.")
    parser.add_argument("--refresh", action="store_true", help="Recalculeaza, ignora cache-ul.")
    args = parser.parse_args()

    matches = load_matches()

    print("Promovari identificate pe sezon:\n")
    for season, teams in newcomers(matches).items():
        print(f"  {season}  {', '.join(sorted(teams)) if teams else '(niciuna)'}")

    prior = load(matches, refresh=args.refresh)

    print(f"\n{'=' * 58}\nPRIOR PENTRU ECHIPE PROMOVATE\n{'=' * 58}")
    print(f"Bazat pe {prior.n_teams} promovari din {prior.n_seasons} sezoane\n")
    print(f"{'':<12}{'medie':>10}{'abatere std':>14}")
    print("-" * 36)
    print(f"{'atac':<12}{prior.attack:>10.3f}{prior.attack_std:>14.3f}")
    print(f"{'aparare':<12}{prior.defence:>10.3f}{prior.defence_std:>14.3f}")

    print("\nIn goluri, fata de o echipa medie de La Liga:")
    print(f"  marcheaza  x{np.exp(prior.attack):.3f}")
    print(f"  primeste   x{np.exp(-prior.defence):.3f}")
    print(f"\nCache: {CACHE_PATH}")


if __name__ == "__main__":
    main()
