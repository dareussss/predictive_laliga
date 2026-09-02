"""Forma asteptata pe urmatoarele meciuri din calendar.

    python -m models.form_forecast
    python -m models.form_forecast --team Barcelona --matches 5

Distributia punctelor e calculata exact prin convolutie: cu 3^n combinatii posibile,
nu are rost sa fie aproximata prin simulare.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models import promoted_prior
from models.dixon_coles import DEFAULT_XI, DixonColes

DEFAULT_HORIZON = 5
MIN_RELIABLE_WEIGHT = 10.0
POINTS = (3, 1, 0)


@dataclass(frozen=True)
class FixtureForecast:
    """Predictia unui meci, din perspectiva echipei analizate."""

    matchday: int
    date: pd.Timestamp
    opponent: str
    at_home: bool
    win: float
    draw: float
    loss: float
    goals_for: float
    goals_against: float

    @property
    def expected_points(self) -> float:
        return 3.0 * self.win + 1.0 * self.draw


@dataclass(frozen=True)
class FormForecast:
    """Forma asteptata a unei echipe pe un orizont de cateva meciuri."""

    team: str
    fixtures: list[FixtureForecast]
    points_distribution: np.ndarray

    @property
    def expected_points(self) -> float:
        return float(sum(fixture.expected_points for fixture in self.fixtures))

    @property
    def max_points(self) -> int:
        return 3 * len(self.fixtures)

    def probability_of_at_least(self, points: int) -> float:
        return float(self.points_distribution[points:].sum())


def build_model(
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    xi: float = DEFAULT_XI,
    as_of: pd.Timestamp | None = None,
    min_weight: float = MIN_RELIABLE_WEIGHT,
) -> tuple[DixonColes, set[str]]:
    """Potriveste modelul si completeaza echipele fara istoric recent utilizabil.

    O echipa absenta ani de zile apare in date cu pondere aproape nula, deci
    parametrii ei raman la zero -- ceea ce s-ar citi gresit ca "echipa medie".
    """
    model = DixonColes(xi=xi).fit(matches, as_of=as_of)
    prior = promoted_prior.load(matches)

    upcoming_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
    substituted: set[str] = set()

    for team in upcoming_teams:
        if model.team_weight.get(team, 0.0) < min_weight:
            model.register_team(team, prior.attack, prior.defence)
            substituted.add(team)

    return model, substituted


def points_distribution(fixtures: list[FixtureForecast]) -> np.ndarray:
    """Distributia exacta a punctelor totale, prin convolutie meci cu meci."""
    distribution = np.zeros(3 * len(fixtures) + 1)
    distribution[0] = 1.0

    for fixture in fixtures:
        updated = np.zeros_like(distribution)
        for points, probability in zip(POINTS, (fixture.win, fixture.draw, fixture.loss)):
            updated[points:] += distribution[: len(distribution) - points] * probability
        distribution = updated

    return distribution


def forecast(
    model: DixonColes,
    fixtures: pd.DataFrame,
    team: str,
    horizon: int = DEFAULT_HORIZON,
) -> FormForecast:
    """Predictia formei pentru urmatoarele `horizon` meciuri ale echipei."""
    involved = fixtures[(fixtures["home_team"] == team) | (fixtures["away_team"] == team)]
    upcoming = involved.sort_values("date").head(horizon)

    forecasts: list[FixtureForecast] = []
    for row in upcoming.itertuples(index=False):
        at_home = row.home_team == team
        opponent = row.away_team if at_home else row.home_team

        prediction = model.predict(row.home_team, row.away_team)

        forecasts.append(
            FixtureForecast(
                matchday=int(row.matchday) if pd.notna(row.matchday) else 0,
                date=row.date,
                opponent=opponent,
                at_home=at_home,
                win=prediction.home_win if at_home else prediction.away_win,
                draw=prediction.draw,
                loss=prediction.away_win if at_home else prediction.home_win,
                goals_for=prediction.expected_home_goals if at_home else prediction.expected_away_goals,
                goals_against=prediction.expected_away_goals if at_home else prediction.expected_home_goals,
            )
        )

    return FormForecast(
        team=team,
        fixtures=forecasts,
        points_distribution=points_distribution(forecasts),
    )


def _print_team(form: FormForecast, substituted: set[str]) -> None:
    flag = "  [rating estimat din prior]" if form.team in substituted else ""
    print(f"\n{form.team}{flag}")
    print(f"{'et.':>4}{'data':>13}{'':>3}{'adversar':<14}{'V':>7}{'X':>7}{'I':>7}{'goluri':>12}{'pct':>6}")
    print("-" * 76)

    for fixture in form.fixtures:
        venue = "vs" if fixture.at_home else "la"
        marker = "*" if fixture.opponent in substituted else " "
        print(
            f"{fixture.matchday:>4}{str(fixture.date.date()):>13}{venue:>3} "
            f"{fixture.opponent + marker:<14}"
            f"{fixture.win:>7.1%}{fixture.draw:>7.1%}{fixture.loss:>7.1%}"
            f"{fixture.goals_for:>7.2f}-{fixture.goals_against:<4.2f}"
            f"{fixture.expected_points:>6.2f}"
        )

    print(f"\n  Puncte asteptate: {form.expected_points:.2f} din {form.max_points}")
    print(f"  Sanse de minim 7 puncte: {form.probability_of_at_least(7):.1%}")
    print(f"  Sanse de maxim ({form.max_points}p): {form.points_distribution[-1]:.2%}")


def _print_ranking(forms: list[FormForecast], substituted: set[str], horizon: int) -> None:
    ordered = sorted(forms, key=lambda form: form.expected_points, reverse=True)
    maximum = 3 * horizon

    print(f"\n{'=' * 72}")
    print(f"PUNCTE ASTEPTATE IN URMATOARELE {horizon} ETAPE")
    print(f"{'=' * 72}")
    print(f"{'echipa':<14}{'pct':>7}{'din':>5}{'>=7p':>8}{'maxim':>8}{'acasa':>7}   distributie")
    print("-" * 72)

    for form in ordered:
        home_games = sum(1 for fixture in form.fixtures if fixture.at_home)
        flag = " *" if form.team in substituted else "  "
        bar = "#" * int(round(form.expected_points / maximum * 20))
        print(
            f"{form.team + flag:<14}{form.expected_points:>7.2f}{maximum:>5}"
            f"{form.probability_of_at_least(7):>8.1%}"
            f"{form.points_distribution[-1]:>8.2%}"
            f"{home_games:>7}   {bar}"
        )

    if substituted:
        print(f"\n  * rating din prior-ul promovatelor: {', '.join(sorted(substituted))}")


def main() -> None:
    from db.queries import load_fixtures, load_matches

    parser = argparse.ArgumentParser(description="Forma asteptata pe urmatoarele meciuri.")
    parser.add_argument("--team", help="Doar o echipa; implicit toate.")
    parser.add_argument("--matches", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--season", default=None, help="Sezonul de calendar, ex. 2026-27.")
    args = parser.parse_args()

    matches = load_matches()
    fixtures = load_fixtures(args.season)
    if fixtures.empty:
        raise SystemExit("Niciun meci programat. Ruleaza mai intai ingestion.football_data_org.")

    model, substituted = build_model(matches, fixtures, xi=args.xi)

    print(f"Model antrenat pe {model.n_matches:,} meciuri ({model.effective_n:,.0f} efective)")
    print(f"Calendar: {len(fixtures)} meciuri, de la {fixtures['date'].min().date()}")

    if args.team:
        _print_team(forecast(model, fixtures, args.team, args.matches), substituted)
        return

    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    forms = [forecast(model, fixtures, team, args.matches) for team in teams]
    _print_ranking(forms, substituted, args.matches)


if __name__ == "__main__":
    main()
