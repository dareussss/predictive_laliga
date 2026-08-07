"""Modelul Dixon-Coles (1997) pentru predictia scorului.

    python -m models.dixon_coles
    python -m models.dixon_coles --home Barcelona --away "Real Madrid"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

DEFAULT_XI = 0.0018
DEFAULT_MAX_GOALS = 10
TAU_FLOOR = 1e-10


@dataclass(frozen=True)
class MatchPrediction:
    """Rezultatul complet al unei predictii de meci."""

    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float
    most_likely_score: tuple[int, int]
    most_likely_score_prob: float
    over_2_5: float
    both_teams_score: float

    def __str__(self) -> str:
        home, away = self.most_likely_score
        return (
            f"{self.home_team} vs {self.away_team}\n"
            f"  1  {self.home_win:6.1%}   X  {self.draw:6.1%}   2  {self.away_win:6.1%}\n"
            f"  Goluri asteptate: {self.expected_home_goals:.2f} - {self.expected_away_goals:.2f}\n"
            f"  Scor cel mai probabil: {home}-{away} ({self.most_likely_score_prob:.1%})\n"
            f"  Peste 2.5 goluri: {self.over_2_5:.1%}   Ambele inscriu: {self.both_teams_score:.1%}"
        )


def _tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Corectia Dixon-Coles pentru dependenta dintre goluri la scoruri mici."""
    tau = np.ones(len(home_goals))

    goalless = (home_goals == 0) & (away_goals == 0)
    away_only = (home_goals == 0) & (away_goals == 1)
    home_only = (home_goals == 1) & (away_goals == 0)
    one_each = (home_goals == 1) & (away_goals == 1)

    tau[goalless] = 1.0 - lam[goalless] * mu[goalless] * rho
    tau[away_only] = 1.0 + lam[away_only] * rho
    tau[home_only] = 1.0 + mu[home_only] * rho
    tau[one_each] = 1.0 - rho

    return np.clip(tau, TAU_FLOOR, None)


def _poisson_logpmf(goals: np.ndarray, log_rate: np.ndarray) -> np.ndarray:
    return goals * log_rate - np.exp(log_rate) - gammaln(goals + 1.0)


@dataclass
class DixonColes:
    """Model de forta atac/aparare cu avantajul terenului si ponderare temporala.

    xi controleaza cat de repede se uita meciurile vechi. 0.0018/zi inseamna ca un
    meci de acum un an cantareste aproximativ jumatate cat unul de ieri.
    """

    xi: float = DEFAULT_XI
    max_goals: int = DEFAULT_MAX_GOALS

    teams: list[str] = field(default_factory=list, init=False)
    attack: dict[str, float] = field(default_factory=dict, init=False)
    defence: dict[str, float] = field(default_factory=dict, init=False)
    home_advantage: float = field(default=0.0, init=False)
    rho: float = field(default=0.0, init=False)
    log_likelihood: float = field(default=0.0, init=False)
    n_matches: int = field(default=0, init=False)
    effective_n: float = field(default=0.0, init=False)
    converged: bool = field(default=False, init=False)

    def fit(self, matches: pd.DataFrame, as_of: datetime | None = None) -> "DixonColes":
        """Estimeaza parametrii prin maxima verosimilitate ponderata temporal.

        Doar meciurile jucate strict inainte de `as_of` sunt folosite, ceea ce face
        imposibila scurgerea de informatie din viitor in backtesting.
        """
        history = matches if as_of is None else matches[matches["date"] < as_of]
        history = history.dropna(subset=["home_goals", "away_goals"])
        if history.empty:
            raise ValueError("Niciun meci disponibil pentru fit.")

        reference = as_of if as_of is not None else history["date"].max()
        age_days = (reference - history["date"]).dt.days.to_numpy(dtype=float)
        weights = np.exp(-self.xi * age_days)

        self.teams = sorted(set(history["home_team"]) | set(history["away_team"]))
        index = {team: position for position, team in enumerate(self.teams)}
        n_teams = len(self.teams)

        home_idx = history["home_team"].map(index).to_numpy()
        away_idx = history["away_team"].map(index).to_numpy()
        home_goals = history["home_goals"].to_numpy(dtype=float)
        away_goals = history["away_goals"].to_numpy(dtype=float)

        def negative_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n_teams]
            defence = params[n_teams : 2 * n_teams]
            home_advantage, rho = params[-2], params[-1]

            attack = attack - attack.mean()
            defence = defence - defence.mean()

            log_lam = attack[home_idx] - defence[away_idx] + home_advantage
            log_mu = attack[away_idx] - defence[home_idx]

            log_prob = _poisson_logpmf(home_goals, log_lam) + _poisson_logpmf(away_goals, log_mu)
            correction = np.log(_tau(home_goals, away_goals, np.exp(log_lam), np.exp(log_mu), rho))

            return -float(np.sum(weights * (log_prob + correction)))

        start = np.concatenate([np.zeros(n_teams), np.zeros(n_teams), [0.25], [-0.05]])
        bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(-1.0, 1.0), (-0.2, 0.2)]

        result = minimize(negative_log_likelihood, start, method="L-BFGS-B", bounds=bounds)

        attack = result.x[:n_teams] - result.x[:n_teams].mean()
        defence = result.x[n_teams : 2 * n_teams] - result.x[n_teams : 2 * n_teams].mean()

        self.attack = dict(zip(self.teams, attack))
        self.defence = dict(zip(self.teams, defence))
        self.home_advantage = float(result.x[-2])
        self.rho = float(result.x[-1])
        self.log_likelihood = -float(result.fun)
        self.n_matches = len(history)
        self.effective_n = float(weights.sum())
        self.converged = bool(result.success)

        return self

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Numarul mediu de goluri asteptat de la fiecare echipa."""
        for team in (home_team, away_team):
            if team not in self.attack:
                raise KeyError(f"Echipa necunoscuta: {team!r}")

        lam = np.exp(self.attack[home_team] - self.defence[away_team] + self.home_advantage)
        mu = np.exp(self.attack[away_team] - self.defence[home_team])
        return float(lam), float(mu)

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Matricea de probabilitati peste toate scorurile pana la max_goals."""
        lam, mu = self.expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1, dtype=float)

        home_probs = np.exp(_poisson_logpmf(goals, np.full_like(goals, np.log(lam))))
        away_probs = np.exp(_poisson_logpmf(goals, np.full_like(goals, np.log(mu))))
        matrix = np.outer(home_probs, away_probs)

        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho

        return matrix / matrix.sum()

    def predict(self, home_team: str, away_team: str) -> MatchPrediction:
        """Predictie completa: 1X2, goluri asteptate, scor probabil, pariuri derivate."""
        matrix = self.score_matrix(home_team, away_team)
        lam, mu = self.expected_goals(home_team, away_team)

        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())

        flat_index = int(matrix.argmax())
        best_home, best_away = np.unravel_index(flat_index, matrix.shape)

        totals = np.add.outer(np.arange(self.max_goals + 1), np.arange(self.max_goals + 1))
        over_2_5 = float(matrix[totals > 2.5].sum())
        both_score = float(matrix[1:, 1:].sum())

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            expected_home_goals=lam,
            expected_away_goals=mu,
            most_likely_score=(int(best_home), int(best_away)),
            most_likely_score_prob=float(matrix[best_home, best_away]),
            over_2_5=over_2_5,
            both_teams_score=both_score,
        )

    def ratings(self) -> pd.DataFrame:
        """Fortele estimate, ordonate dupa calitatea generala."""
        table = pd.DataFrame(
            {
                "team": self.teams,
                "attack": [self.attack[team] for team in self.teams],
                "defence": [self.defence[team] for team in self.teams],
            }
        )
        table["overall"] = table["attack"] + table["defence"]
        return table.sort_values("overall", ascending=False).reset_index(drop=True)


def main() -> None:
    from db.queries import load_matches

    parser = argparse.ArgumentParser(description="Antreneaza Dixon-Coles pe La Liga.")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI, help="Rata de uitare pe zi.")
    parser.add_argument("--home", default="Barcelona")
    parser.add_argument("--away", default="Real Madrid")
    parser.add_argument("--top", type=int, default=12, help="Cate echipe afiseaza in clasament.")
    args = parser.parse_args()

    matches = load_matches()
    model = DixonColes(xi=args.xi).fit(matches)

    print(f"Meciuri folosite:     {model.n_matches:,}")
    print(f"Meciuri efective:     {model.effective_n:,.0f}  (dupa ponderare temporala)")
    print(f"Echipe:               {len(model.teams)}")
    print(f"Convergenta:          {'da' if model.converged else 'NU'}")
    print(f"Log-likelihood:       {model.log_likelihood:,.1f}")
    print(f"Avantaj teren propriu: {model.home_advantage:+.4f}")
    print(f"Rho (dependenta):      {model.rho:+.4f}")

    print(f"\n{'echipa':<18}{'atac':>9}{'aparare':>10}{'total':>9}")
    print("-" * 46)
    for row in model.ratings().head(args.top).itertuples(index=False):
        print(f"{row.team:<18}{row.attack:>9.3f}{row.defence:>10.3f}{row.overall:>9.3f}")

    print()
    print(model.predict(args.home, args.away))


if __name__ == "__main__":
    main()
