# La Liga Data Platform

An end-to-end data pipeline and forecasting system for the Spanish first division:
three heterogeneous sources integrated into one relational schema, a match-outcome
model validated against the betting market, and an interactive dashboard on top.

Built as a portfolio project. The engineering, not the football, is the point.

---

## What it does

| | |
|---|---|
| **Score prediction** | A full probability distribution over every scoreline, from which 1X2, over/under 2.5 and both-teams-to-score all follow |
| **Form forecast** | Expected points over a team's next *n* fixtures, computed exactly rather than simulated |
| **Player index** | Attacking output per 90 minutes, shrunk toward a position baseline by an amount calibrated on real season-to-season transitions |
| **Top scorer race** | Monte Carlo simulation of the 2026/27 season with parameter uncertainty propagated |

## Results

Walk-forward backtest over **2,658 matches** across seven seasons (2019-20 → 2025-26).
The model is retrained at most weekly and only ever on matches played strictly before the
one being predicted.

| Model | Log-loss | Brier | Accuracy |
|---|---|---|---|
| Baseline (historical H/D/A frequencies) | 1.0706 | 0.6470 | 45.1% |
| **Dixon-Coles** | **0.9894** | **0.5885** | 52.1% |
| Closing betting odds | 0.9699 | 0.5765 | 54.0% |

```
(1.0706 - 0.9894) / (1.0706 - 0.9699) = 81%
```

The model closes **81% of the gap** between knowing nothing and the closing market price,
using goals alone — no lineups, no injuries, no transfer news, none of the money that
actually moves the odds. It does not beat the market, which is the expected and correct
result. A model that beat the market would be evidence of a leak, not of skill.

**Calibration.** Across the 10-60% band, where 87% of predictions live, the gap between
predicted and observed frequency stays under 1.5 percentage points. When the model says
25%, it happens 25.3% of the time.

## Data

| Source | Provides | Coverage |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | Historical results, match stats, closing odds | 7,980 matches, 21 seasons (2005-06 → 2025-26) |
| [football-data.org](https://www.football-data.org/) | Fixture list, current-season scorers | 380 fixtures (2026/27), 287 scorers |
| FBref, via [worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data) | Player season stats: minutes, goals, xG, xAG | 9,501 player-seasons, 17 seasons, 3,081 players |

**Attribution.** Player statistics originate from [FBref](https://fbref.com/) and were
collected and published by Jason Zivkovic in the `worldfootballR_data` archive. That
repository is archived, so the last complete season it covers is 2024-25.

Understat and FBref both block programmatic access — Understat disallows all crawling in
`robots.txt`, FBref sits behind bot protection. Neither was scraped; the public archive
was used instead.

---

## Engineering

### Idempotent loading

Every loader looks a record up by its **natural business key** before writing. Running
the full ingestion twice produces an identical database:

```
$ python -m ingestion.football_data_uk
Gata: 21 sezoane | 7980 meciuri inserate | 0 actualizate

$ python -m ingestion.football_data_uk
Gata: 21 sezoane | 0 meciuri inserate | 7980 actualizate
```

For matches the key is `(season, home_team, away_team)` — in a double round-robin every
pair meets exactly once at each ground. **The date is deliberately excluded.** Matches get
rescheduled, and with the date in the key a postponed fixture would insert a second row
instead of updating the first.

### Schema drift

The source CSVs change shape across 21 years: shot and card columns appear in 2005-06,
market-average odds in 2019-20, closing odds later still. The parser maps columns by name
and yields `None` for anything absent, rather than assuming a fixed schema. A pipeline
that breaks because a column from 2007 is missing is a pipeline that will break in
production.

### Entity resolution

The two match sources share **no common team identifier and not a single identical name**.
The API says `Rayo Vallecano de Madrid`, the CSV archive says `Vallecano`. Resolution runs
through an explicit mapping layer that **raises on anything unmapped**:

```python
raise UnknownTeamError(
    f"Echipa {name!r} nu are mapare canonica. Adaug-o in ingestion/team_names.py."
)
```

Failing loudly is the point. An unmapped team means missing matches, and a pipeline that
drops them silently is more dangerous than one that stops.

### Data quality as an assertion

`pipeline/quality_checks.py` and `pipeline/player_checks.py` state what the data *must*
look like — 380 matches and 20 teams per season, minutes never exceeding what appearances
allow, penalties scored never exceeding penalties taken, xG present only from 2017-18
onward, and so on.

They earned their place twice:

- A **field parse silently failing on 88% of rows.** FBref writes player age as `'31'` in
  most seasons but `'25-081'` in two of them. The parser handled only the second format
  and returned `None` for the rest — no error, no crash, just a mostly empty column. The
  fix was not to patch the parser but to stop depending on it: age is now derived from
  birth year, which is consistent across all 17 seasons.
- **Two alerts that turned out to be correct data and incorrect checks.** La Liga fielded
  two different players named Adrián López at Deportivo in 2009-10, born a year apart. The
  database had separated them correctly; the uniqueness check was comparing names instead
  of identities.

---

## Modelling

### Dixon-Coles

Each team carries an attack and a defence rating; goals are Poisson with a home advantage
and a low-score correction. Older matches are down-weighted exponentially.

```python
log_lam = attack[home] - defence[away] + home_advantage
log_mu  = attack[away] - defence[home]
```

Everything else — 1X2, over/under, both-teams-to-score, exact scoreline — is read off the
resulting 11×11 probability matrix. A classifier trained on 1X2 labels cannot do this;
it would need separate training for each market.

**The decay rate was chosen by backtesting, not by literature.** The sweep produces a clean
bias-variance curve:

| xi | Log-loss | Effective sample |
|---|---|---|
| 0 (no decay) | 0.9928 | 7,980 matches |
| 0.0005 | 0.9902 | 2,087 |
| **0.001** | **0.9894** | 1,090 |
| 0.0018 | 0.9905 | 626 |
| 0.003 | 0.9941 | 393 |
| 0.005 | 1.0014 | 252 |

Note that **accuracy picks a different value than log-loss** — 52.4% at `xi=0.0018` against
52.1% at `xi=0.001`. Tuning on accuracy would have selected a worse-calibrated model. Brier
score agrees with log-loss throughout.

### Shrinkage, calibrated rather than assumed

A player with 200 minutes and two goals has a spectacular and meaningless rate per 90. The
usual fix is an arbitrary minutes threshold, which throws information away. Here each rate
is pulled toward its position baseline, and **how far it is pulled is the value that best
predicts the following season**, measured over 2,328 real season-to-season transitions.

| Position | Transitions | Mean npxG+xAG/90 | Shrinkage constant (90s) |
|---|---|---|---|
| GK | 179 | 0.003 | **69.1** |
| DF | 875 | 0.095 | 9.8 |
| MF | 768 | 0.201 | 7.3 |
| FW | 506 | 0.435 | 10.6 |

The goalkeeper constant is the model **refusing to rank goalkeepers** on an attacking
metric: their variation is pure noise, so shrinkage collapses them all to the mean. That is
the correct answer, and it arrives on its own rather than by special-casing.

### Promoted teams

Three of the twenty clubs in 2026/27 last played in the top flight 8 to 14 years ago —
108 of the 380 fixtures. The model technically has parameters for them, but at near-zero
weight those parameters never move from their starting value, which would read as *exactly
league average*. That is worse than having no rating at all.

Teams whose effective sample falls below a threshold are detected and assigned a prior
measured from **60 promotions across 20 seasons**: a newly promoted side scores 0.83× and
concedes 1.18× the league average.

---

## Negative results

Two findings that did not work out. They are here because leaving them out would make the
project look better and be less true.

### The Dixon-Coles correction is worth almost nothing here

The famous low-score adjustment, ablated over the full backtest:

| | Log-loss |
|---|---|
| With the rho correction | 0.9894 |
| Without (plain Poisson) | 0.9902 |

**0.0008** — about 1% of the model's total edge over the baseline. Four lines of code, kept
because they are cheap and directionally correct, but not a headline.

### The age curve is correct and useless

Estimated by the delta method, comparing each player against himself from one season to the
next rather than averaging output by age — which would be distorted by the fact that only
very good players are still on the pitch at 35. The result peaks at **26**, matching the
published literature, over 4,223 season-to-season transitions.

Applying it makes predictions worse, in **every** age band tested:

| Age band | Shrunk only | Shrunk + age | Δ |
|---|---|---|---|
| 18-23 | 0.1584 | 0.1592 | −0.0009 |
| 24-27 | 0.1420 | 0.1435 | −0.0015 |
| 28-31 | 0.1540 | 0.1545 | −0.0005 |
| 32-40 | 0.1325 | 0.1329 | −0.0004 |

Shrinkage already absorbs most of the ageing signal: a declining player produces less, and
regression to the mean pulls him down regardless. What is left is smaller than the noise.
The delta method is also optimistically biased, since players who decline and leave the
league are never measured. The adjustment ships **disabled by default**.

A related result worth keeping in view:

| Predictor of next season's rate | RMSE |
|---|---|
| Position average (ignore the player entirely) | 0.1781 |
| Last season's raw rate | 0.1768 |
| **Last season's rate, shrunk** | **0.1499** |

A player's raw output last season is barely more informative than ignoring him. Shrinkage
is what turns it into a signal.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Five pages: match prediction with a full scoreline heatmap, expected form with a per-fixture
breakdown, the player index, the top-scorer race, and the age curve.

Design notes: the scoreline heatmap uses a single-hue sequential ramp, never a rainbow.
Bars carry one colour — shading them by magnitude would encode bar length twice and burn
the only free channel. Win/draw/loss uses a diverging pair with a neutral midpoint, because
those three outcomes have polarity rather than being arbitrary categories. **Every chart has
a table view behind it**, so no value is reachable only through colour or only through hover.

The age-curve page states plainly that the model it displays does not improve predictions.

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

cp .env.example .env               # add a free football-data.org token
```

```bash
python -m ingestion.football_data_uk              # 21 seasons of results
python -m ingestion.football_data_org             # 2026/27 fixtures
python -m ingestion.football_data_org --scorers 2025
python -m ingestion.fbref_players                 # player season statistics

python -m pipeline.quality_checks                 # assert the data is sane
python -m pipeline.player_checks
```

```bash
python -m models.dixon_coles --home Barcelona --away "Real Madrid"
python -m models.backtest --xi 0.0005 0.001 0.0018 --calibration
python -m models.form_forecast --team Barcelona
python -m models.player_index --by-position
python -m models.player_forecast
python -m models.scoring_race --sims 20000
```

Storage runs through SQLAlchemy behind a single `DB_URL`, so moving from SQLite to
PostgreSQL means changing one line in `.env` and nothing else.

## Layout

```
config.py                  Paths, database URL, source endpoints
db/
  models.py                SQLAlchemy schema
  session.py               Engine and transactional sessions
  queries.py               Shared reads
ingestion/
  football_data_uk.py      Historical results (CSV archives)
  football_data_org.py     Fixtures and scorers (REST API)
  fbref_players.py         Player season statistics
  team_names.py            Cross-source entity resolution
pipeline/
  quality_checks.py        Match-level invariants
  player_checks.py         Player-level invariants
models/
  dixon_coles.py           Scoreline model
  backtest.py              Walk-forward evaluation
  promoted_prior.py        Prior for newly promoted teams
  form_forecast.py         Expected points over upcoming fixtures
  player_index.py          Shrunk performance index
  player_forecast.py       Age curve and next-season projection
  scoring_race.py          Top-scorer simulation
dashboard/
  app.py                   Shell and navigation
  data.py                  Cached loaders
  theme.py                 Palette and chart templates
  views/                   One module per page
```

## Known limitations

- **Squads are those of May 2026.** No free source publishes 2026/27 rosters before the
  season starts, so summer transfers are not modelled.
- **Promoted teams have no players in the data**, so their goals go unallocated in the
  top-scorer simulation. A surprise top scorer from Málaga is impossible in this model.
- **Appearances are held at last season's level.** Injuries and rotation are not modelled.
- **The promoted-team prior treats every promoted side identically.** Its standard
  deviation is as large as its mean — Girona came up in 2022-23 and finished third the year
  after. Fitting the second division jointly would fix this and is the obvious next step.
- **Player data ends at 2024-25** for the FBref-sourced statistics.
