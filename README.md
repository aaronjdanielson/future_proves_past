<p align="center">
  <img src="assets/fig_case_distributions_dark.png" alt="What the models expected, and what happened: six international freshmen from 2025-26" width="880">
</p>

<h1 align="center">Future Proves Past</h1>
<p align="center"><b>Learn backward. Predict forward.</b><br>
Forecasting the first NCAA season of international recruits from their pre-college games,<br>
trained with the help of 10,000+ former NCAA players who later played abroad.</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-194%20passing-brightgreen" alt="194 tests">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/player--games-6.3M-8A2BE2" alt="6.3M player-games">
  <a href="#replicate-the-abstract"><img src="https://img.shields.io/badge/replication-data%20%2B%20models-0F8A6B" alt="replication: data + models"></a>
  <img src="https://img.shields.io/badge/2026--27%20forecasts-282%20freshmen-orange" alt="282 freshmen forecast">
</p>

<p align="center">The data and saved models behind the SSAC27 abstract, with step-by-step replication instructions, are under <a href="#replicate-the-abstract">Replicate the abstract</a>.</p>

---

## The idea in thirty seconds

College programs commit roster spots and money to international prospects before those players have played a single NCAA game. The data that would tell you how a Serbian junior league or the U18 EuroBasket translates to the Big 12 is thin: a few hundred players a decade ago, 113 experienced entrants in 2025–26.

But the *reverse* transition is huge. More than ten thousand former NCAA players went on to play in international leagues and for national teams, and every one of them is measured on both sides of the same gap: we know their college seasons, and we know what they did abroad.

So we train one network on two jobs. **Forward:** pre-college games → the coming NCAA season. **Backward:** post-college games → the college season we already know. The second job is only there to teach the first.

```mermaid
flowchart LR
  subgraph B["Learn backward (auxiliary task)"]
    direction LR
    L["Later international and<br/>national-team games"] --> N1[("Shared game tower")] --> K["Known NCAA seasons"]
  end
  subgraph F["Predict forward (the forecast)"]
    direction LR
    R["Recruit's earlier games"] --> N2[("Same parameters")] --> S["Freshman season<br/>distribution"]
  end
  N1 -. same weights .- N2
```

A worked example from the paper. A forward example takes a prospect's games at about age 19 and predicts the NCAA season that opens when he is 20: the history sits one year *before* the target. A reconstruction example takes a former player's games at about age 23 and reconstructs that same season at 20: the history sits three years *after* it. Age and signed time tell the network which job it is doing; the reconstruction example needs no pre-college history at all, which is why the large population of former players can teach it. The paper's three questions follow directly:

1. Does including post-NCAA international and national-team game histories improve forward freshman forecasts through shared parameters?
2. Which targets benefit: playing opportunity, production rates, or shooting efficiency?
3. Do any gains extend to freshmen with no recorded international or national-team experience?

The picture at the top shows what the answer to the first question buys. Red is a control model trained the ordinary way; purple is the same architecture trained with the former players' later careers; the gold line is what actually happened. Tomislav Buljan's season at New Mexico was twenty million times more probable under the purple model.

## What it predicts

Not a number. A **whole season as a probability distribution**: whether the player appears at all, games and starts, minutes, and thirteen box-score counts (field goals by zone, free throws, rebounds, assists, steals, blocks, turnovers, fouls), everything coupled through an explicit opportunity-before-production structure. From draws of that distribution we read off anything a scout asks for: P(400+ minutes), minutes per game *if he plays*, PER, usage, true shooting, per-40 rates.

Inputs are every tracked game before the season's cutoff (1 October): 6.3 million player-games from 151 international club competitions, 78 national-team competitions and the main US showcases, each game encoded with its box score, opponent strength, competition, age group, the player's age that day and a handful of clocks; plus static context (bio, recruiting rank, destination team).

## How it works

<p align="center">
  <img src="assets/game_tower_reference.png" alt="Game history to season forecast: the reference game tower" width="820">
</p>

The reference tower does three things. It **encodes each game** into a vector with one small shared network. It **summarises the history** with fixed weights, an exposure-weighted long-run mean and a recency-weighted recent mean, plus the contrasts between them, so established performance and recent form are both visible, and attaches evidence about how much history there is. It **combines** that summary with the destination context, the player's age and the signed time to the target, and maps the result to the parameters of the season distribution. International games and any prior NCAA games get separate summaries. There is no learned attention in the registered model (it is an option under test), the tower has 57,363 parameters, and a game that happened after the target is simply never shown to a forward example.

The backward task updates exactly the parts a forward forecast uses: the game encoder, the fusion networks, the translator and the outcome heads. That shared path is what the experiments measure. Dates do the rest. A forward example sees only games dated before the season and released by its cutoff; a reconstruction example sees only games dated after that season ended. Nothing fitted, not even the feature standardisation, may use a record released after the training cutoff.

## Does learning backward help?

<p align="center">
  <img src="assets/fig_learning_and_evidence_dark.png" alt="Learning and evaluation: geometric mean probability ratios, later-career training versus control" width="720">
</p>

Every comparison is paired on the same players, scored by the log probability the model assigns to the season that happened, with a **strict control** that never sees a game dated after a player's first college season. Rolling folds, dated cutoffs, one registered one-shot test.

| Freshmen with international or national-team history | Gain, nats per player (95% CI) | Share improved |
|---|---|---|
| 2021–22 (n 105) | **+1.75** [+0.85, +2.64] | 67% |
| 2022–23, seed 1 (n 124) | **+0.84** [+0.18, +1.50] | 56% |
| 2022–23, seed 2 (n 124) | **+0.86** [+0.31, +1.41] | 64% |
| 2025–26, registered test, two fits (n 307) | +0.26 [−0.12, +0.64], **+0.52** [+0.16, +0.88] | 47% |

A gain of +0.84 nats means the realized seasons were, on average, e^0.84 ≈ 2.3× more probable under the model that learned backward.

### Things the abstract had no room for

- **The gain is a breakout detector.** Per-player gains are heavy-tailed: the ten largest cases carry 99% of the summed gain in 2022–23 (mean +0.82 → +0.01 without them). The median player gains a modest 1.2–1.6× in probability. What the auxiliary data buys is the ability to take seriously the international freshman who becomes a real contributor, which the control treats as near-impossible.
- **Where the gain lives.** Almost all of it is *production given playing time*. Shooting percentages: no gain. Minutes and starts: no gain. Freshmen without any international or national-team history: no gain (−0.05 to +0.06). The careers abroad teach how production translates, not who the coach will play.
- **Against the recruiting industry.** On the registered 2025–26 test the full model's rank correlation with realized minutes was 0.40–0.43 against 0.23 for the 247 composite, with points 0.43–0.46 against 0.28, and its AUC for identifying 400-minute contributors 0.69–0.72 against 0.61.
- **How much more auxiliary signal is there?** Tripling the weight of the backward task changed nothing (−0.03 [−0.32, +0.26]). Re-weighting training toward first seasons helps the big no-history cohort (+0.05 to +0.23 nats) and is neutral for the history cohorts. The next gains will come from architecture and data coverage, not from the loss.
- **The ranking power is the system's; the paired gain is the data's.** The strict control also beats the 247 composite at ranking (rank correlation with minutes 0.37, with points 0.41–0.42, AUC 0.68–0.69). What the former players' careers add shows up in the probability assigned to whole seasons, not in a reshuffled ranking.
- **Every college season is a teacher.** Restricting the backward task to freshman seasons alone shrinks the gains to +0.21, +0.12 and −0.09 nats. Later college seasons supply three times as many reconstruction labels (26,029 against 7,849 on one fold) and sit closer in time to the professional careers.
- **Careers are long.** The median post-college game lies 7.6 years after the player's first NCAA season; a four-year window after college would discard 89% of them. The model uses every later season available at the training cutoff.
- **The translation is real on unseen players.** On 793 former players held out entirely, predicting their known college seasons from their later careers alone beats an intercept-only model by 7.4 nats per season, 4.5 on freshman seasons.

<p align="center">
  <img src="assets/fig_breakout_ladder_dark.png" alt="Every international entrant of the 2025-26 test season, sorted by probability ratio" width="820">
</p>

**Hall of breakouts, 2025–26** (how many times more probable the full model found the actual season): Tomislav Buljan, New Mexico, ×20,288,508 · Ilias Kamardine, Ole Miss, ×157,737 · Tim Rudovskii, Bryant, ×20,982 · Dylan Ducommun, Northern Illinois, ×8,189 · Thijs De Ridder, Virginia, ×5,162 · Filip Brankovic, Texas-RGV, ×3,507. The ladder above is every international entrant of that season, not just the winners: 47% improved, median ×0.94.

## Where freshmen come from, and how production travels

<p align="center">
  <img src="assets/fig_league_translation_dark.png" alt="League translation: forecast first-season PER by league of origin" width="820">
</p>

The model's ranking of leagues by forecast first-season PER tracks an independent measure of league strength, the cross-league RAPM calibration (rank correlation 0.60). Realized PER by league is noisy at ten players a league. A second check uses the former players directly: the production ratio each league implies from the players who *left* college for it, against the ratio the model *applies* to players arriving from it, agree with rank correlation 0.79 across the seven best-covered leagues.

There is a formal version of this. Projecting the fitted model onto standardized player profiles over a declared reference population yields a league-to-league translation rule for each statistic, an exchange rate from source production to implied NCAA production. Under a multiplicative form those rates are league difficulties, so for a composite such as PER they order the international leagues on one NCAA scale. The ordering is model-implied and specific to a target, a population and a horizon, not a causal league effect; two research notes accompanying the paper set out the mathematics and the test of whether the later careers make the rules more reproducible.

## What it does not claim

- **Reverse history is not a new idea; the controlled test is the contribution.** Translating performance across leagues has a literature (Glazer's G League-to-NBA translation factors by matching and difference-in-differences; Penner's 22,500 players across 110 leagues; Held's freshman-impact poster), and practitioners already publish international recruit models: Kalinowski's Recruit Points, The Resource Nexus translation model, and QuantCat's description of comparing a prospect with former NCAA players of similar overseas dominance and consulting their college outcomes. Those are method descriptions, not validated benchmarks. What this project adds is a shared-parameter mechanism tested under a strict control, dated cutoffs and a registered one-shot season, with the gain decomposed by target and cohort.
- **Later careers alone cannot identify the forward mapping.** Professionals and incoming recruits differ in age, selection, role and competition. Signed time and age let the model represent those differences; they do not erase them, which is why forward labels and forward validation stay essential and why a zero-weight comparator is always run.
- **Opportunity is the hard part.** A freshman's minutes depend on a roster that is only partly known before arrival. Prior team ratings and returning-production shares describe the destination; they do not allocate its minutes. That is why the exhibits show efficiencies, and why minutes forecasts carry wide intervals.
- **A probabilistic output is not automatically calibrated.** Interval coverage and the calibration of the rotation probabilities are empirical checks, still being run, not consequences of the model's form.
- **Coverage drifts.** Date of birth is missing for 27% of eligible player-seasons in 2017–18 and 72% in 2025–26 (international entrants: at most 3% in any season); starter flags on international games went from 56% to 100% coverage in the current data build. Masks and evidence features carry the known share, but a mask does not remove the shift.

## Forecasts for 2026–27

The prospective cohort is every 2026–27 freshman in his first NCAA roster season for whom the model has at least one tracked pre-college game: **282 players**, 220 of them international. By history: 141 club and national team, 47 club only, 69 national team only, 25 US showcase only. Median tower: 28 games; a quarter have 74 or more.

All 282 were forecast on 1 October 2026 by the deployment models retrained that morning on the v11 data build: trained through 2024–25 plus half of the 2025–26 players and stopped on the other half, the two halves pooled, 2,000 simulated seasons per player, once with and once without the former players' careers in training. Minutes per game are conditional on appearing; PER is over seasons with 100+ minutes. Treat them as a commitment, not a validation: the outcomes arrive in March 2027. The frozen files carry the run names and data digests.

**Most anticipated** (deployment models v11, frozen 2026-10-01, two cross-stopped halves pooled; by probability of a 400-minute rotation season):

| Player | Team | History | Tracked games | 247 | P(400+ min) with / without | Min/game if plays | PER with / without |
|---|---|---|---|---|---|---|---|
| Caleb Holt | Arizona | national only | 22 | 4 | 1.00 / 0.98 | 30.1 | 21.8 / 19.2 |
| Tyran Stokes | Kansas | national only | 23 | 1 | 0.99 / 0.99 | 32.3 | 24.4 / 23.5 |
| Jordan Smith, Jr. | Arkansas | national only | 16 | 3 | 0.99 / 0.98 | 30.2 | 19.8 / 18.9 |
| Brandon McCoy, Jr. | Michigan | national only | 22 | 14 | 0.98 / 0.99 | 27.5 | 18.8 / 17.3 |
| Christian Collins | USC | showcase only | 3 | 6 | 0.97 / 0.97 | 29.4 | 20.8 / 19.5 |
| J.J. Andrews | Arkansas | showcase only | 2 | 16 | 0.95 / 0.92 | 26.2 | 19.1 / 17.7 |
| Bruce Branch III | Brigham Young | national only | 8 | 8 | 0.95 / 0.87 | 25.8 | 20.2 / 18.5 |
| Aurele Brena-Chemille | San Jose State | club + national | 177 | — | 0.94 / 0.32 | 26.9 | 15.1 / 11.2 |
| Taylen Kinney | Kansas | club + national | 8 | 18 | 0.93 / 0.91 | 24.7 | 12.8 / 12.2 |
| Caleb Gaskins | Miami (FL) | national only | 6 | 13 | 0.91 / 0.76 | 24.6 | 17.5 / 15.0 |
| Anthony Thompson | Ohio State | showcase only | 1 | 9 | 0.91 / 0.90 | 26.2 | 18.8 / 17.9 |
| Marcus Spears, Jr. | Texas | national only | 6 | 7 | 0.88 / 0.79 | 24.0 | 21.3 / 19.8 |

**Where the former players' careers change the forecast most.** Across the cohort the two models agree about the typical player (mean P(400+ min) 0.275 with the careers against 0.267 without; international group mean PER 11.0 against 10.7). They disagree sharply about individuals, and the disagreements cluster where the backward task has the most to say: players with long club towers heading to mid-majors.

| Player | Team | History | Tracked games | P(400+ min) with / without | PER with / without |
|---|---|---|---|---|---|
| Talis Soulhac | Murray State | club + national | 153 | 0.73 / 0.02 | 13.3 / 7.1 |
| Siebe Ledegen | Ohio | club + national | 206 | 0.83 / 0.14 | 13.8 / 8.7 |
| Aurele Brena-Chemille | San Jose State | club + national | 177 | 0.94 / 0.32 | 15.1 / 11.2 |
| Cameron Williams | Duke | showcase only | 3 | 0.74 / 0.12 | 21.1 / 15.2 |
| Saliou Niang | LSU | club + national | 163 | 0.79 / 0.27 | 15.5 / 12.6 |
| Yoav Vitlem | San Francisco | club + national | 74 | 0.70 / 0.19 | 10.5 / 7.3 |
| Nemanja Popovic | Memphis | club + national | 175 | 0.82 / 0.32 | 16.3 / 13.6 |
| *and in the other direction* | | | | | |
| Kajus Mikalauskas | Texas-Arlington | club + national | 120 | 0.08 / 0.78 | 7.4 / 9.1 |
| Armandas Bancevicius | Tennessee-Martin | club + national | 126 | 0.16 / 0.79 | 11.5 / 14.6 |
| Gabrielius Jokubauskas | Northern Kentucky | club + national | 100 | 0.14 / 0.70 | 10.0 / 13.3 |
| Jamal George | Oral Roberts | club + national | 68 | 0.26 / 0.81 | 9.5 / 10.7 |
| Ogbemudia Uagboe | Cleveland State | club + national | 44 | 0.24 / 0.77 | 9.0 / 11.0 |

The full table of all 282 freshmen, with the same columns, is in the frozen forecast files `FORECASTS_2026_27_deploy_v11_pooled.md` (with the careers) and `FORECASTS_2026_27_deploy_control_v11_pooled.md` (without), both in `results_docs.zip` of the release described under [Replicate the abstract](#replicate-the-abstract); the cohort list is [`data/forecast_2027_cohort.csv`](data/forecast_2027_cohort.csv).

## How the experiment is run

- **Prediction unit.** A player's season with one team. Target: the season as drawn above; the schedule length is a labelled condition, never an input.
- **Rolling folds.** Fold *v* trains on seasons through *v*−2, chooses its stopping epoch on *v*−1, and scores *v* once. Development 2019–20 to 2022–23; selection 2023–24 and 2024–25; **registered one-shot test 2025–26**; deployment 2026–27.
- **Cross-stopping for deployment.** The last labelled season is split into two fixed halves; one model trains on everything plus half 1 and stops on half 0, the second the reverse; the forecast is their mixture. The test season enters nothing.
- **Arms.** `A` forward only; `AR` forward plus the backward task on the former players' careers; the strict control `A⁻` additionally never sees a game dated after a player's first college season, in any input, pool or window.
- **Scoring.** Marginal log probability of the observed season, overtime summed out, verified against a quadrature with twice the nodes; paired differences with player-level standard errors; every comparison on identical units.
- **Reproducibility.** Write-once run manifests with the training curve, SHA-256 of the tables and the test registration; the one-shot test was registered before it was scored.

The reference tower has 57,363 parameters and trains in about fourteen minutes an epoch on a laptop CPU; it stops after ten to twenty epochs.

## What is in this repository

| Path | What |
|---|---|
| `fpp/data` | Dated data contracts: calendars and cutoffs, immutable game records, nested reconstruction windows, routing of games and summaries, rolling folds, the table assembler over the upstream stores |
| `fpp/model` | The season distribution and its observed-data likelihood (NumPy/SciPy reference and a differentiable PyTorch twin), feature construction, the game tower, pooling, heads |
| `fpp/experiments` | Training and scoring of one arm on one fold, fold reports, intercept-only recovery |
| `tests/` | 194 tests: likelihood identities, cutoff and leakage guards, strict-control masks, weighting, cross-stopping, every architecture option reproducing the registered model by default |
| `config/protocol.json` | The frozen experimental protocol: calendar, reconstruction horizon, fold layout, numerical tolerances |
| `assets/` | The figures shown here |

Not in the repository: the upstream game databases (built from public box scores), the paper source, and the local run scripts and decision log. The assembled tables those databases produce, and the saved models, are attached to a release (see [Replicate the abstract](#replicate-the-abstract)); frozen copies of the scoring scripts are in `replication/`.

## Run it

```sh
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[model,data,dev]"
python -m pytest tests -q                      # 194 tests, about two minutes
fpp smoke --out artifacts/smoke.json --draws 200 --score-first 3     # synthetic seasons, no data needed
fpp gradcheck --out artifacts/gradcheck.json                          # differentiable likelihood vs the reference
```

With the upstream stores available (read-only; the assembler never writes to them):

```sh
fpp build-tables --stores stores.json --out cache/tables_v11
fpp train --tables cache/tables_v11 --fold 2023 --arm AR --lambda-plus 1 --seed 20260901 \
          --out cache/runs/ARfull_2023 --recon-horizon-years none --recon-k 1,2,3,all
fpp train --tables cache/tables_v11 --fold 2023 --arm A --exclude-post-first-season-logs \
          --seed 20260901 --out cache/runs/control_2023
fpp report-run --run cache/runs/ARfull_2023 --compare cache/runs/control_2023 --tables cache/tables_v11
```

Useful flags: `--cross-stop 0|1`, `--weighting season_balanced --first-year-share 0.5`, `--feature-schema v9`, and the architecture options `--width`, `--hidden`, `--direction-film`, `--attention-pool`, `--head-hidden`, `--game-dropout`, `--average-last`, `--lr-decay`, `--future-weight`. Defaults reproduce the registered model exactly.

## Replicate the abstract

Everything behind the SSAC27 abstract runs from flat files; no database is needed. The data and the saved models are attached to the GitHub release `ssac27-abstract` as four zip archives (about 455 MB in all), with SHA-256 sums in [`replication/SHA256SUMS`](replication/SHA256SUMS):

| Archive | Contents |
|---|---|
| `tables_v7.zip` | The assembled tables behind every number in the abstract: `games.parquet` (6,317,405 player-game lines, 73 columns), `units.parquet` (124,618 player-seasons, 82 columns), competition periods and the manifest whose hashes the code verifies. |
| `tables_v11.zip` | The deployment tables behind the 2026–27 forecasts above (6,561,231 lines, 97 columns, plus player, league and destination tables). |
| `runs_registered.zip` | The ten registered run directories (folds 2021–22, 2022–23 ×2 seeds, 2025–26 ×2 seeds; full model and strict control each) with checkpoints, per-unit scores and per-player prediction tables, and the four 2026–27 deployment runs. |
| `results_docs.zip` | The result files the abstract cites, with their per-unit CSVs: `FIRST_RESULT_*`, `TEST_RESULTS_2026`, `TEST_IDENTIFICATION_2026_*`, `TEST_REGISTRATION_2026`, `CRPS_*`, `FORECASTS_2026_27_*`. |

The tables hold per-game box-score lines and the player fields published on public sites (name, date of birth, nationality, height, weight, position); nothing private. `replication/scripts/` holds the scoring scripts frozen as of the abstract; `replication/data/` holds the NCAA team totals and PER constants those scripts previously read from a database; `replication/abstract/` holds the abstract source and its figure scripts.

```sh
# 0. Install (see Run it), then fetch, verify and unpack the archives
mkdir -p cache out
for f in tables_v7 tables_v11 runs_registered results_docs; do
  curl -L -o cache/$f.zip https://github.com/aaronjdanielson/future_proves_past/releases/download/ssac27-abstract/$f.zip
done
(cd cache && shasum -a 256 -c ../replication/SHA256SUMS && unzip -q tables_v7.zip && unzip -q tables_v11.zip \
   && unzip -q runs_registered.zip && unzip -q -o results_docs.zip -d ../docs)

# 1. Paired log scores from the saved checkpoints: exact, a few minutes per pair, no training
python3 replication/scripts/first_result.py --tables cache/tables_v7 \
    --with cache/runs/bc_ARfull_l1_2023_20260901 --without cache/runs/bc_Aexcl_l0_2023_20260901 \
    --label "full model vs strict control, 2022-23, fit 1" --out out/FIRST_RESULT_2023_s1.md

# 2. CRPS by statistic and the test season's log score (300 season draws per run and player; about five minutes)
python3 replication/scripts/crps_forecasts.py --tables cache/tables_v7 \
    --with cache/runs/bc_ARfull_l1_2026_20260902 --without cache/runs/bc_Aexcl_l0_2026_20260902 \
    --cohort primary --n-draws 300 --label "2025-26 test, fit 2" --out out/CRPS_2026_s2.md

# 3. Ranking the test freshmen against the 247Sports composite
python3 replication/scripts/identification.py --tables cache/tables_v7 \
    --predictions cache/runs/bc_ARfull_l1_2026_20260901/per_player_predictions_vs_Aexcl.csv \
    --label "fit 1" --out out/TEST_IDENTIFICATION_s1.md

# 4. Retrain a pair from scratch with the registered recipe (CPU, hours per run); fold 2026 adds
#    --test-registration docs/TEST_REGISTRATION_2026.md
fpp train --tables cache/tables_v7 --fold 2023 --arm AR --lambda-plus 1 --seed 20260901 \
          --recon-horizon-years none --recon-k 1,2,3,all --out cache/runs/my_ARfull_2023
fpp train --tables cache/tables_v7 --fold 2023 --arm A --exclude-post-first-season-logs \
          --seed 20260901 --out cache/runs/my_control_2023

# 5. The abstract's evidence figure, from the per-unit CRPS files in docs/
(cd replication/abstract && python3 make_evidence_crps_figure.py --docs ../../docs --show-training)
```

What to expect (the registered values; Monte Carlo steps reproduce to the precision of their draws):

| Quantity | Value | File in `results_docs.zip` |
|---|---|---|
| 2021–22, joint log-score gain, freshmen with history (n 105) | +1.747 [+0.849, +2.644] nats, 5.75× | `FIRST_RESULT_ARfull_vs_Aexcl_2022.md` |
| 2022–23, fit 1 (n 124) | +0.842 [+0.182, +1.502], 2.32× | `FIRST_RESULT_ARfull_vs_Aexcl_2023.md` |
| 2022–23, fit 2 | +0.862 [+0.310, +1.415], 2.36× | `FIRST_RESULT_ARfull_vs_Aexcl_2023_s2.md` |
| 2025–26 test, fits 1 and 2, primary cohort (n 307) | +0.26 [−0.12, +0.64] and +0.52 [+0.16, +0.88] | `TEST_RESULTS_2026.md`, `CRPS_TEST_2026_s*_primary.md` |
| 2025–26 test, CRPS skill, fit 2 | minutes +9.4%, points +7.0%, rebounds +8.5%, assists +9.3% | `CRPS_TEST_2026_s2_primary.md` |
| 2025–26 test, Spearman with season minutes / AUC for 400 minutes | 0.40–0.43 / 0.69–0.72 (composite 0.23 / 0.61) | `TEST_IDENTIFICATION_2026_s*.md`, `TEST_RESULTS_2026.md` |
| Freshmen without such history | −0.08 and −0.09 nats on the held-out seasons; unchanged on the test | the same `FIRST_RESULT_*` and `TEST_RESULTS_2026.md` |

The figure's rows average the two fits of a season player by player (`replication/abstract/figures/evidence_crps.csv` records every value plotted). The league and translation figures of the paper are not covered: they read the international RAPM and league databases.

<!-- ## Status

- **Paper:** *Future Proves Past: Using Post-NCAA Careers Abroad to Forecast Incoming International Recruits*, in revision.
- **MIT Sloan Sports Analytics Conference 2027:** research paper competition abstract, October 2026.
- **2026–27:** deployment forecasts for the 282-player cohort, dated and frozen; scored when the season ends.
- **Next:** cross-stopping and architecture ladders (attention pooling, two-layer heads, learning-rate decay), the v9 feature blocks (prior league, usage, destination context, international RAPM), youth-league coverage. -->

## People

Aaron John Danielson (Department of Computer Science, The University of Texas at Austin), Denis Beausoleil, Hasan Alanam.
Contact: aaron.danielson@austin.utexas.edu
