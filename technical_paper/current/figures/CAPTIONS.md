# Figure captions (the words kept off the canvases)

All figures are produced by `scripts/make_figures.py` from saved outputs (`scripts/per_unit_forecasts.py` for the model-side
inputs, `scripts/figures_post.sh` for the full refresh). Each exists in a light version for the paper and a `_dark` version for
slides. Where a figure compares "with" and "without", "with" is the model trained with the post-NCAA careers of former college
players (arm AR-full) and "without" is the strict control, which never sees a game dated after a player's first college season.

**fig_future_proves_past.** Former NCAA players measured twice: NCAA PER in their last college season (which must have at
least 200 minutes) against points, rebounds and assists per 40 minutes in the 730 days after that season's end, in the league
where they played the most minutes (at least 10 games and 200 minutes there). Panels are the eight leagues with the most such players, ordered by
the league's cross-league strength (the RAPM calibration shift of `intl_rapm.db`, EuroLeague = 0); the line is the least-squares
fit and r its correlation. These are the reverse examples the model learns the translation from. Data only; no model.

**fig_breakout_cases.** The eight international freshmen whose actual first college season was most more probable under the
model with the post-NCAA data than under the control (fold 2022–23, first-season weighted models, seeds pooled). Bars are the
90% pre-season predictive intervals of points per game and minutes per game, white ticks their medians; diamonds are the season
the player went on to play; the right margin gives how many times more probable the full model found it. `fig_breakout_cases_mirror`
shows the eight largest cases in the other direction. These cases are selected on the outcome; the ladder figure shows every player.

**fig_case_distributions** ("Largest Predictive Gains"). Selected examples: for named international freshmen of the 2025–26 test season with at least 20
pre-college games and at least 400 minutes, the full pre-season predictive distributions of minutes per game, PER and usage rate under the
control (red) and under the model with the post-NCAA data (purple), from 2,000 season draws per model (two seeds pooled); the
gold line is the season the player went on to play. Distributions condition on playing (minutes per game) and on at least 100 minutes (PER, usage).
The case file also carries assist-rate draws (the rate whose point forecasts gain most from the later careers); the panel is not shown. Points per game is deliberately
not shown: scoring volume depends on the roster around the player, which the model does not condition on; efficiencies are the player's own. The players are the largest cases in the full model's favour and are chosen on the outcome;
the ladder figure shows every player of the season.

**fig_breakout_ladder.** Every international entrant of the scored season, sorted by how many times more probable his actual
first season was under the model with the post-NCAA data than under the control (log scale; blue to the right of 1, red to the
left). The five largest cases are named with their ratio. The title line gives the share of players improved and the median
ratio; the mean gain in nats is in the first-result documents.

**fig_forecast_vs_actual.** International freshmen with at least 100 minutes: PER, points per game and usage rate forecast
before the season against the actual value, each player shown twice, control (hollow) and with the post-NCAA data (filled),
joined by a hairline; thick lines are running medians of the forecast within bins of the actual value; the dashed line is a
perfect forecast; named points are the largest improvements. Forecasts are the expected box score at the player's actual
exposure (games and minutes), the quantity the paper's production-given-opportunity term scores. Strips beneath each panel are
the distributions of signed error for both forecasts. Point forecasts move little for this group; the gain is in the
probability the models assign to the breakout seasons (ladder and cases figures).

**fig_league_translation.** International entrants by their main pre-college club league (the league with the most minutes
in the two years before the forecast cutoff; leagues with at least 12 entrants). Left: median and interquartile range of the
forecast first-season NCAA PER under the full model, with the median actual PER of those who played 100 minutes. Right: the
league's median forecast against its cross-league strength from the RAPM calibration, with the Spearman rank correlation.

**table_forecasts_2027.tex.** The 35 players of interest for 2026–27: pooled deployment forecasts (two seeds, tables v10)
with and without the post-NCAA data: season minutes, points per game if the player plays, PER at the forecast rates, and the
probability of a 400-minute rotation season.
