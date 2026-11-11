# Which rule kills the account?

Code and results for the paper "Which Rule Kills the Account? A Shapley
Decomposition of Retail Prop-Firm Rules and Their Tightness" (Nguyen Ngoc Minh
Quan, submitted to the Journal of High School Science).

The code simulates retail prop-firm accounts under three rulesets, switches
every combination of rules on and off, and splits the lost payout probability
between the rules with Shapley values. It then repeats the decomposition while
the rule thresholds are varied.

## What is in this repository

| Path | What it is |
|---|---|
| `data.py` | Loads 15-minute bars, builds the trade libraries (random-entry EURUSD trades, gold Asian-range breakout, gold 50-day trend) |
| `engine.py` | The Monte Carlo engine (Numba). One account per path, day by day |
| `rules.py` | The three rulesets and the rule switches |
| `core.py` | One simulation cell, the on/off table, Shapley values, pairwise interaction index |
| `build_libs.py` | Builds `libs.npz` from the price files |
| `run_final.py` | Main runs: 3 rulesets x 4 traders x 4 risk levels x naive/rule-aware (50,000 paths) |
| `run_sens.sh` | Seven robustness variants |
| `sweeps.py` | Extended decomposition, one-threshold sweeps, two-threshold sweeps |
| `optimise.py` | Rule-aware trader with a searched policy |
| `plausibility.py` | Stage-by-stage pass and payout rates for comparison with published figures |
| `strategy_search_intraday.py` | First strategy search: 102 intraday signals on the random-entry trade (none profitable) |
| `strategy_search.py` | Second strategy search: 96 daily trend rules (selection 2012 to 2019, check 2020 to 2026) |
| `make_numbers.py` | Collects every number quoted in the paper into `numbers.json` |
| `verify.py` | Barrier-ceiling, Shapley-efficiency, monotonicity and edge checks |
| `validation/SPEC.md` | Written specification of the account rules |
| `validation/reference.py` | Second, plain-Python implementation written from the specification only |
| `validation/test_reference.py` | 55 hand-worked scenarios for the reference implementation |
| `validation/crosscheck.py` | Path-by-path comparison of `engine.py` with `reference.py` |
| `make_fig12.py`, `make_fig_sweeps.py`, `figstyle.py` | Figures |
| `results/` | Every CSV the paper's numbers come from (the scripts write to, and read from, the working directory; copy the files up one level to redraw the figures without re-running) |
| `libs.npz`, `libs_spread2.npz` | The trade libraries (results of every candidate trade in units of risk), enough to rerun every simulation without the raw prices |

## How to reproduce

Python 3.11 or later with `numpy`, `pandas`, `numba`, `matplotlib`.

With the trade libraries in this repository (no price data needed):

```
python3 run_final.py            # main results, about 6 minutes on 2 cores
sh run_sens.sh                  # robustness variants
python3 sweeps.py ext           # extended decomposition
python3 sweeps.py sweep1d       # one-threshold sweeps
python3 sweeps.py sweep2d       # two-threshold sweeps
python3 plausibility.py
python3 optimise.py             # searched policy
python3 validation/test_reference.py
python3 validation/crosscheck.py 150
python3 make_fig12.py && python3 make_fig_sweeps.py
```

Every run uses fixed seeds, so the numbers come out the same.

## Price data

The raw 15-minute prices are not redistributed here.

- EURUSD and XAUUSD up to 31 December 2020: the public repository
  https://github.com/ejtraderLabs/historical-data (files `EURUSD/EURUSDm15.csv`
  and `XAUUSD/XAUUSDm15.csv`, prices stored as integers: EURUSD x 100,000,
  XAUUSD x 100).
- 4 January 2021 to 7 October 2026: exported from the MetaTrader 5 server
  AudaCityGlobal-Live (AudaCity Global Limited) with a short MQL5 script that
  calls `CopyRates(symbol, PERIOD_M15, ...)` and writes
  `Date,open,high,low,close,tick_volume,spread`. Any MetaTrader 5 broker whose
  server clock is GMT+2/+3 (New York close = midnight) gives comparable data.
  Over January 2021 to March 2022, where the two sources overlap, they matched
  bar for bar.

To rebuild the libraries from prices, put `EURUSDm15.csv` and `XAUUSDm15.csv`
(columns `Date,open,high,low,close,tick_volume`, integer-scaled prices as
above) in a folder and run

```
PROPSIM_DATA=<folder> python3 build_libs.py
PROPSIM_DATA=<folder> python3 build_libs.py 2.0 _spread2
```

## Use of AI

The simulation and analysis code was written with an AI assistant (Claude,
Anthropic) under the author's direction. The second implementation in
`validation/reference.py` was produced by a separate AI session that was given
only `validation/SPEC.md` and no access to the engine code.

## Licence

MIT for the code. The results files may be reused with citation.
