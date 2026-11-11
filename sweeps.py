"""Tightness sweeps and the extended decomposition.

Usage: python3 sweeps.py ext | sweep1d | sweep2d  [SIMS=...]
  ext      Shapley values with the maximum loss and the evaluation added as
           two more switchable rules (50,000 paths).
  sweep1d  one threshold varied at a time, Shapley values recomputed at each
           value (20,000 paths).
  sweep2d  two thresholds varied together for four pairs chosen in advance
           (20,000 paths).
Outputs: <mode>_shapley.csv and <mode>_interactions.csv
"""
import itertools, sys, time
import numpy as np
import pandas as pd
from engine import (TGT1, TGT2, EVAL_CONS, DAILY, MAXDD, F_MIN_QDAYS, F_QTHRESH, F_CONS, CAP, N_EVAL)
from rules import ARCHETYPES
from core import TRADERS, table, shapley, interaction, pair_dividend, se

MODE = sys.argv[1]
opts = dict(a.split("=", 1) for a in sys.argv[2:])
SHORT = {"A": "A_two_phase_static", "B": "B_one_step_eod_trailing", "C": "C_instant_funding"}


def apply(akey, cfg, name, value):
    c = cfg
    if name == "CONS":
        c[EVAL_CONS if akey == "B" else F_CONS] = value
    elif name == "MAXDD":
        c[MAXDD] = value
    elif name == "DAILY":
        c[DAILY] = value
    elif name == "QDAYS":
        c[F_MIN_QDAYS] = value
    elif name == "QTHRESH":
        c[F_QTHRESH] = value
    elif name == "CAP":
        c[CAP] = value
    elif name == "TGT1":
        c[TGT1] = value
        if akey == "A":
            c[TGT2] = value / 2          # second phase stays at half the first
    else:
        raise ValueError(name)


def decompose(cfg, rules, tkey, risk, smart, sims, tag):
    vals, bp, bm = table(cfg, rules, TRADERS[tkey], risk, smart, sims=sims)
    e = frozenset()
    full = frozenset(rules)
    srows, irows = [], []
    for metric, batches in (("p_payout", bp), ("e_payout", bm)):
        v = {S: vals[S][metric] for S in vals}
        sh, shb = shapley(rules, v), shapley(rules, batches)
        for r in rules:
            srows.append(dict(tag, trader=tkey, risk=risk, trader_mode="rule_aware" if smart else "naive",
                              metric=metric, rule=r, shapley_gain=sh[r], se=se(shb[r]),
                              baseline=v[e], all_removed=v[full],
                              one_at_a_time_gain=v[frozenset([r])] - v[e]))
        for i, j in itertools.combinations(rules, 2):
            irows.append(dict(tag, trader=tkey, risk=risk, trader_mode="rule_aware" if smart else "naive",
                              metric=metric, rule_i=i, rule_j=j, interaction=interaction(rules, v, i, j),
                              se=se(interaction(rules, batches, i, j)), pair_dividend=pair_dividend(v, i, j)))
    return srows, irows


def points():
    if MODE == "ext":
        for a in "ABC":
            yield a, {}, "extended"
    elif MODE == "sweep1d":
        grid = {
            "C": dict(CONS=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50], MAXDD=[0.04, 0.05, 0.06, 0.07, 0.08, 0.10],
                      DAILY=[0.02, 0.03, 0.04, 0.05], QDAYS=[3, 4, 5, 6, 8, 10],
                      QTHRESH=[0.0025, 0.005, 0.0075, 0.01], CAP=[0.005, 0.01, 0.02, 0.05]),
            "B": dict(MAXDD=[0.02, 0.03, 0.04, 0.05, 0.06], DAILY=[0.01, 0.015, 0.02, 0.03, 0.04],
                      CONS=[0.30, 0.40, 0.50, 0.60], QDAYS=[3, 5, 7, 10], TGT1=[0.04, 0.06, 0.08, 0.10]),
            "A": dict(TGT1=[0.06, 0.08, 0.10, 0.12], MAXDD=[0.06, 0.08, 0.10, 0.12], DAILY=[0.03, 0.04, 0.05]),
        }
        for a, g in grid.items():
            for name, values in g.items():
                for x in values:
                    yield a, {name: x}, name
    elif MODE == "sweep2d":
        cons = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        pairs = [("C", "CONS", cons, "QTHRESH", [0.0025, 0.005, 0.0075, 0.01]),
                 ("C", "CONS", cons, "MAXDD", [0.04, 0.05, 0.07, 0.10]),
                 ("B", "MAXDD", [0.02, 0.03, 0.04, 0.05, 0.06], "DAILY", [0.01, 0.015, 0.02, 0.03, 0.04]),
                 ("C", "CONS", cons, "CAP", [0.005, 0.01, 0.02, 0.05])]
        for a, n1, v1, n2, v2 in pairs:
            for x in v1:
                for y in v2:
                    yield a, {n1: x, n2: y}, f"{n1}x{n2}"


SIMS = int(opts.get("SIMS", 50_000 if MODE == "ext" else 20_000))
RISKS = [float(x) for x in opts.get("RISKS", "0.005,0.01").split(",")]
S, I = [], []
t0 = time.time()
for n, (a, over, sweep) in enumerate(points()):
    akey = SHORT[a]
    A = ARCHETYPES[akey]
    cfg = A["cfg"].copy()
    for name, value in over.items():
        apply(a, cfg, name, value)
    rules = list(A["rules"])
    if MODE == "ext":
        rules += ["max_loss"] + (["evaluation"] if cfg[N_EVAL] > 0 else [])
    names = list(over)
    tag = dict(sweep=sweep, archetype=akey,
               p1=names[0] if names else "", v1=over[names[0]] if names else np.nan,
               p2=names[1] if len(names) > 1 else "", v2=over[names[1]] if len(names) > 1 else np.nan)
    for tkey in TRADERS:
        for risk in RISKS:
            for smart in (0, 1):
                s, i = decompose(cfg, rules, tkey, risk, smart, SIMS, tag)
                S += s
                I += i
    print(f"{MODE} point {n} {a} {over} done {time.time() - t0:.0f}s", flush=True)
    if n % 5 == 4:
        pd.DataFrame(S).to_csv(f"{MODE}_shapley.csv", index=False)
        pd.DataFrame(I).to_csv(f"{MODE}_interactions.csv", index=False)
pd.DataFrame(S).to_csv(f"{MODE}_shapley.csv", index=False)
pd.DataFrame(I).to_csv(f"{MODE}_interactions.csv", index=False)
print("ALLDONE", MODE, "SIMS", SIMS)
