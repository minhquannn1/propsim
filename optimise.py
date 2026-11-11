"""Rule-aware trader with a searched policy.

For every on/off combination of the rules, the trader picks the policy with the
highest payout probability from a fixed grid:
  risk per trade   0.25, 0.5, 1.0, 1.5 %
  sizing           never risk more than 10 / 25 / 50 % of the distance to the
                   nearest loss floor, or no such limit
  day stop         bank the day at 0.5x or 1x the best day the consistency rule
                   allows, or no such stop
  qualifying stop  bank the day once it qualifies, or not
  profit target    (gold strategies only) the strategy's own exit, or a fixed
                   target of 1R, 2R or 3R
The search uses 10,000 paths with one random seed. The chosen policy is then
re-run on 50,000 paths with the main seed, so the reported numbers are not
flattered by the search. Shapley values are computed on those re-run values:
they measure what each rule costs a trader who responds to every ruleset as
well as this grid allows.

Usage: python3 optimise.py [tag] [ARCH=A,B,C] [NAME=value ...] [SEARCH_SIMS=10000]
  NAME=value changes one threshold before the search (CONS, MAXDD, DAILY,
  QDAYS, QTHRESH, CAP, TGT1), for example: python3 optimise.py _cons30 ARCH=C CONS=0.30
Outputs: opt_policy<tag>.csv, opt_shapley<tag>.csv, opt_interactions<tag>.csv
"""
import itertools, sys, time
import numpy as np
import pandas as pd
from engine import EVAL_CONS, F_CONS, F_QTHRESH, F_MIN_QDAYS, N_EVAL, MAXDD, DAILY, CAP, TGT1, TGT2
from rules import ARCHETYPES, relax
from core import TRADERS, EDGE, OFF, run_cell, subsets, shapley, interaction, pair_dividend, se

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
opts = dict(a.split("=", 1) for a in sys.argv[2:])
SEARCH_SEED, SEARCH_SIMS = 777, int(opts.pop("SEARCH_SIMS", 10_000))
ARCHS = opts.pop("ARCH", "A,B,C").split(",")
SHORT = {"A": "A_two_phase_static", "B": "B_one_step_eod_trailing", "C": "C_instant_funding"}


def override(a, cfg):
    for name, value in opts.items():
        value = float(value)
        if name == "CONS":
            cfg[EVAL_CONS if a == "B" else F_CONS] = value
        elif name == "TGT1":
            cfg[TGT1] = value
            if a == "A":
                cfg[TGT2] = value / 2
        else:
            cfg[{"MAXDD": MAXDD, "DAILY": DAILY, "QDAYS": F_MIN_QDAYS, "QTHRESH": F_QTHRESH, "CAP": CAP}[name]] = value
    return cfg


RISKS = [0.0025, 0.005, 0.01, 0.015]
ROOMS = [0.10, 0.25, 0.50, OFF]
DMULTS = [0.5, 1.0, OFF]
VARIANTS = {
    "zero_edge_EURUSD": {"own exit (1.5R)": TRADERS["zero_edge_EURUSD"]},
    "edge_+0.10R_EURUSD": {"own exit (1.5R)": TRADERS["edge_+0.10R_EURUSD"]},
    "breakout_XAUUSD": {"own exit (2R)": TRADERS["breakout_XAUUSD"],
                        "1R": dict(ttype=1, edge=0.0, lib="b1"), "3R": dict(ttype=1, edge=0.0, lib="b3")},
    "trend_XAUUSD": {"own exit (none)": TRADERS["trend_XAUUSD"], "1R": dict(ttype=1, edge=0.0, lib="t1"),
                     "2R": dict(ttype=1, edge=0.0, lib="t2"), "3R": dict(ttype=1, edge=0.0, lib="t3")},
}


def grid(cfg):
    cons_on = (cfg[F_CONS] < 0.999) if cfg[N_EVAL] == 0 else (cfg[EVAL_CONS] < 0.999 or cfg[F_CONS] < 0.999)
    q_on = cfg[F_QTHRESH] > 0 and cfg[F_MIN_QDAYS] > 0
    for risk in RISKS:
        for room in ROOMS:
            for dm in (DMULTS if cons_on else [OFF]):
                for q in ((0, 1) if q_on else (0,)):
                    yield risk, room, dm, q


P, S, I = [], [], []
t0 = time.time()
for a in ARCHS:
    akey = SHORT[a]
    A = ARCHETYPES[akey]
    base_cfg = override(a, A["cfg"].copy())
    rules = A["rules"]
    for tkey, variants in VARIANTS.items():
        vals, bp, bm = {}, {}, {}
        for Sset in subsets(rules):
            cfg = relax(base_cfg, Sset)
            best = None
            for vname, tr in variants.items():
                for risk, room, dm, q in grid(cfg):
                    m, _, _ = run_cell(cfg, tr, risk, 1, sims=SEARCH_SIMS, seed=SEARCH_SEED,
                                       room_frac=room, dstop_mult=dm, qstop=q)
                    if best is None or m["p_payout"] > best[0]:
                        best = (m["p_payout"], vname, risk, room, dm, q)
            _, vname, risk, room, dm, q = best
            m, b1, b2 = run_cell(cfg, variants[vname], risk, 1, room_frac=room, dstop_mult=dm, qstop=q)
            vals[Sset], bp[Sset], bm[Sset] = m, b1, b2
            P.append(dict(archetype=akey, trader=tkey, removed="+".join(sorted(Sset)) or "none", n_removed=len(Sset),
                          target=vname, risk=risk, room_frac=room, dstop_mult=dm, qstop=q,
                          p_search=best[0], p_payout=m["p_payout"], e_payout=m["e_payout"],
                          ev_per_fee=m["e_payout"] / A["fee"]))
        e, full = frozenset(), frozenset(rules)
        for metric, batches in (("p_payout", bp), ("e_payout", bm)):
            v = {k: vals[k][metric] for k in vals}
            sh, shb = shapley(rules, v), shapley(rules, batches)
            for r in rules:
                S.append(dict(archetype=akey, trader=tkey, metric=metric, rule=r, shapley_gain=sh[r], se=se(shb[r]),
                              baseline=v[e], all_removed=v[full], one_at_a_time_gain=v[frozenset([r])] - v[e]))
            for i, j in itertools.combinations(rules, 2):
                I.append(dict(archetype=akey, trader=tkey, metric=metric, rule_i=i, rule_j=j,
                              interaction=interaction(rules, v, i, j), se=se(interaction(rules, batches, i, j)),
                              pair_dividend=pair_dividend(v, i, j)))
        print(f"{akey} {tkey} done {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(P).to_csv(f"opt_policy{TAG}.csv", index=False)
        pd.DataFrame(S).to_csv(f"opt_shapley{TAG}.csv", index=False)
        pd.DataFrame(I).to_csv(f"opt_interactions{TAG}.csv", index=False)
print("ALLDONE")
