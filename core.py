"""Shared pieces: trade libraries, traders, one simulation cell, the table of
all on/off rule combinations, Shapley values and pairwise interaction terms."""
import itertools, math
import numpy as np
from engine import simulate
from rules import ACCOUNT, relax

SEED = 20261003
NB = 20                                   # batches for standard errors
OFF = 1e9                                 # switches a policy element off

Z1, Z3 = np.zeros(1), np.zeros((1, 96, 2))
NOB = (Z1, Z1, Z1, np.zeros(1, dtype=np.bool_))
NOR = (Z3, Z3, Z3, np.zeros((1, 96, 2), dtype=np.int64))

_LIBS = {}


def libs(tag=""):
    if tag not in _LIBS:
        L = np.load(f"libs{tag}.npz")
        _LIBS[tag] = {k: L[k] for k in L.files}
    return _LIBS[tag]


EDGE = float(libs()["edge"])               # same skill in every variant
TRADERS = {
    "zero_edge_EURUSD": dict(ttype=0, edge=0.0, lib="r"),
    "edge_+0.10R_EURUSD": dict(ttype=0, edge=EDGE, lib="r"),
    "breakout_XAUUSD": dict(ttype=1, edge=0.0, lib="b"),
    "trend_XAUUSD": dict(ttype=1, edge=0.0, lib="t"),
}


def run_cell(cfg, tr, risk, smart=0, sims=50_000, H=250, block=20, wgoal=0.02, seed=SEED,
             libtag="", room_frac=0.25, dstop_mult=1.0, qstop=1):
    """One simulation cell. Returns summary numbers and per-batch means."""
    L = libs(libtag)
    if tr["ttype"] == 0:
        lib, b = (L["R"], L["M"], L["F"], L["E"]), NOB
    else:
        x = tr["lib"]
        lib, b = NOR, (L["R" + x], L["M" + x], L["F" + x], L["h" + x])
    o, p, u, s = simulate(cfg, tr["ttype"], risk, tr["edge"], 2, *lib, *b,
                          sims, H, seed, block, wgoal, smart, room_frac, dstop_mult, qstop)
    paid = (o == 1).astype(float)
    money = p * ACCOUNT
    return dict(
        p_payout=paid.mean(), e_payout=money.mean(),
        fail_daily=(o == 2).mean(), fail_maxdd=(o == 3).mean(), censored=(o == 0).mean(),
        pass_eval=(s >= cfg[0]).mean(),
        median_days_to_payout=float(np.median(u[o == 1])) if (o == 1).any() else np.nan,
    ), paid.reshape(NB, -1).mean(1), money.reshape(NB, -1).mean(1)


def subsets(rules):
    for k in range(len(rules) + 1):
        for S in itertools.combinations(rules, k):
            yield frozenset(S)


def table(cfg, rules, tr, risk, smart=0, **kw):
    """Run every on/off combination. Returns {removed set: summary},
    {set: batch payout rate}, {set: batch expected payout}."""
    vals, bp, bm = {}, {}, {}
    for S in subsets(rules):
        vals[S], bp[S], bm[S] = run_cell(relax(cfg, S), tr, risk, smart, **kw)
    return vals, bp, bm


def shapley(rules, v):
    """Shapley value of each rule. v maps a set of removed rules to a number
    (or an array with one entry per batch)."""
    n = len(rules)
    out = {}
    for r in rules:
        others = [x for x in rules if x != r]
        tot = 0.0
        for k in range(n):
            for S in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                tot = tot + w * (v[frozenset(S) | {r}] - v[frozenset(S)])
        out[r] = tot
    return out


def interaction(rules, v, i, j):
    """Shapley interaction index of rules i and j (Grabisch and Roubens):
    the average, over the other rules, of
    v(S + i + j) - v(S + i) - v(S + j) + v(S).
    Positive: removing both gains more than the two separate gains added up
    (the rules overlap: either one alone still blocks the payout).
    Negative: the rules reinforce each other."""
    n = len(rules)
    others = [x for x in rules if x not in (i, j)]
    tot = 0.0
    for k in range(n - 1):
        for S in itertools.combinations(others, k):
            S = frozenset(S)
            w = math.factorial(k) * math.factorial(n - k - 2) / math.factorial(n - 1)
            tot = tot + w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
    return tot


def pair_dividend(v, i, j):
    """Harsanyi dividend of the pair: the joint gain from removing only i and j
    beyond the two single gains, starting from the full ruleset."""
    e = frozenset()
    return v[frozenset([i, j])] - v[frozenset([i])] - v[frozenset([j])] + v[e]


def se(batch_values):
    return float(np.std(batch_values, ddof=1) / math.sqrt(len(batch_values)))
