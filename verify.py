"""Sanity checks for the simulator (run after run_final.py)."""
import numpy as np, pandas as pd
from data import load_grid, random_library
from engine import simulate
from rules import _cfg, ARCHETYPES
from core import libs, NOB

L = libs()
print("== 1. Barrier ceiling check (Villahermosa: P(pass) = L/(T+L) with no edge)")
import os
shape = L["R"].shape
libs_to_test = []
# synthetic fair coin: every trade is +1R or -1R, long and short exact mirrors (zero drift)
rng = np.random.default_rng(1)
Rs = rng.choice([-1.0, 1.0], size=shape); Rs[..., 1] = -Rs[..., 0]
Ms = np.minimum(Rs, 0.0); Fs = np.maximum(Rs, 0.0); Es = np.full(shape, 95, dtype=np.int64)
libs_to_test.append(((Rs, Ms, Fs, Es), "fair coin"))
DATA = os.environ.get("PROPSIM_DATA", "data_combined")
if os.path.exists(f"{DATA}/EURUSDm15.csv"):            # needs the raw prices
    eu = load_grid("EURUSD", DATA)
    R0, M0, F0, E0 = random_library(eu, costs=False)
    print("   mean R per trade without costs:", R0[:, 40:76, :].mean().round(4))
    libs_to_test.append(((R0, M0, F0, E0), "real EURUSD, no costs"))
libs_to_test.append(((L["R"], L["M"], L["F"], L["E"]), "real EURUSD, 1 pip"))
for T, Lm in [(0.10, 0.10), (0.08, 0.05), (0.05, 0.10)]:
    cfg = _cfg(N_EVAL=1, TGT1=T, MAXDD=Lm, TRAIL_MODE=0, F_MIN_PROFIT=1e9)
    line = f"   T={T:.0%} L={Lm:.0%} theory {Lm/(T+Lm):.3f}"
    for lib, name in libs_to_test:
        o, p, u, s = simulate(cfg, 0, 0.0025, 0.0, 2, *lib, *NOB, 20000, 20000, 7, 20, 0.0, 0)
        line += f" | {name}: {(s>=1).mean():.3f}"
    print(line)

print("== 2. Shapley efficiency (sum of rule shares = total effect)")
for fn in ["final_shapley.csv", "ext_shapley.csv", "sweep1d_shapley.csv", "sweep2d_shapley.csv", "opt_shapley.csv"]:
    try:
        sh = pd.read_csv(fn)
    except FileNotFoundError:
        continue
    keys = [c for c in ["sweep", "archetype", "p1", "v1", "p2", "v2", "trader_mode", "trader", "risk", "metric"] if c in sh.columns]
    g = sh.groupby(keys, dropna=False).agg(s=("shapley_gain", "sum"), b=("baseline", "first"), a=("all_removed", "first"))
    print(f"   {fn}: max |sum - (all_removed - baseline)| = {float((g.s - (g.a - g.b)).abs().max()):.2e}")

print("== 3. Monotonicity: removing a rule should never lower the naive trader's P(payout)")
f = pd.read_csv("final_factorial.csv")
f = f[f.trader_mode == "naive"]
for akey, A in ARCHETYPES.items():
    viol, worst = 0, 0.0
    sub = f[f.archetype == akey]
    for (tr, rk), grp in sub.groupby(["trader", "risk"]):
        v = {frozenset([] if r == "none" else r.split("+")): x for r, x in zip(grp.removed, grp.p_payout)}
        for S in v:
            for r in A["rules"]:
                if r not in S:
                    d = v[S | {r}] - v[S]
                    if d < -1e-12:
                        viol += 1; worst = min(worst, d)
    print(f"   {akey}: violations {viol}, worst {worst * 100:.3f} points")

print("== 4. Edge trader expectancy check")
R = L["R"][:, 40:76, :]
best = np.maximum(R[..., 0], R[..., 1]).mean(); rnd = R.mean(); e = float(L["edge"])
print("   implied mean R per trade:", round(e * best + (1 - e) * rnd, 4))
