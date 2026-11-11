"""Final runs: full factorial x archetypes x traders x risk levels, for the naive
and the rule-aware trader, with Shapley values and batch standard errors.

Usage: python3 run_final.py [tag] [key=value ...]
  keys: WGOAL, BLOCK, H, LIBS (libs file suffix), RISKS (comma list), SIMS
Outputs: final_factorial<tag>.csv, final_shapley<tag>.csv
"""
import itertools, math, sys, time
import numpy as np
import pandas as pd
from rules import ARCHETYPES, ACCOUNT, relax

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
opts = dict(a.split("=", 1) for a in sys.argv[2:])
SIMS = int(opts.get("SIMS", 50_000))
H = int(opts.get("H", 250))
BLOCK = int(opts.get("BLOCK", 20))
WGOAL = float(opts.get("WGOAL", 0.02))
RISKS = [float(x) for x in opts.get("RISKS", "0.0025,0.005,0.01,0.015").split(",")]
NB = 20

from core import TRADERS, run_cell, shapley, interaction, se, libs


def run(cfg, tr, risk, smart):
    return run_cell(cfg, tr, risk, smart, sims=SIMS, H=H, block=BLOCK, wgoal=WGOAL,
                    libtag=opts.get("LIBS", ""))


if __name__ == "__main__":
    rows, shap_rows, int_rows = [], [], []
    t0 = time.time()
    for smart in (0, 1):
        for akey, A in ARCHETYPES.items():
            rules = A["rules"]
            for tkey, tr in TRADERS.items():
                for risk in RISKS:
                    vals, bp, bm = {}, {}, {}
                    for k in range(len(rules) + 1):
                        for S in itertools.combinations(rules, k):
                            m, batch_p, batch_m = run(relax(A["cfg"], S), tr, risk, smart)
                            m.update(trader_mode="rule_aware" if smart else "naive", archetype=akey,
                                     trader=tkey, risk=risk, removed="+".join(S) or "none",
                                     n_removed=k, ev_per_fee=m["e_payout"] / A["fee"])
                            rows.append(m)
                            vals[frozenset(S)] = m
                            bp[frozenset(S)], bm[frozenset(S)] = batch_p, batch_m
                    for metric, batches in (("p_payout", bp), ("e_payout", bm)):
                        v = {S: vals[S][metric] for S in vals}
                        sh = shapley(rules, v)
                        sh_b = shapley(rules, batches)          # one value per batch
                        for i, j in itertools.combinations(rules, 2):
                            ib = interaction(rules, batches, i, j)
                            int_rows.append(dict(
                                trader_mode="rule_aware" if smart else "naive", archetype=akey,
                                trader=tkey, risk=risk, metric=metric, rule_i=i, rule_j=j,
                                interaction=interaction(rules, v, i, j), se=se(ib),
                                pair_dividend=v[frozenset([i, j])] - v[frozenset([i])] - v[frozenset([j])] + v[frozenset()]))
                        for r in rules:
                            shap_rows.append(dict(
                                trader_mode="rule_aware" if smart else "naive", archetype=akey,
                                trader=tkey, risk=risk, metric=metric, rule=r,
                                shapley_gain=sh[r], se=float(np.std(sh_b[r], ddof=1) / math.sqrt(NB)),
                                baseline=v[frozenset()], all_removed=v[frozenset(rules)],
                                one_at_a_time_gain=v[frozenset([r])] - v[frozenset()]))
            print(f"smart={smart} {akey} done {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(f"final_factorial{TAG}.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(f"final_shapley{TAG}.csv", index=False)
    pd.DataFrame(int_rows).to_csv(f"final_interactions{TAG}.csv", index=False)
    print("rows", len(rows), "settings", dict(SIMS=SIMS, H=H, BLOCK=BLOCK, WGOAL=WGOAL, RISKS=RISKS, LIBS=opts.get("LIBS", "")))
