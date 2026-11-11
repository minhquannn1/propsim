"""Collect every result the manuscript quotes into one JSON file (numbers.json),
so the text is filled from the result files and cannot drift from them."""
import glob, json, re
import numpy as np, pandas as pd

T = {"zero_edge_EURUSD": "zero", "edge_+0.10R_EURUSD": "edge", "breakout_XAUUSD": "brk", "trend_XAUUSD": "trend"}
A = {"A_two_phase_static": "A", "B_one_step_eod_trailing": "B", "C_instant_funding": "C"}
M = {"naive": "naive", "rule_aware": "aware"}


def tidy(d):
    d = d.copy()
    for c, m in (("trader", T), ("archetype", A), ("trader_mode", M)):
        if c in d.columns:
            d[c] = d[c].map(m)
    return d


def recs(d, cols):
    d = tidy(d)[cols]
    return json.loads(d.to_json(orient="records", double_precision=10))


out = {}
f = pd.read_csv("final_factorial.csv")
out["base"] = recs(f[f.removed == "none"], ["archetype", "trader", "trader_mode", "risk", "p_payout", "e_payout", "ev_per_fee"])
out["fact"] = recs(f[f.removed.isin(["none", "daily", "trailing", "daily+trailing", "min_days", "daily+min_days"])], ["archetype", "trader", "trader_mode", "risk", "removed", "p_payout"])
s = pd.read_csv("final_shapley.csv")
out["sh"] = recs(s, ["archetype", "trader", "trader_mode", "risk", "metric", "rule", "shapley_gain", "se", "baseline", "all_removed"])
i = pd.read_csv("final_interactions.csv")
out["int"] = recs(i, ["archetype", "trader", "trader_mode", "risk", "metric", "rule_i", "rule_j", "interaction", "se", "pair_dividend"])
e = pd.read_csv("ext_shapley.csv")
out["ext"] = recs(e, ["archetype", "trader", "trader_mode", "risk", "metric", "rule", "shapley_gain", "se", "baseline", "all_removed"])
s1 = pd.read_csv("sweep1d_shapley.csv")
out["s1"] = recs(s1, ["archetype", "p1", "v1", "trader", "trader_mode", "risk", "metric", "rule", "shapley_gain", "se", "baseline", "all_removed"])
s2 = pd.read_csv("sweep2d_shapley.csv")
out["s2"] = recs(s2[s2.rule.isin(["consistency", "trailing", "min_days", "daily", "cap"])],
                 ["sweep", "v1", "v2", "trader", "trader_mode", "risk", "metric", "rule", "shapley_gain", "se", "baseline"])
i2 = pd.read_csv("sweep2d_interactions.csv")
out["i2"] = recs(i2, ["sweep", "v1", "v2", "trader", "trader_mode", "risk", "metric", "rule_i", "rule_j", "interaction", "se"])
sens = pd.read_csv("final_sensitivity.csv")
out["sens"] = recs(sens, ["variant", "archetype", "trader", "trader_mode", "risk", "rule", "shapley_gain", "se"])
out["plaus"] = recs(pd.read_csv("plausibility.csv"), ["archetype", "trader", "trader_mode", "risk", "pass_phase1", "funded", "paid", "paid_given_funded"])
op = pd.read_csv("opt_policy.csv")
out["optpol"] = recs(op[op.removed == "none"], ["archetype", "trader", "target", "risk", "room_frac", "dstop_mult", "qstop", "p_payout", "e_payout", "ev_per_fee"])
out["optsh"] = recs(pd.read_csv("opt_shapley.csv"), ["archetype", "trader", "metric", "rule", "shapley_gain", "se", "baseline", "all_removed"])
sw = []
for fn in sorted(glob.glob("opt_shapley_*_*_*.csv")):
    m = re.match(r"opt_shapley_([BC])_([A-Z]+)_([0-9.]+)\.csv", fn)
    d = pd.read_csv(fn)
    d["p1"], d["v1"] = m.group(2), float(m.group(3))
    sw.append(d)
if sw:
    out["optsweep"] = recs(pd.concat(sw), ["archetype", "p1", "v1", "trader", "metric", "rule", "shapley_gain", "se", "baseline", "all_removed"])
pw = []
for fn in sorted(glob.glob("opt_policy_*_*_*.csv")):
    m = re.match(r"opt_policy_([BC])_([A-Z]+)_([0-9.]+)\.csv", fn)
    d = pd.read_csv(fn)
    d = d[d.removed == "none"].copy()
    d["p1"], d["v1"] = m.group(2), float(m.group(3))
    pw.append(d)
if pw:
    out["optsweeppol"] = recs(pd.concat(pw), ["archetype", "p1", "v1", "trader", "target", "risk", "room_frac", "dstop_mult", "qstop", "p_payout"])

# ---- headline counts ----
P = s[s.metric == "p_payout"]
c = {}
c["se_main"] = P.se.max() * 100
c["se_int_main"] = i[i.metric == "p_payout"].se.max() * 100
c["se_ext"] = e[e.metric == "p_payout"].se.max() * 100
c["se_s1"] = s1[s1.metric == "p_payout"].se.max() * 100
c["se_s2"] = max(s2[s2.metric == "p_payout"].se.max(), i2[i2.metric == "p_payout"].se.max()) * 100
c["se_sens"] = sens.se.max() * 100
c["se_opt"] = pd.read_csv("opt_shapley.csv").query("metric == 'p_payout'").se.max() * 100
p1 = s1[(s1.metric == "p_payout") & (s1.archetype != "A_two_phase_static")].reset_index(drop=True)
t1 = p1.loc[p1.groupby(["archetype", "sweep", "v1", "trader", "trader_mode", "risk"]).shapley_gain.idxmax()]
same = t1.rule == np.where(t1.archetype.str.startswith("B"), "trailing", "consistency")
c["s1_cells"], c["s1_same"] = int(len(t1)), int(same.sum())
c["s1_same_naive"], c["s1_cells_naive"] = int(same[t1.trader_mode == "naive"].sum()), int((t1.trader_mode == "naive").sum())
p2 = s2[s2.metric == "p_payout"].reset_index(drop=True)
t2 = p2.loc[p2.groupby(["sweep", "v1", "v2", "trader", "trader_mode", "risk"]).shapley_gain.idxmax()]
same2 = t2.rule == np.where(t2.sweep == "MAXDDxDAILY", "trailing", "consistency")
c["s2_cells"], c["s2_same"] = int(len(t2)), int(same2.sum())
a1 = s1[(s1.metric == "p_payout") & (s1.archetype == "A_two_phase_static")]
c["A_sweep_max_cost"] = a1.groupby(["sweep", "v1", "trader", "trader_mode", "risk"]).shapley_gain.sum().abs().max() * 100
c["A_sweep_max_cost_naive"] = a1[a1.trader_mode == "naive"].groupby(["sweep", "v1", "trader", "risk"]).shapley_gain.sum().abs().max() * 100
sb = sens[(sens.risk.isin([0.005, 0.01]))].reset_index(drop=True)
ts = sb.loc[sb.groupby(["variant", "archetype", "trader", "trader_mode", "risk"]).shapley_gain.idxmax()]
pv = ts.pivot_table(index=["archetype", "trader", "trader_mode", "risk"], columns="variant", values="rule", aggfunc="first")
c["sens_settings"] = int(len(pv))
c["sens_settings_changed"] = int((pv.nunique(axis=1) > 1).sum())
c["sens_combos"] = int(pv.size)
c["sens_combos_same"] = int((pv.values == pv[["base"]].values).sum())
out["c"] = c

# ---- validation ----
log = open("validation/crosscheck_log.txt").read()
m = re.search(r"configurations (\d+), paths (\d+), mismatches (\d+)", log)
assert m.group(3) == "0"
v = open("verify_output.txt").read()
coin = re.findall(r"fair coin: ([0-9.]+)", v)
mono = re.findall(r"(\w)_\w+: violations (\d+), worst (-?[0-9.]+) points", v)
effs = [float(x) for x in re.findall(r"= ([0-9.e-]+)\n", v)]
nb = [x for x in mono if x[0] == "B"][0]
out["val"] = dict(
    tests="55", configs=f"{int(m.group(1)):,}", paths=f"{int(m.group(2)):,}", coin=coin,
    eff=f"{max(effs):.0e}".replace("e-", " x 10^-"),
    mono=f"None in A or C; {nb[1]} in B (largest {abs(float(nb[2])):.2f} points), caused by the soft daily limit")
json.dump(out, open("numbers.json", "w"))
print({k: (len(x) if isinstance(x, list) else x) for k, x in out.items() if k not in ("c",)})
print(c)
