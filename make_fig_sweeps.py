"""Figures 3 to 6: tightness sweeps and pair heat maps."""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from figstyle import *

B, C = "B_one_step_eod_trailing", "C_instant_funding"
PCT = {"CONS": "Consistency Limit (%)", "MAXDD": "Drawdown Distance (%)",
       "DAILY": "Daily Loss Limit (%)", "QTHRESH": "Qualifying-Day Threshold (%)", "CAP": "Payout Cap (US$)"}
BASE = {(C, "CONS"): 20, (C, "MAXDD"): 7, (B, "MAXDD"): 3, (B, "DAILY"): 2, (C, "QTHRESH"): 0.5, (C, "CAP"): 1000}


def scale(name, v):
    return v * 100_000 if name == "CAP" else v * 100


# ---------- Figure 3: one threshold at a time ----------
s = pd.read_csv("sweep1d_shapley.csv")
s = s[s.metric == "p_payout"]
rows = [(C, "CONS", 0.005, "Ruleset C, Consistency Limit Varied, 0.5% Risk"),
        (C, "MAXDD", 0.005, "Ruleset C, Trailing Distance Varied, 0.5% Risk"),
        (B, "MAXDD", 0.01, "Ruleset B, Trailing Distance Varied, 1% Risk"),
        (B, "DAILY", 0.01, "Ruleset B, Daily Loss Limit Varied, 1% Risk")]
show = ["consistency", "trailing", "min_days", "daily"]
fig, axes = plt.subplots(4, 4, figsize=(11, 11.6), sharey="row", constrained_layout=True)
for i, (ak, sw, risk, rowname) in enumerate(rows):
    for j, tk in enumerate(TRADERS):
        ax = axes[i, j]
        style(ax, left=True)
        d = s[(s.archetype == ak) & (s.sweep == sw) & (s.risk == risk) & (s.trader == tk)]
        for mk, ls, lw in (("naive", "-", 2.0), ("rule_aware", "--", 1.5)):
            for r in show:
                x = d[(d.trader_mode == mk) & (d.rule == r)].sort_values("v1")
                ax.plot(scale(sw, x.v1), x.shapley_gain * 100, ls, color=RULE_COL[r], lw=lw,
                        marker="o" if mk == "naive" else None, ms=3.5, markeredgecolor=SURF, markeredgewidth=0.6)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.axvline(BASE[(ak, sw)], color=MUTED, lw=0.8, ls=":")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_xticks(sorted(scale(sw, d.v1.unique())))
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel(PCT[sw], fontsize=8)
        if i == 0:
            ax.set_title(TRADERS[tk], fontsize=9.5, fontweight="bold")
        if j == 0:
            ax.set_ylabel("Shapley Value (Points)", fontsize=8.5)
            t = ax.text(0.0, 1.13 if i == 0 else 1.05, rowname, transform=ax.transAxes, fontsize=9,
                        fontweight="bold", color=INK, ha="left", va="bottom")
            t.set_in_layout(False)
h = [Line2D([], [], color=RULE_COL[r], lw=2, label=RULE_NAMES[r]) for r in show]
h += [Line2D([], [], color=MUTED, lw=2, ls="-", marker="o", ms=3.5, label="Naive"),
      Line2D([], [], color=MUTED, lw=1.5, ls="--", label="Rule-Aware"),
      Line2D([], [], color=MUTED, lw=0.8, ls=":", label="Published Setting")]
fig.legend(handles=h, loc="outside upper center", ncol=7, frameon=False, fontsize=8.5)
fig.get_layout_engine().set(h_pad=0.12, hspace=0.06)
save(fig, "Nguyen_Figure3")
print("fig3 saved")


# ---------- Figures 4 to 6: two thresholds at a time ----------
S2 = pd.read_csv("sweep2d_shapley.csv")
I2 = pd.read_csv("sweep2d_interactions.csv")


def grid_of(df, value, **sel):
    d = df
    for k, v in sel.items():
        d = d[d[k] == v]
    return d.pivot_table(index="v2", columns="v1", values=value).sort_index(ascending=False)


def heat(ax, g, kind, n1, n2, fmt="{:.0f}", mult=100.0, vmax=None):
    z = g.values * mult
    if kind == "div":
        m = vmax or max(1e-9, np.nanmax(np.abs(z)))
        im = ax.imshow(z, cmap=DIV, vmin=-m, vmax=m, aspect="auto")
        dark = lambda v: abs(v) > 0.6 * m
    else:
        m = vmax or max(1e-9, np.nanmax(z))
        im = ax.imshow(z, cmap=SEQ, vmin=0, vmax=m, aspect="auto")
        dark = lambda v: v > 0.55 * m
    for a in range(z.shape[0]):
        for b in range(z.shape[1]):
            lab = fmt.format(z[a, b])
            if float(lab) == 0:
                lab = lab.replace("-", "")
            ax.text(b, a, lab, ha="center", va="center", fontsize=7.5, color="#ffffff" if dark(z[a, b]) else INK)
    ax.set_xticks(range(z.shape[1]), [f"{scale(n1, v):g}" for v in g.columns], fontsize=7.5)
    ax.set_yticks(range(z.shape[0]), [f"{scale(n2, v):g}" for v in g.index], fontsize=7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # outline the published setting
    bx, by = BASE2[(n1, n2)]
    cols, idx = list(np.round(g.columns, 6)), list(np.round(g.index, 6))
    if round(bx, 6) in cols and round(by, 6) in idx:
        ax.add_patch(plt.Rectangle((cols.index(round(bx, 6)) - 0.5, idx.index(round(by, 6)) - 0.5), 1, 1,
                                   fill=False, edgecolor=INK, lw=1.4))
    return im


BASE2 = {("CONS", "QTHRESH"): (0.20, 0.005), ("CONS", "MAXDD"): (0.20, 0.07), ("MAXDD", "DAILY"): (0.03, 0.02),
         ("CONS", "CAP"): (0.20, 0.01)}
XL = {"CONS": "Consistency Limit (%)", "MAXDD": "Trailing Distance (%)"}
I2["pair"] = [tuple(sorted(p)) for p in zip(I2.rule_i, I2.rule_j)]
YL = {"QTHRESH": "Qualifying-Day\nThreshold (%)", "MAXDD": "Trailing\nDistance (%)", "DAILY": "Daily Loss\nLimit (%)",
      "CAP": "Payout Cap (US$)"}


def panel_figure(name, rows, cols, n1, n2, figsize, shared=True):
    """rows: (label, selection dict). cols: (title, source, value column, extra selection, kind, mult, fmt).
    shared=True: one colour scale per column, so rows can be compared."""
    fig, axes = plt.subplots(len(rows), len(cols), figsize=figsize, constrained_layout=True, squeeze=False)
    for j, (title, src, val, csel, kind, mult, fmt) in enumerate(cols):
        grids = [grid_of(S2 if src == "S" else I2, val, **rsel, **csel) for _, rsel in rows]
        vmax = max(np.nanmax(np.abs(g.values)) for g in grids) * mult if shared else None
        for i, (rlabel, rsel) in enumerate(rows):
            ax = axes[i, j]
            g = grids[i]
            heat(ax, g, kind, n1, n2, fmt=fmt, mult=mult, vmax=vmax)
            if i == 0:
                ax.set_title(title, fontsize=8.8, fontweight="bold")
            if i == len(rows) - 1:
                ax.set_xlabel(XL[n1], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{rlabel}\n\n{YL[n2]}", fontsize=8, color=INK)
    save(fig, name)


sp = dict(metric="p_payout")
# Figure 4: ruleset C, consistency limit with the qualifying-day threshold and with the trailing distance
for name, sweep, n2, second, sname in [("Nguyen_Figure4", "CONSxQTHRESH", "QTHRESH", "min_days", "Qualifying Days"),
                                       ("Nguyen_Figure5", "CONSxMAXDD", "MAXDD", "trailing", "Trailing Drawdown")]:
    rows = [(f"{TRADERS[t]}, {MODES[m]}", dict(sweep=sweep, trader=t, trader_mode=m, risk=0.005))
            for t, m in [("zero_edge_EURUSD", "naive"), ("zero_edge_EURUSD", "rule_aware"),
                         ("edge_+0.10R_EURUSD", "naive"), ("trend_XAUUSD", "naive")]]
    cols = [("Paid With All Rules (%)", "S", "baseline", dict(sp, rule="consistency"), "seq", 100, "{:.0f}"),
            ("Shapley Value: Consistency Rule", "S", "shapley_gain", dict(sp, rule="consistency"), "seq", 100, "{:.0f}"),
            (f"Shapley Value: {sname}", "S", "shapley_gain", dict(sp, rule=second), "seq", 100, "{:.1f}"),
            ("Interaction Index", "I", "interaction", dict(sp, pair=tuple(sorted([second, "consistency"]))), "div", 100, "{:.1f}")]
    panel_figure(name, rows, cols, "CONS", n2, (11, 8.6))
print("fig4-5 saved")

# Figure 6: ruleset B, trailing distance with the daily loss limit, 1% risk
rows = [(f"{TRADERS[t]}, {MODES[m]}", dict(sweep="MAXDDxDAILY", trader=t, trader_mode=m, risk=0.01))
        for t, m in [("edge_+0.10R_EURUSD", "naive"), ("edge_+0.10R_EURUSD", "rule_aware"),
                     ("trend_XAUUSD", "naive"), ("trend_XAUUSD", "rule_aware")]]
cols = [("Paid With All Rules (%)", "S", "baseline", dict(sp, rule="trailing"), "seq", 100, "{:.0f}"),
        ("Shapley Value: Trailing Drawdown", "S", "shapley_gain", dict(sp, rule="trailing"), "seq", 100, "{:.1f}"),
        ("Shapley Value: Daily Loss Limit", "S", "shapley_gain", dict(sp, rule="daily"), "div", 100, "{:.1f}"),
        ("Interaction Index", "I", "interaction", dict(sp, pair=("daily", "trailing")), "div", 100, "{:.1f}")]
panel_figure("Nguyen_Figure6", rows, cols, "MAXDD", "DAILY", (11, 8.6))

# Figure 7: ruleset C, consistency limit with the payout cap, expected first payout in dollars, 0.5% risk
se = dict(metric="e_payout")
rows = [(f"{TRADERS[t]}, Naive", dict(sweep="CONSxCAP", trader=t, trader_mode="naive", risk=0.005))
        for t in ["zero_edge_EURUSD", "edge_+0.10R_EURUSD", "trend_XAUUSD"]]
cols = [("Expected First Payout (US$)", "S", "baseline", dict(se, rule="cap"), "seq", 1, "{:.0f}"),
        ("Shapley Value: Payout Cap (US$)", "S", "shapley_gain", dict(se, rule="cap"), "seq", 1, "{:.0f}"),
        ("Shapley Value: Consistency Rule (US$)", "S", "shapley_gain", dict(se, rule="consistency"), "div", 1, "{:.0f}"),
        ("Interaction Index (US$)", "I", "interaction", dict(se, pair=("cap", "consistency")), "div", 1, "{:.0f}")]
panel_figure("Nguyen_Figure7", rows, cols, "CONS", "CAP", (11, 6.6), shared=False)
print("fig6-7 saved")
