"""Figure 1 (payout probability by risk) and Figure 2 (Shapley bars at 0.5% risk)."""
import pandas as pd
from matplotlib.lines import Line2D
from figstyle import *

f = pd.read_csv("final_factorial.csv")
f = f[f.removed == "none"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True)
SHORT = {"A_two_phase_static": "A: Two-Phase, Static Drawdown", "B_one_step_eod_trailing": "B: One-Step, End-of-Day Trailing",
         "C_instant_funding": "C: Instant Funding"}
for ax, (ak, an) in zip(axes, SHORT.items()):
    style(ax, left=True)
    for tk in TRADERS:
        for mk, ls in (("naive", "-"), ("rule_aware", "--")):
            d = f[(f.archetype == ak) & (f.trader == tk) & (f.trader_mode == mk)].sort_values("risk")
            ax.plot(d.risk * 100, d.p_payout * 100, ls, color=TRADER_COL[tk], lw=2, marker="o", ms=4.5,
                    markeredgecolor=SURF, markeredgewidth=1)
    ax.set_title(an, loc="left", fontsize=9, fontweight="bold")
    ax.set_xticks([0.25, 0.5, 1.0, 1.5])
    ax.set_xlabel("Risk Per Trade (% of Starting Balance)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylim(0, 100)
axes[0].set_ylabel("Payout Within One Year (%)")
h = [Line2D([], [], color=TRADER_COL[t], lw=2, label=TRADERS[t]) for t in TRADERS]
h += [Line2D([], [], color=MUTED, lw=2, ls="-", label="Naive"), Line2D([], [], color=MUTED, lw=2, ls="--", label="Rule-Aware")]
fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "Nguyen_Figure1")

s = pd.read_csv("final_shapley.csv")
s = s[(s.metric == "p_payout") & (s.risk == 0.005)]
fig, axes = plt.subplots(2, 1, figsize=(9, 8.6), sharex=True)
for ax, ak in zip(axes, ["B_one_step_eod_trailing", "C_instant_funding"]):
    style(ax)
    sub = s[s.archetype == ak]
    labels, ticks, y = [], [], 0.0
    for tk, tn in TRADERS.items():
        for mk, mn in MODES.items():
            row = sub[(sub.trader == tk) & (sub.trader_mode == mk)].set_index("rule")
            left = 0.0
            for r in RULES:
                w = max(row.loc[r, "shapley_gain"], 0) * 100
                if w <= 0.05:
                    continue
                ax.barh(y, w, left=left, height=0.7, color=RULE_COL[r], edgecolor=SURF, linewidth=1.5)
                if w >= 4:
                    ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=8.5,
                            color="#0b0b0b" if r == "min_days" else "#ffffff", fontweight="bold")
                left += w
            b, a = row.baseline.iloc[0] * 100, row.all_removed.iloc[0] * 100
            ax.text(left + 0.7, y, f"{b:.0f}% Paid With Rules, {a:.0f}% Without", va="center", fontsize=8, color=MUTED)
            labels.append(f"{tn}, {mn}")
            ticks.append(y)
            y += 1
        y += 0.4
    ax.set_yticks(ticks, labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_title(ARCH[ak], loc="left", fontsize=10, fontweight="bold", pad=4)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", length=0)
axes[-1].set_xlim(0, 75)
axes[-1].set_xlabel("Percentage Points of Payout Probability Lost to Each Rule (Shapley Value)")
shown = [r for r in RULES if (s[s.rule == r].shapley_gain > 0.0005).any()]
fig.legend([plt.Rectangle((0, 0), 1, 1, color=RULE_COL[r]) for r in shown], [RULE_NAMES[r] for r in shown],
           loc="upper center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.965))
save(fig, "Nguyen_Figure2")
print("saved")
