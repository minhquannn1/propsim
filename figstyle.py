"""Shared figure style (one typeface, title-free figures, title-case labels)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURF, INK, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9"
RULES = ["consistency", "trailing", "min_days", "daily", "cap"]
RULE_NAMES = {"consistency": "Consistency Rule", "trailing": "Trailing Drawdown", "daily": "Daily Loss Limit",
              "min_days": "Minimum or Qualifying Days", "cap": "Payout Cap",
              "evaluation": "Evaluation (Profit Targets)", "max_loss": "Maximum Loss"}
RULE_COL = {"consistency": "#2a78d6", "trailing": "#eb6834", "min_days": "#eda100", "daily": "#1baf7a",
            "cap": "#e87ba4", "evaluation": "#008300", "max_loss": "#6250d6"}
TRADERS = {"zero_edge_EURUSD": "No Edge (EURUSD)", "edge_+0.10R_EURUSD": "Small Edge (EURUSD)",
           "breakout_XAUUSD": "Gold Breakout", "trend_XAUUSD": "Gold Trend"}
TRADER_COL = {"zero_edge_EURUSD": "#2a78d6", "edge_+0.10R_EURUSD": "#eb6834",
              "breakout_XAUUSD": "#1baf7a", "trend_XAUUSD": "#eda100"}
ARCH = {"A_two_phase_static": "A: Two-Phase Evaluation, Static Drawdown",
        "B_one_step_eod_trailing": "B: One-Step Evaluation, End-of-Day Trailing",
        "C_instant_funding": "C: Instant Funding"}
MODES = {"naive": "Naive", "rule_aware": "Rule-Aware"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c0312f"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def style(ax, left=False):
    for sp in ["top", "right"] + ([] if left else ["left"]):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if left:
        ax.spines["left"].set_color(GRID)
    ax.set_axisbelow(True)


def save(fig, name):
    from PIL import Image
    fig.savefig(f"figures/{name}.png", dpi=200)
    Image.open(f"figures/{name}.png").convert("RGB").save(f"figures/{name}.jpg", quality=95, dpi=(200, 200))
    plt.close(fig)
