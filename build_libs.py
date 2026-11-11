"""Build the trade libraries once and save them (libs.npz, or libs_<tag>.npz)."""
import sys
import numpy as np
from data import load_grid, random_library, breakout_library, trend_library, WIN_START, WIN_END

spread_mult = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
tag = sys.argv[2] if len(sys.argv) > 2 else ""
import os
DATA = os.environ.get("PROPSIM_DATA", "data")
eu, xa = load_grid("EURUSD", DATA), load_grid("XAUUSD", DATA)
eu["spread"] *= spread_mult
xa["spread"] *= spread_mult
R, M, F, E = random_library(eu)
Rb, Mb, Fb, hb = breakout_library(xa)
Rt, Mt, Ft, ht = trend_library(xa)
w = slice(WIN_START, WIN_END)
r = R[:, w, :]
best = np.maximum(r[..., 0], r[..., 1]).mean()
edge = (0.10 - r.mean()) / (best - r.mean())
print(f"days EURUSD {len(eu['days'])} ({eu['days'][0]} to {eu['days'][-1]}), XAUUSD {len(xa['days'])}")
print(f"random mean R {r.mean():.4f}; edge prob for +0.10R {edge:.4f}; breakout mean R {Rb[hb].mean():.4f} over {hb.sum()} trades")
print(f"gold trend mean R {Rt[ht].mean():.4f} over {ht.sum()} trades, win rate {(Rt[ht] > 0).mean():.3f}, stopped {(Rt[ht] <= -0.99).mean():.3f}")
extra = {}
if tag == "":
    # target variants for the policy search: the same entries with a profit target
    for rr in (1.0, 2.0, 3.0):
        a = trend_library(xa, rr=rr)
        extra.update({f"Rt{rr:g}": a[0], f"Mt{rr:g}": a[1], f"Ft{rr:g}": a[2], f"ht{rr:g}": a[3]})
        print(f"gold trend with a {rr:g}R target: mean R {a[0][a[3]].mean():.4f}")
    for rr in (1.0, 3.0):
        a = breakout_library(xa, rr=rr)
        extra.update({f"Rb{rr:g}": a[0], f"Mb{rr:g}": a[1], f"Fb{rr:g}": a[2], f"hb{rr:g}": a[3]})
        print(f"gold breakout with a {rr:g}R target: mean R {a[0][a[3]].mean():.4f}")
np.savez_compressed(f"libs{tag}.npz", R=R, M=M, F=F, E=E, Rb=Rb, Mb=Mb, Fb=Fb, hb=hb, Rt=Rt, Mt=Mt, Ft=Ft, ht=ht, edge=edge, **extra)
