import numpy as np, pandas as pd
from data import load_grid, random_library
out = []
for sym in ["XAUUSD", "EURUSD"]:
    g = load_grid(sym, "data_combined")
    R, M, F, E = random_library(g)
    days = pd.DatetimeIndex(g["days"]); nd = len(days)
    sel = days.year <= 2019; hold = ~sel
    dc = g["c"][:, 95]; do = g["o"][:, 0]
    def ma(n):
        s = pd.Series(dc).rolling(n).mean().shift(1).values
        return s
    sig = {}
    for n in [5, 10, 20, 50, 100, 200]:
        m = ma(n)
        for bar in [40, 56, 62]:
            px = g["o"][:, bar]
            sig[f"trend{n}_b{bar}"] = (bar, np.where(np.isnan(m), 0, np.sign(px - m)))
            sig[f"trendlong{n}_b{bar}"] = (bar, np.where(np.isnan(m), 0, (px > m).astype(float)))
    for bar in [40, 56, 62]:
        px = g["o"][:, bar]
        sig[f"asiancont_b{bar}"] = (bar, np.sign(px - do))
        sig[f"asianrev_b{bar}"] = (bar, -np.sign(px - do))
        pd_ = np.concatenate([[0], np.sign(dc[:-1] - do[:-1])])
        sig[f"prevday_b{bar}"] = (bar, pd_)
        sig[f"prevdayrev_b{bar}"] = (bar, -pd_)
        sig[f"long_b{bar}"] = (bar, np.ones(nd))
    for k, (bar, s) in sig.items():
        side = np.where(s > 0, 0, 1)
        r = R[np.arange(nd), bar, side]
        act = s != 0
        def st(mask):
            x = r[mask & act]
            return (x.mean(), x.mean() / x.std() * np.sqrt(len(x)), len(x)) if len(x) > 30 else (np.nan, np.nan, len(x))
        a, b, c = st(sel), st(hold), st(np.ones(nd, bool))
        out.append(dict(sym=sym, sig=k, sel_mean=a[0], sel_t=a[1], hold_mean=b[0], hold_t=b[1], all_mean=c[0], all_t=c[1], n=c[2]))
df = pd.DataFrame(out)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(df.sort_values("sel_mean", ascending=False).head(25).round(3).to_string())
print(df.sort_values("all_mean", ascending=False).head(15).round(3).to_string())
