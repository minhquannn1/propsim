import numpy as np, pandas as pd
from numba import njit
from data import load_grid, FLAT_BAR, BARS   # run with the price files in data_combined/

@njit(cache=True)
def day_library(o, h, l, side, entry_bar, stopdist, spread, rr):
    """One trade a day: enter at the open of entry_bar in direction side[d]
    (+1 long, -1 short, 0 no trade), stop stopdist[d] away, optional target
    rr*stop (rr<=0: none), flat at the open of FLAT_BAR. Stop checked first."""
    nd = o.shape[0]
    R = np.zeros(nd); MAE = np.zeros(nd); MFE = np.zeros(nd)
    has = np.zeros(nd, dtype=np.bool_)
    for d in range(nd):
        sg = side[d]; sl = stopdist[d]
        if sg == 0 or not (sl > 0):
            continue
        entry = o[d, entry_bar]
        cost = spread / sl
        mae = 0.0; mfe = 0.0; res = 0.0; done = False
        for j in range(entry_bar, FLAT_BAR):
            adverse = (l[d, j] - entry) / sl if sg > 0 else (entry - h[d, j]) / sl
            if adverse <= -1.0:
                mae = -1.0; res = -1.0; done = True
                break
            if adverse < mae:
                mae = adverse
            fav = (h[d, j] - entry) / sl if sg > 0 else (entry - l[d, j]) / sl
            if rr > 0 and fav >= rr:
                res = rr; mfe = rr; done = True
                break
            if fav > mfe:
                mfe = fav
        if not done:
            res = sg * (o[d, FLAT_BAR] - entry) / sl
        R[d] = res - cost; MAE[d] = mae - cost; MFE[d] = max(mfe - cost, res - cost)
        has[d] = True
    return R, MAE, MFE, has

def daily_feats(g):
    dc = g["c"][:, BARS - 1]; dh = g["h"].max(1); dl = g["l"].min(1)
    pc = np.concatenate([[dc[0]], dc[:-1]])
    tr = np.maximum(dh - dl, np.maximum(abs(dh - pc), abs(dl - pc)))
    atr = pd.Series(tr).rolling(20).mean().shift(1).values      # known before the day starts
    return dc, atr

if __name__ == "__main__":
    out = []
    for sym in ["XAUUSD", "EURUSD"]:
        g = load_grid(sym, "data_combined")
        days = pd.DatetimeIndex(g["days"]); nd = len(days)
        sel = days.year <= 2019
        dc, atr = daily_feats(g)
        print(sym, "first close", dc[0], "last close", dc[-1], "median daily ATR", np.nanmedian(atr), "spread", g["spread"])
        for n in [20, 50, 100, 200]:
            ma = pd.Series(dc).rolling(n).mean().shift(1).values
            prev = np.concatenate([[np.nan], dc[:-1]])
            up = prev > ma; ok = ~np.isnan(ma) & ~np.isnan(atr)
            for mode in ["longshort", "longonly"]:
                side = np.where(up, 1, -1 if mode == "longshort" else 0); side = np.where(ok, side, 0)
                for bar in [8, 40]:
                    for k in [0.5, 1.0, 2.0]:
                        R, M, F, has = day_library(g["o"], g["h"], g["l"], side.astype(np.int64), bar, np.nan_to_num(k * atr), g["spread"], 0.0)
                        def st(m):
                            x = R[has & m]
                            return x.mean(), x.mean() / x.std() * np.sqrt(len(x)), len(x)
                        a, b, c = st(sel), st(~sel), st(np.ones(nd, bool))
                        out.append(dict(sym=sym, sig=f"ma{n}_{mode}_b{bar}_k{k}", sel_mean=a[0], sel_t=a[1], hold_mean=b[0], hold_t=b[1],
                                        all_mean=c[0], all_t=c[1], n=c[2], win=(R[has] > 0).mean(), stop=(R[has] <= -0.99).mean()))
    df = pd.DataFrame(out)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(df.sort_values("sel_mean", ascending=False).head(30).round(3).to_string())
    df.to_csv("strategy_search.csv", index=False)
