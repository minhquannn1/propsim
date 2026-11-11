"""Load M15 bars, put them on a fixed 96-bar daily grid, and build trade libraries.

Times in the source files are broker server time (GMT+2/+3, day starts at the
New York 17:00 close). Server midnight is used as the daily reset for every
ruleset.
"""
import numpy as np
import pandas as pd
from numba import njit

BARS = 96               # M15 bars per day
WIN_START = 40          # 10:00 server = London open (approx.)
WIN_END = 76            # 19:00 server: last bar a new random trade can open
FLAT_BAR = 92           # 23:00 server: all positions closed at this bar's open

# price scale in the csv files and typical retail spread in raw units
SYMBOLS = {
    "EURUSD": dict(scale=1e5, spread=10.0),   # 1.0 pip
    "XAUUSD": dict(scale=1e2, spread=30.0),   # $0.30
}


def load_grid(symbol, path="data"):
    df = pd.read_csv(f"{path}/{symbol}m15.csv", parse_dates=["Date"])
    df = df[df.Date.dt.dayofweek < 5]
    df["day"] = df.Date.dt.normalize()
    df["slot"] = df.Date.dt.hour * 4 + df.Date.dt.minute // 15
    counts = df.groupby("day").size()
    good = counts[counts >= 80].index
    df = df[df.day.isin(good)]
    days = np.array(sorted(df.day.unique()))
    nd = len(days)
    o = np.full((nd, BARS), np.nan)
    h, l, c = o.copy(), o.copy(), o.copy()
    di = np.searchsorted(days, df.day.values)
    s = df.slot.values
    o[di, s], h[di, s], l[di, s], c[di, s] = df.open, df.high, df.low, df.close
    # forward-fill missing bars as flat bars at the previous close
    for d in range(nd):
        last = np.nan
        for k in range(BARS):
            if np.isnan(o[d, k]):
                if np.isnan(last):
                    nxt = np.where(~np.isnan(o[d]))[0][0]
                    last = o[d, nxt]
                o[d, k] = h[d, k] = l[d, k] = c[d, k] = last
            last = c[d, k]
    # ATR(14) on M15, continuous across days, lagged one bar (no look-ahead)
    H, L, C = h.ravel(), l.ravel(), c.ravel()
    prevc = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H - L, np.maximum(abs(H - prevc), abs(L - prevc)))
    atr = pd.Series(tr).rolling(14).mean().shift(1).bfill().values.reshape(nd, BARS)
    return dict(days=days, o=o, h=h, l=l, c=c, atr=atr, **SYMBOLS[symbol])


@njit(cache=True)
def _random_library(o, h, l, c, atr, spread, k_sl, rr):
    """For every (day, entry bar, direction): net R, worst open R (MAE), best open R (MFE), exit bar."""
    nd = o.shape[0]
    R = np.zeros((nd, BARS, 2))
    MAE = np.zeros((nd, BARS, 2))
    MFE = np.zeros((nd, BARS, 2))
    EX = np.zeros((nd, BARS, 2), dtype=np.int64)
    for d in range(nd):
        for i in range(WIN_START, WIN_END):
            sl = k_sl * atr[d, i]
            cost = spread / sl
            for side in range(2):
                sgn = 1.0 if side == 0 else -1.0
                entry = o[d, i]
                stop = entry - sgn * sl
                tgt = entry + sgn * rr * sl
                mae = 0.0
                mfe = 0.0
                res = 0.0
                ex = FLAT_BAR - 1
                done = False
                for j in range(i, FLAT_BAR):
                    adverse = (l[d, j] - entry) / sl if sgn > 0 else (entry - h[d, j]) / sl
                    if adverse <= -1.0:          # stop checked first (conservative)
                        mae = -1.0
                        res = -1.0
                        ex = j
                        done = True
                        break
                    if adverse < mae:
                        mae = adverse
                    fav = (h[d, j] - entry) / sl if sgn > 0 else (entry - l[d, j]) / sl
                    if fav >= rr:
                        res = rr
                        mfe = rr
                        ex = j
                        done = True
                        break
                    if fav > mfe:
                        mfe = fav
                if not done:
                    res = sgn * (o[d, FLAT_BAR] - entry) / sl
                R[d, i, side] = res - cost
                MAE[d, i, side] = mae - cost
                MFE[d, i, side] = max(mfe - cost, res - cost)
                EX[d, i, side] = ex
    return R, MAE, MFE, EX


def random_library(g, k_sl=3.0, rr=1.5, costs=True):
    return _random_library(g["o"], g["h"], g["l"], g["c"], g["atr"],
                           g["spread"] if costs else 0.0, k_sl, rr)


@njit(cache=True)
def _breakout_library(o, h, l, c, spread, rr):
    """Asian-range breakout: range = 00:00-10:00 server. First break between
    10:00 and 19:00 enters on a stop order at the range edge, stop at range
    midpoint, target rr*R, flat at 23:00. One trade a day at most."""
    nd = o.shape[0]
    R = np.zeros(nd)
    MAE = np.zeros(nd)
    MFE = np.zeros(nd)
    has = np.zeros(nd, dtype=np.bool_)
    for d in range(nd):
        hi = h[d, 0]
        lo = l[d, 0]
        for k in range(1, WIN_START):
            hi = max(hi, h[d, k])
            lo = min(lo, l[d, k])
        mid = 0.5 * (hi + lo)
        sl = hi - mid
        if sl <= 0:
            continue
        side = 0
        start = -1
        for j in range(WIN_START, WIN_END):
            up = h[d, j] > hi
            dn = l[d, j] < lo
            if up and dn:
                break                      # ambiguous bar, skip the day
            if up:
                side = 1
                start = j
                break
            if dn:
                side = -1
                start = j
                break
        if start < 0:
            continue
        entry = hi if side > 0 else lo
        cost = spread / sl
        mae = 0.0
        mfe = 0.0
        res = 0.0
        done = False
        for j in range(start, FLAT_BAR):
            # on the entry bar a stop touch is counted as a loss (order unknown)
            adverse = (l[d, j] - entry) / sl if side > 0 else (entry - h[d, j]) / sl
            if adverse <= -1.0:
                mae = -1.0
                res = -1.0
                done = True
                break
            if adverse < mae:
                mae = adverse
            fav = (h[d, j] - entry) / sl if side > 0 else (entry - l[d, j]) / sl
            if fav >= rr and j > start:
                res = rr
                mfe = rr
                done = True
                break
            if j > start and fav > mfe:
                mfe = fav
        if not done:
            res = side * (o[d, FLAT_BAR] - entry) / sl
        R[d] = res - cost
        MAE[d] = mae - cost
        MFE[d] = max(mfe - cost, res - cost)
        has[d] = True
    return R, MAE, MFE, has


def breakout_library(g, rr=2.0, costs=True):
    return _breakout_library(g["o"], g["h"], g["l"], g["c"],
                             g["spread"] if costs else 0.0, rr)


@njit(cache=True)
def _day_library(o, h, l, side, entry_bar, stopdist, spread, rr):
    """One trade a day: enter at the open of entry_bar in direction side[d]
    (+1 long, -1 short, 0 no trade) with a stop stopdist[d] away and a target
    at rr times the stop distance (rr <= 0: no target); flat at the open of
    FLAT_BAR. The stop is checked first."""
    nd = o.shape[0]
    R = np.zeros(nd)
    MAE = np.zeros(nd)
    MFE = np.zeros(nd)
    has = np.zeros(nd, dtype=np.bool_)
    for d in range(nd):
        sg = side[d]
        sl = stopdist[d]
        if sg == 0 or not (sl > 0):
            continue
        entry = o[d, entry_bar]
        cost = spread / sl
        mae = 0.0
        mfe = 0.0
        res = 0.0
        done = False
        for j in range(entry_bar, FLAT_BAR):
            adverse = (l[d, j] - entry) / sl if sg > 0 else (entry - h[d, j]) / sl
            if adverse <= -1.0:
                mae = -1.0
                res = -1.0
                done = True
                break
            if adverse < mae:
                mae = adverse
            fav = (h[d, j] - entry) / sl if sg > 0 else (entry - l[d, j]) / sl
            if rr > 0 and fav >= rr:
                res = rr
                mfe = rr
                done = True
                break
            if fav > mfe:
                mfe = fav
        if not done:
            res = sg * (o[d, FLAT_BAR] - entry) / sl
        R[d] = res - cost
        MAE[d] = mae - cost
        MFE[d] = max(mfe - cost, res - cost)
        has[d] = True
    return R, MAE, MFE, has


def daily_features(g):
    """Daily close and a 20-day average true range, both known before the day starts."""
    dc = g["c"][:, BARS - 1]
    dh, dl = g["h"].max(1), g["l"].min(1)
    pc = np.concatenate([[dc[0]], dc[:-1]])
    tr = np.maximum(dh - dl, np.maximum(abs(dh - pc), abs(dl - pc)))
    atr = pd.Series(tr).rolling(20).mean().shift(1).values
    return dc, atr


def trend_signal(g, n_ma=50, long_only=False):
    """+1 if yesterday's close is above the average of the n_ma closes before
    today, -1 (or 0 if long_only) if below, 0 while the average is undefined."""
    dc, atr = daily_features(g)
    ma = pd.Series(dc).rolling(n_ma).mean().shift(1).values
    prev = np.concatenate([[np.nan], dc[:-1]])
    ok = ~np.isnan(ma) & ~np.isnan(atr)
    side = np.where(prev > ma, 1, 0 if long_only else -1)
    return np.where(ok, side, 0).astype(np.int64), atr


def trend_library(g, n_ma=50, entry_bar=WIN_START, k=0.5, long_only=False, costs=True, rr=0.0):
    """Daily trend trade: at 10:00 server go with the n_ma-day trend, stop
    k x 20-day ATR away, no target unless rr > 0, flat at 23:00. Uses past data only."""
    side, atr = trend_signal(g, n_ma, long_only)
    return _day_library(g["o"], g["h"], g["l"], side, entry_bar, np.nan_to_num(k * atr),
                        g["spread"] if costs else 0.0, rr)
