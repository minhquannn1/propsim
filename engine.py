"""Monte Carlo engine: one simulated trader account per path, day by day.

All money is in units of the initial balance (1.0 = $100k on a $100k account).
Random numbers come from a counter-based hash of (seed, path, day, slot), so
every rule combination sees exactly the same days and the same trade choices
(common random numbers). Differences between combinations are then caused by
the rules alone, not by sampling noise.
"""
import numpy as np
from numba import njit, prange

from data import WIN_START, WIN_END

# ---- ruleset vector layout -------------------------------------------------
N_EVAL, TGT1, TGT2, EVAL_MIN_DAYS, EVAL_CONS = 0, 1, 2, 3, 4
DAILY, DAILY_BASE, MAXDD, TRAIL_MODE, TRAIL_LOCK = 5, 6, 7, 8, 9
F_MIN_PROFIT, F_MIN_QDAYS, F_QTHRESH, F_CONS, SPLIT, CAP = 10, 11, 12, 13, 14, 15
DAILY_SOFT, F_FRAC = 16, 17
CFG_LEN = 18
# TRAIL_MODE: 0 static from initial balance, 1 trails closed balance intraday,
#             2 trails end-of-day balance, 3 trails peak equity (open profit
#             counts). TRAIL_LOCK=1: floor stops at the initial balance.
# DAILY_SOFT=1: hitting the daily limit closes the trade at the limit and ends
#             the day, but the account survives (futures-style).
# Payout = min(SPLIT * F_FRAC * profit, CAP); F_FRAC < 1 means only part of
#             the profit can be withdrawn at once.
# DAILY_BASE: 0 limit measured from initial balance, 1 from the day's start balance
# F_QTHRESH < 0 means any day with a trade counts as a qualifying day.

OUT_CENSORED, OUT_PAYOUT, OUT_FAIL_DAILY, OUT_FAIL_MAXDD = 0, 1, 2, 3


@njit(inline="always")
def _u01(seed, a, b, c):
    x = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
    x ^= np.uint64(a) * np.uint64(0xBF58476D1CE4E5B9)
    x ^= np.uint64(b) * np.uint64(0x94D049BB133111EB)
    x ^= np.uint64(c) * np.uint64(0xD6E8FEB86659FD93)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(parallel=True, cache=True)
def simulate(cfg, ttype, risk, edge, ntr, R, MAE, MFE, EX, Rb, MAEb, MFEb, hasb,
             sims, H, seed, block, wgoal, smart=0, room_frac=0.25, dstop_mult=1.0, qstop=1):
    """ttype 0: random-entry trader on the R/MAE/EX library. With probability
    `edge` it picks the side that turns out better (a stand-in for skill).
    ttype 1: the breakout library (Rb/MAEb/hasb), one trade a day.
    wgoal: the trader asks for a payout once profit reaches this level (and
    the firm's own conditions are met). Fixed across rule combinations so that
    removing a rule cannot make the trader cash out earlier for less.
    smart=1: a rule-aware trader. It (a) stops for the day once the day's gain
    reaches the largest best day the active consistency rule would allow at the
    smallest possible payout, (b) stops for the day once the day qualifies when
    qualifying days are still needed, and (c) never risks more than room_frac
    of the distance to the nearest loss floor, skipping the day when that is
    under a tenth of normal risk. It only reacts to rules that are switched on.
    dstop_mult scales the day-stop in (a) (1e9 switches it off), qstop=0
    switches (b) off, and a very large room_frac switches (c) off, so the
    rule-aware policy can be searched over a grid."""
    nd = R.shape[0] if ttype == 0 else Rb.shape[0]
    outcome = np.zeros(sims, dtype=np.int64)
    payout = np.zeros(sims)
    used = np.zeros(sims, dtype=np.int64)
    stage = np.zeros(sims, dtype=np.int64)
    n_eval = int(cfg[N_EVAL])
    min_profit = max(cfg[F_MIN_PROFIT], wgoal)
    no_eval_cons = cfg[EVAL_CONS] >= 0.999
    no_f_cons = cfg[F_CONS] >= 0.999
    for s in prange(sims):
        phase = 0
        bal = 1.0
        hwm = 1.0
        traded_days = 0
        qdays = 0
        best_day = 0.0
        res = OUT_CENSORED
        t_used = H
        for t in range(H):
            blk = t // block
            start = int(_u01(seed, s, blk, 999983) * (nd - block))
            d = start + (t % block)
            funded = phase >= n_eval
            target = cfg[TGT1] if phase == 0 else cfg[TGT2]
            day_start = bal
            if cfg[DAILY_BASE] == 0:
                daily_floor = day_start - cfg[DAILY]      # % of initial, from day start
            else:
                daily_floor = day_start * (1.0 - cfg[DAILY])
            traded = False
            failed = 0
            dstop = 1e9
            if smart == 1:
                if funded:
                    if (not no_f_cons) and dstop_mult < 1e8:
                        need = max(min_profit, cfg[F_MIN_QDAYS] * max(cfg[F_QTHRESH], 0.0))
                        dstop = dstop_mult * cfg[F_CONS] * need
                    if qstop == 1 and cfg[F_QTHRESH] > 0 and qdays < cfg[F_MIN_QDAYS]:
                        dstop = min(dstop, cfg[F_QTHRESH] * day_start)
                elif (not no_eval_cons) and dstop_mult < 1e8:
                    dstop = dstop_mult * cfg[EVAL_CONS] * target
            # ---- trading for the day ------------------------------------
            if ((not funded) and (bal - 1.0 >= target)
                    and (no_eval_cons or best_day <= cfg[EVAL_CONS] * (bal - 1.0))):
                traded = True        # target met and consistency satisfied: a token trade to tick a day
            else:
                ntrades = ntr if ttype == 0 else 1
                pos = WIN_START + int(_u01(seed, s, t, 0) * (WIN_END - WIN_START))
                for k in range(ntrades):
                    if ttype == 0:
                        if pos >= WIN_END:
                            break
                        if _u01(seed, s, t, 100 + k) < edge:
                            side = 0 if R[d, pos, 0] >= R[d, pos, 1] else 1
                        else:
                            side = 0 if _u01(seed, s, t, 200 + k) < 0.5 else 1
                        rv = R[d, pos, side]
                        mv = MAE[d, pos, side]
                        fv = MFE[d, pos, side]
                        ex = EX[d, pos, side]
                    else:
                        if not hasb[d]:
                            break
                        rv = Rb[d]
                        mv = MAEb[d]
                        fv = MFEb[d]
                        ex = WIN_END
                    if cfg[TRAIL_MODE] == 0:
                        ddfloor = 1.0 - cfg[MAXDD]
                    else:
                        ddfloor = hwm - cfg[MAXDD]
                        if cfg[TRAIL_LOCK] > 0 and ddfloor > 1.0:
                            ddfloor = 1.0
                    floor = max(daily_floor, ddfloor)
                    r = risk
                    if smart == 1:
                        r = min(risk, room_frac * (bal - floor))
                        if r < 0.1 * risk:
                            break                      # too close to a floor: sit out the day
                    if bal + r * mv < floor - 1e-12:
                        if cfg[DAILY_SOFT] > 0 and daily_floor >= ddfloor:
                            bal = daily_floor              # liquidated at the limit, day over
                            traded = True
                            break
                        failed = OUT_FAIL_DAILY if daily_floor >= ddfloor else OUT_FAIL_MAXDD
                        break
                    peak = bal + r * fv
                    bal += r * rv
                    traded = True
                    if cfg[TRAIL_MODE] == 1 and bal > hwm:
                        hwm = bal
                    if cfg[TRAIL_MODE] == 3 and peak > hwm:
                        hwm = peak
                    if cfg[TRAIL_MODE] == 3:
                        nf = hwm - cfg[MAXDD]
                        if cfg[TRAIL_LOCK] > 0 and nf > 1.0:
                            nf = 1.0
                        if bal < nf - 1e-12:          # gave back more than the trail after a new equity high
                            failed = OUT_FAIL_MAXDD
                            break
                    if smart == 1 and bal - day_start >= dstop:
                        break                          # bank the day
                    # stop for the day once the stage goal is met
                    profit = bal - 1.0
                    if not funded:
                        if profit >= target:
                            break
                    else:
                        dp = bal - day_start
                        qd = qdays
                        if cfg[F_QTHRESH] < 0 or dp >= cfg[F_QTHRESH] * day_start:
                            qd += 1
                        bd = max(best_day, dp)
                        if (profit >= min_profit and qd >= cfg[F_MIN_QDAYS]
                                and (no_f_cons or bd <= cfg[F_CONS] * profit)):
                            break
                    if ttype == 0:
                        nxt = ex + 1
                        if nxt >= WIN_END:
                            break
                        pos = nxt + int(_u01(seed, s, t, 300 + k) * (WIN_END - nxt))
            if failed > 0:
                res = failed
                t_used = t + 1
                break
            # ---- end of day bookkeeping --------------------------------
            dp = bal - day_start
            if traded:
                traded_days += 1
            if dp > best_day:
                best_day = dp
            if traded and (cfg[F_QTHRESH] < 0 or dp >= cfg[F_QTHRESH] * day_start):
                qdays += 1
            if cfg[TRAIL_MODE] == 2 and bal > hwm:
                hwm = bal
            profit = bal - 1.0
            if not funded:
                if (profit >= target and traded_days >= cfg[EVAL_MIN_DAYS]
                        and (no_eval_cons or best_day <= cfg[EVAL_CONS] * profit)):
                    phase += 1                       # next stage, fresh account
                    bal = 1.0
                    hwm = 1.0
                    traded_days = 0
                    qdays = 0
                    best_day = 0.0
            else:
                if (profit >= min_profit and profit > 0 and qdays >= cfg[F_MIN_QDAYS]
                        and (no_f_cons or best_day <= cfg[F_CONS] * profit)):
                    res = OUT_PAYOUT
                    payout[s] = min(cfg[SPLIT] * cfg[F_FRAC] * profit, cfg[CAP])
                    t_used = t + 1
                    break
        outcome[s] = res
        used[s] = t_used
        stage[s] = phase
    return outcome, payout, used, stage
