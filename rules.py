"""Ruleset archetypes and the rule switches used in the factorial design.

Parameters were checked against firms' published terms on 5 October 2026
(sources in the paper). They are archetypes, not copies of any single firm.
A: two-phase forex evaluation (FTMO, FundedNext, FundingPips, The5ers).
B: one-step futures-style evaluation (Topstep, Tradeify, Apex).
C: instant funding (Upcomers Vanguard, FundingPips Zero, Goat).
"""
import numpy as np
from engine import (CFG_LEN, N_EVAL, TGT1, TGT2, EVAL_MIN_DAYS, EVAL_CONS, DAILY,
                    DAILY_BASE, MAXDD, TRAIL_MODE, TRAIL_LOCK, F_MIN_PROFIT,
                    F_MIN_QDAYS, F_QTHRESH, F_CONS, SPLIT, CAP, DAILY_SOFT, F_FRAC)

BIG = 1e9
ACCOUNT = 100_000


def _cfg(**kw):
    c = np.zeros(CFG_LEN)
    c[EVAL_CONS] = c[F_CONS] = 1.0
    c[CAP] = c[DAILY] = BIG
    c[SPLIT] = 1.0
    c[F_FRAC] = 1.0
    c[F_QTHRESH] = -1.0
    for k, v in kw.items():
        c[globals()[k]] = v
    return c


ARCHETYPES = {
    # FTMO 2-Step: 10% / 5% targets, 5% daily from initial, 10% static, 4 days per
    # phase, first payout from day 14 (~10 trading days), 80% split, no cap.
    # Fee: FundedNext $549.99, FundingPips $529 -> $540.
    "A_two_phase_static": dict(
        label="A: two-phase evaluation, static drawdown",
        fee=540.0,
        cfg=_cfg(N_EVAL=2, TGT1=0.10, TGT2=0.05, EVAL_MIN_DAYS=4,
                 DAILY=0.05, DAILY_BASE=0, MAXDD=0.10, TRAIL_MODE=0,
                 F_MIN_PROFIT=1e-9, F_MIN_QDAYS=10, F_QTHRESH=-1, SPLIT=0.8),
        rules=["daily", "min_days"],
    ),
    # Topstep / Tradeify / Apex $100k: $6,000 target, $3,000 end-of-day trailing
    # loss that locks at the start balance, $2,000 daily limit that liquidates
    # for the day without failing (Topstep), best day <= 50% in the evaluation
    # (Topstep 55%, Tradeify 40%, TPT 50%), payout after 5 winning days of $150+,
    # payout <= 50% of profit and <= $3,000 (Topstep), 90% split. Fee ~$150.
    "B_one_step_eod_trailing": dict(
        label="B: one-step evaluation, end-of-day trailing drawdown",
        fee=150.0,
        cfg=_cfg(N_EVAL=1, TGT1=0.06, EVAL_MIN_DAYS=2, EVAL_CONS=0.50,
                 DAILY=0.02, DAILY_BASE=0, DAILY_SOFT=1, MAXDD=0.03, TRAIL_MODE=2, TRAIL_LOCK=1,
                 F_MIN_PROFIT=0.01, F_MIN_QDAYS=5, F_QTHRESH=0.0015,
                 SPLIT=0.9, F_FRAC=0.5, CAP=0.027),
        rules=["daily", "trailing", "min_days", "consistency", "cap"],
    ),
    # Upcomers Vanguard $100k: 4% daily from the day's start balance, 7% trailing
    # on peak equity that locks at the initial balance, 6 days of >= 0.5%,
    # >= 1% profit, best day <= 20% (soft), 90% default split, $1,000 first payout
    # cap (firm support reply, Sept 2026). Fee $287.88 (author's purchase).
    "C_instant_funding": dict(
        label="C: instant funding, no evaluation",
        fee=287.88,
        cfg=_cfg(N_EVAL=0, DAILY=0.04, DAILY_BASE=1, MAXDD=0.07, TRAIL_MODE=3, TRAIL_LOCK=1,
                 F_MIN_PROFIT=0.01, F_MIN_QDAYS=6, F_QTHRESH=0.005, F_CONS=0.20,
                 SPLIT=0.9, CAP=0.01),
        rules=["daily", "trailing", "min_days", "consistency", "cap"],
    ),
}


def relax(cfg, removed):
    """Return a copy of cfg with each rule in `removed` switched off."""
    c = cfg.copy()
    for r in removed:
        if r == "daily":
            c[DAILY] = BIG
        elif r == "trailing":
            c[TRAIL_MODE] = 0          # same distance, but static from initial balance
            c[TRAIL_LOCK] = 0
        elif r == "min_days":
            c[EVAL_MIN_DAYS] = 0
            c[F_MIN_QDAYS] = 0
        elif r == "consistency":
            c[EVAL_CONS] = 1.0
            c[F_CONS] = 1.0
        elif r == "cap":
            c[CAP] = BIG               # no cap on the amount
            c[F_FRAC] = 1.0            # and the whole profit can be withdrawn
        elif r == "max_loss":
            c[MAXDD] = BIG             # no maximum loss at all (extended decomposition only)
        elif r == "evaluation":
            c[N_EVAL] = 0              # no evaluation: the account starts funded (extended only)
        else:
            raise ValueError(r)
    return c
