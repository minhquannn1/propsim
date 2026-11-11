"""Path-by-path comparison of the fast simulator (engine.py) with the
independent plain-Python reference (validation/reference.py).

The reference was written from validation/SPEC.md only. This harness replays
the same days and the same candidate trades for each path (the random-number
hash and the trade selection are copied here because they are not rules) and
compares outcome, payout, days used and phases passed.

Usage: python3 validation/crosscheck.py [paths per configuration, default 150]
"""
import itertools, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from reference import run_account
from engine import (simulate, N_EVAL, TGT1, TGT2, EVAL_MIN_DAYS, EVAL_CONS, DAILY, DAILY_BASE, MAXDD,
                    TRAIL_MODE, TRAIL_LOCK, F_MIN_PROFIT, F_MIN_QDAYS, F_QTHRESH, F_CONS, SPLIT, CAP,
                    DAILY_SOFT, F_FRAC)
from rules import ARCHETYPES, relax
from core import TRADERS, libs, NOB, NOR, OFF
from data import WIN_START, WIN_END

M = (1 << 64) - 1


def u01(seed, a, b, c):
    x = (seed * 0x9E3779B97F4A7C15) & M
    x ^= (a * 0xBF58476D1CE4E5B9) & M
    x ^= (b * 0x94D049BB133111EB) & M
    x ^= (c * 0xD6E8FEB86659FD93) & M
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & M
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & M
    x ^= x >> 31
    return (x >> 11) * (1.0 / 9007199254740992.0)


def candidate_days(tr, L, s, H, seed, block, ntr=2):
    """The trades path s would take on each day if no rule or goal stopped it."""
    x = tr["lib"]
    nd = L["R"].shape[0] if tr["ttype"] == 0 else L["R" + x].shape[0]
    days = []
    for t in range(H):
        start = int(u01(seed, s, t // block, 999983) * (nd - block))
        d = start + (t % block)
        cands = []
        if tr["ttype"] == 1:
            if L["h" + x][d]:
                cands.append((float(L["R" + x][d]), float(L["M" + x][d]), float(L["F" + x][d])))
        else:
            pos = WIN_START + int(u01(seed, s, t, 0) * (WIN_END - WIN_START))
            for k in range(ntr):
                if pos >= WIN_END:
                    break
                if u01(seed, s, t, 100 + k) < tr["edge"]:
                    side = 0 if L["R"][d, pos, 0] >= L["R"][d, pos, 1] else 1
                else:
                    side = 0 if u01(seed, s, t, 200 + k) < 0.5 else 1
                cands.append((float(L["R"][d, pos, side]), float(L["M"][d, pos, side]), float(L["F"][d, pos, side])))
                nxt = int(L["E"][d, pos, side]) + 1
                if nxt >= WIN_END:
                    break
                pos = nxt + int(u01(seed, s, t, 300 + k) * (WIN_END - nxt))
        days.append(cands)
    return days


def as_dict(c):
    return dict(n_eval=int(c[N_EVAL]), tgt1=c[TGT1], tgt2=c[TGT2], eval_min_days=c[EVAL_MIN_DAYS],
                eval_cons=c[EVAL_CONS], daily=c[DAILY], daily_base=int(c[DAILY_BASE]),
                daily_soft=int(c[DAILY_SOFT]), maxdd=c[MAXDD], trail_mode=int(c[TRAIL_MODE]),
                trail_lock=int(c[TRAIL_LOCK]), f_min_profit=c[F_MIN_PROFIT], f_min_qdays=c[F_MIN_QDAYS],
                f_qthresh=c[F_QTHRESH], f_cons=c[F_CONS], split=c[SPLIT], f_frac=c[F_FRAC], cap=c[CAP])


OUT = {0: "censored", 1: "payout", 2: "fail_daily", 3: "fail_maxdd"}


def compare(cfg, tkey, risk, policy, n, H=250, seed=4242, block=20, wgoal=0.02):
    tr = TRADERS[tkey]
    L = libs()
    if tr["ttype"] == 0:
        lib, b = (L["R"], L["M"], L["F"], L["E"]), NOB
    else:
        x = tr["lib"]
        lib, b = NOR, (L["R" + x], L["M" + x], L["F" + x], L["h" + x])
    if policy is None:
        o, p, u, st = simulate(cfg, tr["ttype"], risk, tr["edge"], 2, *lib, *b, n, H, seed, block, wgoal, 0)
    else:
        o, p, u, st = simulate(cfg, tr["ttype"], risk, tr["edge"], 2, *lib, *b, n, H, seed, block, wgoal, 1,
                               policy["room_frac"], policy["dstop_mult"], policy["qstop"])
    cd = as_dict(cfg)
    bad = []
    for s in range(n):
        ref = run_account(cd, candidate_days(tr, L, s, H, seed, block), risk, wgoal, policy)
        same = (OUT[int(o[s])] == ref["outcome"] and abs(p[s] - ref["payout"]) < 1e-9
                and int(u[s]) == ref["days_used"] and int(st[s]) == ref["phase"])
        if not same:
            bad.append((s, OUT[int(o[s])], float(p[s]), int(u[s]), int(st[s]), ref))
    return bad


def configurations(rng):
    """(label, cfg) pairs: every on/off combination of the published rulesets,
    with the maximum loss and the evaluation as extra switches, plus rulesets
    with randomly drawn thresholds and modes."""
    for akey, A in ARCHETYPES.items():
        ext = list(A["rules"]) + ["max_loss"] + (["evaluation"] if A["cfg"][N_EVAL] > 0 else [])
        for k in range(len(ext) + 1):
            for S in itertools.combinations(ext, k):
                yield f"{akey} minus {'+'.join(S) or 'nothing'}", relax(A["cfg"], S)
    for i in range(400):
        base = ARCHETYPES[rng.choice(list(ARCHETYPES))]["cfg"].copy()
        base[N_EVAL] = rng.choice([0, 1, 2])
        base[TGT1] = rng.choice([0.03, 0.06, 0.10]); base[TGT2] = rng.choice([0.03, 0.05])
        base[EVAL_MIN_DAYS] = rng.choice([0, 2, 4]); base[EVAL_CONS] = rng.choice([1.0, 0.3, 0.5])
        base[DAILY] = rng.choice([OFF, 0.01, 0.02, 0.04]); base[DAILY_BASE] = rng.choice([0, 1])
        base[DAILY_SOFT] = rng.choice([0, 1])
        base[MAXDD] = rng.choice([OFF, 0.03, 0.05, 0.07, 0.10])
        base[TRAIL_MODE] = rng.choice([0, 1, 2, 3]); base[TRAIL_LOCK] = rng.choice([0, 1])
        base[F_MIN_PROFIT] = rng.choice([1e-9, 0.01, 0.03]); base[F_MIN_QDAYS] = rng.choice([0, 3, 6, 10])
        base[F_QTHRESH] = rng.choice([-1.0, 0.0015, 0.005, 0.01]); base[F_CONS] = rng.choice([1.0, 0.15, 0.2, 0.4])
        base[SPLIT] = rng.choice([0.8, 0.9, 1.0]); base[F_FRAC] = rng.choice([0.5, 1.0])
        base[CAP] = rng.choice([OFF, 0.01, 0.027])
        yield f"random ruleset {i}", base


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    rng = random.Random(7)
    policies = [None, dict(room_frac=0.25, dstop_mult=1.0, qstop=1), dict(room_frac=0.1, dstop_mult=0.5, qstop=0),
                dict(room_frac=OFF, dstop_mult=OFF, qstop=1), dict(room_frac=0.5, dstop_mult=OFF, qstop=0)]
    total = mism = ncfg = 0
    for label, cfg in configurations(rng):
        tkey = rng.choice(list(TRADERS))
        risk = rng.choice([0.0025, 0.005, 0.01, 0.015])
        pol = rng.choice(policies)
        bad = compare(cfg, tkey, risk, pol, n)
        total += n; mism += len(bad); ncfg += 1
        if bad:
            print("MISMATCH", label, tkey, risk, pol, "first:", bad[0][:5], bad[0][5])
    print(f"configurations {ncfg}, paths {total}, mismatches {mism}")
