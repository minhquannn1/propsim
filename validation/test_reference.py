"""Hand-worked scenarios for reference.run_account.

Every expected value below was derived by hand from SPEC.md; the arithmetic is
written in the comment above each assertion. Run with either

    python3 -m pytest -q test_reference.py
    python3 test_reference.py

Notation used in the comments: bal = balance, ds = day_start, DF = daily
floor, MF = maximum-loss floor, EF = effective floor = max(DF, MF),
q = qualifying_days, best = best_day, EOD = end of day.
A trade is written (R, MAE, MFE).

Numbers are chosen so that no comparison sits on a knife edge in decimal
arithmetic; where an exact tie is the point of the test, binary-exact values
(0.25, 0.125, 0.5 ...) are used.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import run_account  # noqa: E402

OFF = 1e9


def mk(**overrides):
    """cfg with neutral defaults; a scenario sets only what it needs."""
    cfg = dict(
        n_eval=0, tgt1=0, tgt2=0, eval_min_days=0, eval_cons=1.0,
        daily=1e9, daily_base=0, daily_soft=0,
        maxdd=1e9, trail_mode=0, trail_lock=0,
        f_min_profit=0, f_min_qdays=0, f_qthresh=-1, f_cons=1.0,
        split=1.0, f_frac=1.0, cap=1e9,
    )
    for key in overrides:
        assert key in cfg, "unknown cfg key: %s" % key
    cfg.update(overrides)
    return cfg


def pol(room=OFF, dstop=OFF, qstop=0):
    return {"room_frac": room, "dstop_mult": dstop, "qstop": qstop}


def check(res, outcome, payout, days_used, phase):
    assert res["outcome"] == outcome, res
    assert abs(res["payout"] - payout) < 1e-9, res
    assert res["days_used"] == days_used, res
    assert res["phase"] == phase, res


WIN = (1, 0, 1)      # +1R, never under water
LOSS = (-1, -1, 0)   # -1R, straight down
NEVER = 1.0          # a wgoal that is never reached: no goal stop, no payout


# =========================================================== basics / payout

def test_01_basic_payout():
    # Funded from day 1 (n_eval=0). risk 0.01, trade +1R: bal 1.00 -> 1.01.
    # Goal stop irrelevant (single trade). EOD: profit 0.01 >= min_profit 0,
    # > 0, q 1 >= 0, consistency off -> payout = 1.0 * 1.0 * 0.01 = 0.01.
    res = run_account(mk(), [[WIN]], 0.01, 0.0)
    check(res, "payout", 0.01, 1, 0)


def test_02_censored_after_loss_and_empty_day():
    # Day 1: -1R -> bal 0.99, profit -0.01, not > 0 -> no payout.
    # Day 2: empty list -> nothing. Days run out -> censored, 2 days.
    res = run_account(mk(), [[LOSS], []], 0.01, 0.0)
    check(res, "censored", 0.0, 2, 0)


def test_03_zero_days():
    # No days at all: censored, days_used = len(days) = 0.
    res = run_account(mk(), [], 0.01, 0.0)
    check(res, "censored", 0.0, 0, 0)


def test_04_funded_goal_stop_and_wgoal_within_day():
    days = [[WIN, WIN]]
    # wgoal 0.005 -> min_profit 0.005. After trade 1 profit 0.01 >= 0.005,
    # q (0 + today) 1 >= 0, consistency off -> stop. Trade 2 not taken.
    # EOD payout = 0.01.
    check(run_account(mk(), days, 0.01, 0.005), "payout", 0.01, 1, 0)
    # wgoal 0.015: after trade 1 profit 0.01 < 0.015 -> continue.
    # After trade 2 profit 0.02 >= 0.015 -> payout 0.02.
    check(run_account(mk(), days, 0.01, 0.015), "payout", 0.02, 1, 0)


def test_05_wgoal_and_f_min_profit_take_the_larger():
    days = [[(3, 0, 3)], [(3, 0, 3)]]
    # f_min_profit 0.02, wgoal 0.05 -> min_profit 0.05.
    # Day 1 profit 0.03 < 0.05 -> no payout. Day 2 profit 0.06 -> payout 0.06.
    check(run_account(mk(f_min_profit=0.02), days, 0.01, 0.05),
          "payout", 0.06, 2, 0)
    # f_min_profit 0.05, wgoal 0 -> min_profit 0.05, same path.
    check(run_account(mk(f_min_profit=0.05), days, 0.01, 0.0),
          "payout", 0.06, 2, 0)
    # f_min_profit 0.02, wgoal 0 -> min_profit 0.02: day 1 profit 0.03 pays.
    check(run_account(mk(f_min_profit=0.02), days, 0.01, 0.0),
          "payout", 0.03, 1, 0)


def test_06_profit_must_be_strictly_positive():
    # One flat trade (0R): bal stays exactly 1.0. profit 0 >= min_profit 0 but
    # "profit > 0" fails -> no payout -> censored.
    res = run_account(mk(), [[(0, 0, 0)]], 0.01, 0.0)
    check(res, "censored", 0.0, 1, 0)


def test_07_split_frac_and_cap():
    days = [[(10, 0, 10)]]
    # profit = 0.01 * 10 = 0.10. split 0.8, f_frac 0.5 -> 0.8*0.5*0.10 = 0.04.
    check(run_account(mk(split=0.8, f_frac=0.5), days, 0.01, 0.0),
          "payout", 0.04, 1, 0)
    # cap 0.03 < 0.04 -> payout 0.03.
    check(run_account(mk(split=0.8, f_frac=0.5, cap=0.03), days, 0.01, 0.0),
          "payout", 0.03, 1, 0)
    # cap 0.05 > 0.04 -> payout 0.04.
    check(run_account(mk(split=0.8, f_frac=0.5, cap=0.05), days, 0.01, 0.0),
          "payout", 0.04, 1, 0)


# ====================================================== static maximum loss

def test_08_static_maxdd_fails_on_third_loss():
    # maxdd 0.05, mode 0: MF = 0.95 always. risk 0.02, one -1R per day.
    # Day 1: 1.00 - 0.02 = 0.98 >= 0.95 ok. Day 2: 0.96 >= 0.95 ok.
    # Day 3: 0.94 < 0.95 -> breach. DF is off (ds - 1e9) so DF < MF
    # -> fail_maxdd on day 3.
    res = run_account(mk(maxdd=0.05), [[LOSS], [LOSS], [LOSS]], 0.02, 0.0)
    check(res, "fail_maxdd", 0.0, 3, 0)


def test_09_breach_is_tested_on_mae_not_on_r():
    cfg = mk(maxdd=0.05)
    # Winner that dipped 3R: 1.00 + 0.02*(-3) = 0.94 < 0.95 -> fail_maxdd,
    # although R = +1.
    check(run_account(cfg, [[(1, -3, 1)]], 0.02, 0.0), "fail_maxdd", 0.0, 1, 0)
    # Dip of 2R: 0.96 >= 0.95 ok -> bal 1.02 -> payout 0.02.
    check(run_account(cfg, [[(1, -2, 1)]], 0.02, 0.0), "payout", 0.02, 1, 0)


def test_10_static_floor_does_not_move_but_mode1_does():
    days = [[(3, 0, 3)], [(-5, -5, 0)]]
    # risk 0.02. Day 1: bal 1.06 (profit 0.06 < wgoal 1 -> no payout).
    # Day 2: 1.06 - 0.10 = 0.96.
    # Mode 0: MF 0.95, 0.96 >= 0.95 -> survives -> censored.
    check(run_account(mk(maxdd=0.05), days, 0.02, NEVER),
          "censored", 0.0, 2, 0)
    # Mode 1: high 1.06 after day 1's trade, MF = 1.01, 0.96 < 1.01
    # -> fail_maxdd on day 2.
    check(run_account(mk(maxdd=0.05, trail_mode=1), days, 0.02, NEVER),
          "fail_maxdd", 0.0, 2, 0)


def test_11_exactly_on_the_floor_is_not_a_breach():
    # Binary-exact numbers. maxdd 0.25 -> MF 0.75. risk 0.125.
    # Day 1: MAE -2 -> 1 - 0.25 = 0.75, not strictly below 0.75 -> no breach;
    # R -2 -> bal 0.75. Censored if that is the only day.
    cfg = mk(maxdd=0.25)
    check(run_account(cfg, [[(-2, -2, 0)]], 0.125, 0.0), "censored", 0.0, 1, 0)
    # Day 2: MAE -0.5 -> 0.75 - 0.0625 = 0.6875 < 0.75 -> fail_maxdd.
    check(run_account(cfg, [[(-2, -2, 0)], [(-0.5, -0.5, 0)]], 0.125, 0.0),
          "fail_maxdd", 0.0, 2, 0)


# ============================================================ trailing modes

def test_12_mode1_moves_intraday_mode2_does_not():
    days = [[(3, 0, 3), (-4, -4, 0)]]
    # risk 0.02, maxdd 0.05, wgoal never reached (no goal stop).
    # Trade 1: bal 1.06. Trade 2 worst point: 1.06 - 0.08 = 0.98.
    # Mode 1: high 1.06 right after trade 1 -> MF 1.01; 0.98 < 1.01
    # -> fail_maxdd on day 1.
    check(run_account(mk(maxdd=0.05, trail_mode=1), days, 0.02, NEVER),
          "fail_maxdd", 0.0, 1, 0)
    # Mode 2: high still 1.0 during the day -> MF 0.95; 0.98 >= 0.95 ok;
    # bal 0.98; EOD high = max(1.0, 0.98) = 1.0 -> censored.
    check(run_account(mk(maxdd=0.05, trail_mode=2), days, 0.02, NEVER),
          "censored", 0.0, 1, 0)
    # Mode 0: MF 0.95 -> also survives.
    check(run_account(mk(maxdd=0.05, trail_mode=0), days, 0.02, NEVER),
          "censored", 0.0, 1, 0)


def test_13_mode2_floor_moves_at_end_of_day():
    # Same two trades, but on separate days. Mode 2: day 1 EOD bal 1.06
    # -> high 1.06 -> MF 1.01. Day 2: 1.06 - 0.08 = 0.98 < 1.01 -> fail_maxdd.
    days = [[(3, 0, 3)], [(-4, -4, 0)]]
    check(run_account(mk(maxdd=0.05, trail_mode=2), days, 0.02, NEVER),
          "fail_maxdd", 0.0, 2, 0)


def test_14_mode2_uses_eod_balance_not_intraday_peak():
    days = [[(5, 0, 5), (-3, -3, 0)], [(-4, -4, 0)], [(-2, -2, 0)]]
    # risk 0.01, maxdd 0.05.
    # Day 1: 1.05, then worst/close 1.02.
    # Mode 2: intraday MF 0.95 -> ok. EOD high = 1.02 -> MF 0.97.
    #   Day 2: 1.02 - 0.04 = 0.98 >= 0.97 ok, bal 0.98, high stays 1.02.
    #   Day 3: 0.98 - 0.02 = 0.96 < 0.97 -> fail_maxdd on day 3.
    check(run_account(mk(maxdd=0.05, trail_mode=2), days, 0.01, NEVER),
          "fail_maxdd", 0.0, 3, 0)
    # Mode 1: high 1.05 after trade 1 -> MF 1.00; trade 2: 1.02 >= 1.00 ok.
    #   Day 2: 0.98 < 1.00 -> fail_maxdd on day 2.
    check(run_account(mk(maxdd=0.05, trail_mode=1), days, 0.01, NEVER),
          "fail_maxdd", 0.0, 2, 0)
    # Mode 0: MF 0.95; lowest point 0.96 >= 0.95 -> censored after 3 days.
    check(run_account(mk(maxdd=0.05, trail_mode=0), days, 0.01, NEVER),
          "censored", 0.0, 3, 0)


def test_15_mode3_open_profit_raises_the_floor():
    days = [[(1, 0, 4), (-3, -3, 0)]]
    # risk 0.01, maxdd 0.05.
    # Trade 1: best open equity 1.04, bal 1.01.
    # Mode 3: high 1.04 -> MF 0.99; give-back check 1.01 >= 0.99 ok.
    #   Trade 2: 1.01 - 0.03 = 0.98 < 0.99 -> fail_maxdd on day 1.
    check(run_account(mk(maxdd=0.05, trail_mode=3), days, 0.01, NEVER),
          "fail_maxdd", 0.0, 1, 0)
    # Mode 1: high = closed balance 1.01 -> MF 0.96; 0.98 >= 0.96 -> survives.
    check(run_account(mk(maxdd=0.05, trail_mode=1), days, 0.01, NEVER),
          "censored", 0.0, 1, 0)


def test_16_mode3_giveback_failure_on_a_winning_trade():
    days = [[(1, 0, 7)]]
    # risk 0.01, maxdd 0.05. No breach on MAE (0). Best open equity 1.07,
    # bal 1.01. Mode 3: high 1.07 -> MF 1.02; 1.01 < 1.02 -> fail_maxdd.
    check(run_account(mk(maxdd=0.05, trail_mode=3), days, 0.01, 0.0),
          "fail_maxdd", 0.0, 1, 0)
    # Mode 1 for contrast: high 1.01, no give-back rule -> payout 0.01.
    check(run_account(mk(maxdd=0.05, trail_mode=1), days, 0.01, 0.0),
          "payout", 0.01, 1, 0)


def test_17_mode3_giveback_failure_on_a_losing_trade():
    # risk 0.01, maxdd 0.03, trade (-1, -1, 3).
    # Breach test: 1.00 - 0.01 = 0.99 >= MF 0.97 ok.
    # Best open equity 1.03, bal 0.99. high 1.03 -> MF 1.00; 0.99 < 1.00
    # -> fail_maxdd.
    check(run_account(mk(maxdd=0.03, trail_mode=3), [[(-1, -1, 3)]], 0.01, 0.0),
          "fail_maxdd", 0.0, 1, 0)
    # With maxdd 0.05: MF after trade = 0.98; 0.99 >= 0.98 ok.
    # Day 2 trade (-1.5,-1.5,0): 0.99 - 0.015 = 0.975 < 0.98 -> fail_maxdd.
    days = [[(-1, -1, 3)], [(-1.5, -1.5, 0)]]
    check(run_account(mk(maxdd=0.05, trail_mode=3), days, 0.01, 0.0),
          "fail_maxdd", 0.0, 2, 0)
    # Mode 0 on the same days: MF 0.95; 0.975 >= 0.95 -> censored.
    check(run_account(mk(maxdd=0.05, trail_mode=0), days, 0.01, 0.0),
          "censored", 0.0, 2, 0)


# ================================================================== the lock

def test_18_lock_mode1():
    days = [[(10, 0, 10)], [(-8, -8, 0)], [(-3, -3, 0)]]
    # risk 0.01, maxdd 0.05. Day 1: bal 1.10, high 1.10.
    # Day 2 worst point: 1.10 - 0.08 = 1.02.
    # No lock: MF = 1.05; 1.02 < 1.05 -> fail_maxdd on day 2.
    check(run_account(mk(maxdd=0.05, trail_mode=1), days, 0.01, NEVER),
          "fail_maxdd", 0.0, 2, 0)
    # Lock: MF = min(1.05, 1.0) = 1.0; 1.02 >= 1.0 ok, bal 1.02.
    # Day 3: 1.02 - 0.03 = 0.99 < 1.0 -> fail_maxdd on day 3.
    check(run_account(mk(maxdd=0.05, trail_mode=1, trail_lock=1), days, 0.01,
                      NEVER), "fail_maxdd", 0.0, 3, 0)
    # Static for contrast: MF 0.95; 0.99 >= 0.95 -> censored.
    check(run_account(mk(maxdd=0.05), days, 0.01, NEVER),
          "censored", 0.0, 3, 0)


def test_19_lock_has_no_effect_while_floor_is_below_initial_balance():
    days = [[(3, 0, 3)], [(-5.5, -5.5, 0)]]
    # risk 0.01, maxdd 0.05, mode 1 with lock. Day 1: bal 1.03, high 1.03,
    # MF = 0.98 (< 1.0, lock not binding).
    # Day 2: 1.03 - 0.055 = 0.975 < 0.98 -> fail_maxdd.
    check(run_account(mk(maxdd=0.05, trail_mode=1, trail_lock=1), days, 0.01,
                      NEVER), "fail_maxdd", 0.0, 2, 0)
    # Static: 0.975 >= 0.95 -> censored.
    check(run_account(mk(maxdd=0.05), days, 0.01, NEVER),
          "censored", 0.0, 2, 0)


def test_20_lock_mode2():
    days = [[(10, 0, 10)], [(-8, -8, 0)], [(-3, -3, 0)]]
    # Mode 2: EOD day 1 high 1.10.
    # No lock: MF 1.05; day 2 worst 1.02 < 1.05 -> fail_maxdd on day 2.
    check(run_account(mk(maxdd=0.05, trail_mode=2), days, 0.01, NEVER),
          "fail_maxdd", 0.0, 2, 0)
    # Lock: MF 1.0; day 2 1.02 ok (EOD high stays 1.10);
    # day 3 0.99 < 1.0 -> fail_maxdd on day 3.
    check(run_account(mk(maxdd=0.05, trail_mode=2, trail_lock=1), days, 0.01,
                      NEVER), "fail_maxdd", 0.0, 3, 0)


def test_21_lock_mode3_prevents_giveback_failure_above_initial_balance():
    # Trade (1, 0, 7), risk 0.01, maxdd 0.05, mode 3 with lock.
    # high 1.07 -> unlocked MF 1.02, locked MF = 1.0. bal 1.01 >= 1.0 ok.
    # Goal stop; EOD profit 0.01 -> payout 0.01.
    res = run_account(mk(maxdd=0.05, trail_mode=3, trail_lock=1),
                      [[(1, 0, 7)]], 0.01, 0.0)
    check(res, "payout", 0.01, 1, 0)


# =============================================================== daily limit

def test_22_hard_daily_base0_versus_base1():
    days = [[(5, 0, 5)], [(-1.2, -1.2, 0)]]
    # risk 0.1, daily 0.1. Day 1: ds 1.0, bal 1.5 (profit 0.5 < wgoal 1).
    # Day 2: ds 1.5, worst point 1.5 - 0.12 = 1.38.
    # Base 0: DF = 1.5 - 0.1 = 1.40; 1.38 < 1.40; MF off -> fail_daily.
    check(run_account(mk(daily=0.1, daily_base=0), days, 0.1, NEVER),
          "fail_daily", 0.0, 2, 0)
    # Base 1: DF = 1.5 * 0.9 = 1.35; 1.38 >= 1.35 -> survives -> censored.
    check(run_account(mk(daily=0.1, daily_base=1), days, 0.1, NEVER),
          "censored", 0.0, 2, 0)


def test_23_daily_floor_is_fixed_from_day_start():
    days = [[(5, 0, 5), (-6, -6, 0)], [(-3.5, -3.5, 0)]]
    # risk 0.01, daily 0.03 base 0.
    # Day 1: ds 1.0 -> DF 0.97 all day. Trade 1: 1.05. Trade 2 worst:
    # 1.05 - 0.06 = 0.99 >= 0.97 ok (it does not trail the intraday 1.05).
    # Day 2: ds 0.99 -> DF 0.96; 0.99 - 0.035 = 0.955 < 0.96 -> fail_daily.
    check(run_account(mk(daily=0.03), days, 0.01, NEVER),
          "fail_daily", 0.0, 2, 0)


def test_24_soft_daily_base0():
    days = [[(-2, -2, 0), (-2, -2, 0), (5, 0, 5)], [(4, 0, 4)]]
    # risk 0.01, daily 0.03 soft, base 0. Day 1: DF 0.97.
    # Trade 1: 0.98 >= 0.97 ok, bal 0.98 (profit -0.02, no goal stop).
    # Trade 2: 0.98 - 0.02 = 0.96 < 0.97 -> soft breach (DF >= MF, MF off):
    # bal = 0.97, day over, trade 3 NOT taken. EOD profit -0.03, no payout.
    # Day 2: ds 0.97, DF 0.94. Trade: bal 0.97 + 0.04 = 1.01.
    # EOD profit 0.01 > 0 -> payout 0.01 on day 2.
    check(run_account(mk(daily=0.03, daily_soft=1), days, 0.01, 0.0),
          "payout", 0.01, 2, 0)
    # Hard version of the same account: fail_daily on day 1.
    check(run_account(mk(daily=0.03, daily_soft=0), days, 0.01, 0.0),
          "fail_daily", 0.0, 1, 0)


def test_25_soft_daily_base1_and_base0_close_at_their_own_floor():
    days = [[(5, 0, 5)], [(-2, -2, 0), (9, 0, 9)], [(1, 0, 1)]]
    # risk 0.1, daily 0.1 soft, 3 qualifying days needed (no threshold) so
    # that nothing is paid before day 3.
    # Day 1: bal 1.5; q-so-far 1 < 3 -> no goal stop; EOD q = 1.
    # Day 2: ds 1.5; worst point 1.5 - 0.2 = 1.3.
    #   Base 1: DF = 1.35; 1.3 < 1.35 -> bal = 1.35, stop (trade 2 skipped).
    #   The day counts as traded and (no threshold) qualifies -> q = 2.
    # Day 3: bal 1.35 + 0.1 = 1.45, q = 3, profit 0.45 -> payout 0.45.
    cfg = mk(daily=0.1, daily_base=1, daily_soft=1, f_min_qdays=3)
    check(run_account(cfg, days, 0.1, 0.0), "payout", 0.45, 3, 0)
    #   Base 0: DF = 1.40 -> bal = 1.40; day 3: 1.50 -> payout 0.50.
    cfg = mk(daily=0.1, daily_base=0, daily_soft=1, f_min_qdays=3)
    check(run_account(cfg, days, 0.1, 0.0), "payout", 0.50, 3, 0)


# ====================================== which floor is higher decides the type

def test_26_daily_floor_higher_gives_fail_daily():
    # daily 0.03, maxdd 0.05 static, risk 0.01. Fresh account: DF 0.97,
    # MF 0.95, EF 0.97. Worst point 1 - 0.04 = 0.96 < 0.97; DF >= MF
    # -> fail_daily.
    res = run_account(mk(daily=0.03, maxdd=0.05), [[(-4, -4, 0)]], 0.01, 0.0)
    check(res, "fail_daily", 0.0, 1, 0)


def test_27_maxdd_floor_higher_gives_fail_maxdd():
    days = [[(-2.5, -2.5, 0)], [(-3, -3, 0)]]
    # Day 1: DF 0.97; 1 - 0.025 = 0.975 >= 0.97 ok; bal 0.975.
    # Day 2: DF = 0.975 - 0.03 = 0.945, MF 0.95, EF 0.95.
    # Worst point 0.975 - 0.03 = 0.945 < 0.95; DF < MF -> fail_maxdd.
    res = run_account(mk(daily=0.03, maxdd=0.05), days, 0.01, 0.0)
    check(res, "fail_maxdd", 0.0, 2, 0)


def test_28_soft_daily_does_not_save_when_maxdd_floor_is_higher():
    days = [[(-4, -4, 0)], [(-4, -4, 0)]]
    # daily 0.03 soft, maxdd 0.05 static, risk 0.01.
    # Day 1: DF 0.97 >= MF 0.95; 0.96 < 0.97 -> soft: bal = 0.97.
    # Day 2: DF 0.94 < MF 0.95, EF 0.95; 0.97 - 0.04 = 0.93 < 0.95;
    # DF is not >= MF -> hard failure, fail_maxdd on day 2.
    res = run_account(mk(daily=0.03, daily_soft=1, maxdd=0.05), days, 0.01, 0.0)
    check(res, "fail_maxdd", 0.0, 2, 0)


def test_29_equal_floors_count_as_daily():
    # Binary-exact: daily 0.25 and maxdd 0.25 -> DF = MF = 0.75 exactly.
    # risk 0.5, MAE -1 -> 0.5 < 0.75. "daily floor >= max-loss floor" holds
    # -> fail_daily.
    cfg = mk(daily=0.25, maxdd=0.25)
    check(run_account(cfg, [[LOSS]], 0.5, 0.0), "fail_daily", 0.0, 1, 0)
    # With the soft daily limit the same tie means the account survives at
    # bal = 0.75 -> censored.
    cfg = mk(daily=0.25, maxdd=0.25, daily_soft=1)
    check(run_account(cfg, [[LOSS]], 0.5, 0.0), "censored", 0.0, 1, 0)


# ========================================================== qualifying days

def test_30_qualifying_days_without_threshold():
    days = [[WIN], [], [LOSS], [WIN]]
    # f_min_qdays 3, f_qthresh -1 (every traded day qualifies), risk 0.01.
    # Day 1: bal 1.01, q = 1 (< 3, no payout).
    # Day 2: empty -> not traded, q stays 1.
    # Day 3: -1R -> bal 1.00; a losing day still qualifies -> q = 2.
    # Day 4: bal 1.01, q = 3, profit 0.01 > 0 -> payout 0.01 on day 4.
    res = run_account(mk(f_min_qdays=3), days, 0.01, 0.0)
    check(res, "payout", 0.01, 4, 0)


def test_31_qualifying_days_with_threshold():
    days = [[(0.3, 0, 0.3)], [WIN], [WIN]]
    # f_qthresh 0.005, f_min_qdays 2, risk 0.01.
    # Day 1: day profit 0.003 < 0.005 * 1.0 -> not qualifying, q = 0.
    # Day 2: ds 1.003, day profit 0.01 >= 0.005015 -> q = 1 (no payout).
    # Day 3: ds 1.013, day profit 0.01 >= 0.005065 -> q = 2,
    #        profit 0.023 -> payout 0.023 on day 3.
    res = run_account(mk(f_min_qdays=2, f_qthresh=0.005), days, 0.01, 0.0)
    check(res, "payout", 0.023, 3, 0)


def test_32_qualifying_threshold_scales_with_day_start_balance():
    days = [[(5, 0, 5)], [(0.12, 0, 0.12)], [(0.2, 0, 0.2)]]
    # f_qthresh 0.01, f_min_qdays 2, risk 0.1.
    # Day 1: ds 1.0, day profit 0.5 >= 0.01 -> q = 1. bal 1.5.
    # Day 2: ds 1.5, need 0.015; day profit 0.012 < 0.015 -> q stays 1
    #        (it would qualify against the initial balance, 0.012 >= 0.01).
    # Day 3: ds 1.512, need 0.01512; day profit 0.02 -> q = 2.
    #        profit = 0.5 + 0.012 + 0.02 = 0.532 -> payout 0.532 on day 3.
    res = run_account(mk(f_min_qdays=2, f_qthresh=0.01), days, 0.1, 0.0)
    check(res, "payout", 0.532, 3, 0)


# ====================================================== funded consistency

def test_33_funded_consistency_delays_the_payout():
    days = [[(3, 0, 3)], [(2, 0, 2)], [(2, 0, 2)]]
    # f_cons 0.5, risk 0.01.
    # Day 1: profit 0.03, best 0.03; 0.03 <= 0.5*0.03 = 0.015? no.
    # Day 2: profit 0.05; 0.03 <= 0.025? no.
    # Day 3: profit 0.07; 0.03 <= 0.035? yes -> payout 0.07 on day 3.
    res = run_account(mk(f_cons=0.5), days, 0.01, 0.0)
    check(res, "payout", 0.07, 3, 0)


def test_34_funded_goal_stop_counts_today_as_possible_best_day():
    days = [[(2.5, 0, 2.5)], [WIN, WIN, WIN, WIN], [WIN], [WIN]]
    # f_cons 0.5, risk 0.01, wgoal 0.
    # Day 1: profit 0.025, best 0.025; 0.025 <= 0.0125? no.
    # Day 2 (best so far 0.025), after each trade, best = max(0.025, today):
    #   t1: profit 0.035, today 0.01 -> 0.025 <= 0.0175? no.
    #   t2: profit 0.045, today 0.02 -> 0.025 <= 0.0225? no.
    #   t3: profit 0.055, today 0.03 -> 0.030 <= 0.0275? no  (with the old
    #       best 0.025 alone it would be yes: this is the point of the test).
    #   t4: profit 0.065, today 0.04 -> 0.040 <= 0.0325? no.
    #   EOD: best 0.04, no payout.
    # Day 3: profit 0.075; 0.04 <= 0.0375? no.
    # Day 4: profit 0.085; 0.04 <= 0.0425? yes -> payout 0.085 on day 4.
    res = run_account(mk(f_cons=0.5), days, 0.01, 0.0)
    check(res, "payout", 0.085, 4, 0)


# ================================================================ evaluation

def test_35_one_phase_pass_goal_stop_and_fresh_account():
    days = [[(6, 0, 6), WIN], [WIN]]
    # n_eval 1, tgt1 0.05, risk 0.01.
    # Day 1: trade 1 -> profit 0.06 >= 0.05 -> goal stop, trade 2 not taken.
    #   EOD: target met, min days 0, consistency off -> phase 1, fresh account.
    # Day 2 (funded): bal 1.00 -> 1.01, profit 0.01 -> payout 0.01 (not 0.07:
    #   the balance was reset).
    res = run_account(mk(n_eval=1, tgt1=0.05), days, 0.01, 0.0)
    check(res, "payout", 0.01, 2, 1)


def test_36_eval_minimum_days_filled_with_token_days():
    days = [[(6, 0, 6)], [(-10, -10, 0)], [(-10, -10, 0)], [WIN]]
    # n_eval 1, tgt1 0.05, eval_min_days 3, risk 0.01.
    # Day 1: profit 0.06, traded_days 1 < 3 -> not passed.
    # Day 2: target already met, consistency off -> token day: the -10R
    #   candidate is NOT used; traded_days 2.
    # Day 3: same -> traded_days 3 -> passed on day 3.
    # Day 4 (funded): +0.01 -> payout 0.01.
    res = run_account(mk(n_eval=1, tgt1=0.05, eval_min_days=3), days, 0.01, 0.0)
    check(res, "payout", 0.01, 4, 1)


def test_37_eval_target_met_but_consistency_not_yet_satisfied():
    days = [[(6, 0, 6), (9, 0, 9)], [(2, 0, 2), (9, 0, 9)], [(2, 0, 2)],
            [(3, 0, 3)], [WIN]]
    # n_eval 1, tgt1 0.05, eval_cons 0.5, risk 0.01.
    # Day 1: t1 -> profit 0.06 >= 0.05 -> goal stop (9R not taken).
    #   EOD: best 0.06 <= 0.5*0.06 = 0.03? no -> not passed.
    # Day 2: target met but consistency fails -> trades normally.
    #   t1 -> profit 0.08 -> goal stop (9R not taken). 0.06 <= 0.04? no.
    # Day 3: t1 -> profit 0.10. 0.06 <= 0.05? no.
    # Day 4: t1 -> profit 0.13. 0.06 <= 0.065? yes -> passed on day 4.
    # Day 5 (funded): +0.01 -> payout 0.01.
    res = run_account(mk(n_eval=1, tgt1=0.05, eval_cons=0.5), days, 0.01, 0.0)
    check(res, "payout", 0.01, 5, 1)


def test_38_eval_consistency_satisfied_then_trader_sits_on_target():
    days = [[(3, 0, 3)], [(4, 0, 4)], [(2, 0, 2)], [(-9, -9, 0)], [WIN]]
    # n_eval 1, tgt1 0.05, eval_cons 0.5, eval_min_days 4, risk 0.01.
    # Day 1: profit 0.03 < 0.05. traded 1, best 0.03.
    # Day 2: profit 0.07 (goal stop). traded 2, best 0.04; 2 < 4 days.
    # Day 3: start: target met, 0.04 <= 0.035? no -> trades: profit 0.09.
    #   traded 3; 0.04 <= 0.045 yes, but 3 < 4 days -> not passed.
    # Day 4: start: target met and 0.04 <= 0.045 -> token day, the -9R
    #   candidate is not used. traded 4 -> passed on day 4.
    # Day 5 (funded): +0.01 -> payout 0.01.
    cfg = mk(n_eval=1, tgt1=0.05, eval_cons=0.5, eval_min_days=4)
    check(run_account(cfg, days, 0.01, 0.0), "payout", 0.01, 5, 1)


def test_39_failure_and_censoring_during_evaluation_report_phase():
    # n_eval 1, tgt1 0.05, daily 0.03, risk 0.01: 1 - 0.04 = 0.96 < 0.97
    # -> fail_daily on day 1 with 0 phases passed.
    cfg = mk(n_eval=1, tgt1=0.05, daily=0.03)
    check(run_account(cfg, [[(-4, -4, 0)]], 0.01, 0.0), "fail_daily", 0.0, 1, 0)
    # One +1R day: profit 0.01 < 0.05 -> still in evaluation -> censored,
    # no payout even though profit > 0.
    check(run_account(cfg, [[WIN]], 0.01, 0.0), "censored", 0.0, 1, 0)


def test_40_two_phases_use_tgt1_then_tgt2():
    days = [[(9, 0, 9)], [(6, 0, 6)], [(2, 0, 2)]]
    # n_eval 2, tgt1 0.08, tgt2 0.05, risk 0.01.
    # Day 1: profit 0.09 >= 0.08 -> phase 1, reset.
    # Day 2: profit 0.06 >= tgt2 0.05 (it would not meet tgt1) -> phase 2.
    # Day 3 (funded): profit 0.02 -> payout 0.02.
    res = run_account(mk(n_eval=2, tgt1=0.08, tgt2=0.05), days, 0.01, 0.0)
    check(res, "payout", 0.02, 3, 2)


def test_41_two_phase_reset_of_balance_and_trailing_high():
    cfg = mk(n_eval=2, tgt1=0.08, tgt2=0.05, maxdd=0.05, trail_mode=1)
    # Day 1: bal 1.09, high 1.09 -> phase 1 passed, reset: bal 1.0, high 1.0.
    # Day 2: MF = 1.0 - 0.05 = 0.95 (without the reset it would be 1.04).
    #   1.00 - 0.04 = 0.96 >= 0.95 -> survives. Censored in phase 1.
    days = [[(9, 0, 9)], [(-4, -4, 0)]]
    check(run_account(cfg, days, 0.01, 0.0), "censored", 0.0, 2, 1)
    # Day 3: 0.96 - 0.02 = 0.94 < 0.95 -> fail_maxdd, 1 phase passed.
    days = [[(9, 0, 9)], [(-4, -4, 0)], [(-2, -2, 0)]]
    check(run_account(cfg, days, 0.01, 0.0), "fail_maxdd", 0.0, 3, 1)


def test_42_two_phase_reset_of_the_day_counter():
    days = [[(6, 0, 6)], [WIN], [(6, 0, 6)], [WIN], [(2, 0, 2)]]
    # n_eval 2, tgt1 = tgt2 = 0.05, eval_min_days 2, risk 0.01.
    # Day 1: profit 0.06, traded 1 (< 2).
    # Day 2: token day (candidate ignored), traded 2 -> phase 1, counters 0.
    # Day 3: profit 0.06, traded 1 (< 2; it would be 3 without the reset).
    # Day 4: token day (the +1R candidate is ignored), traded 2 -> phase 2.
    # Day 5 (funded): +0.02 -> payout 0.02 on day 5.
    cfg = mk(n_eval=2, tgt1=0.05, tgt2=0.05, eval_min_days=2)
    check(run_account(cfg, days, 0.01, 0.0), "payout", 0.02, 5, 2)


# ======================================================== rule-aware trader

def test_43_rule_aware_sizing_from_the_maxdd_floor():
    days = [[LOSS, (4, 0, 4)]]
    cfg = mk(maxdd=0.05)
    # risk 0.04, room_frac 0.5, wgoal 0.01. MF 0.95.
    # t1: size = min(0.04, 0.5*(1.00-0.95)) = 0.025. Worst 0.975 >= 0.95.
    #     bal 0.975.
    # t2: size = min(0.04, 0.5*(0.975-0.95)) = 0.0125 (>= 0.004).
    #     bal 0.975 + 4*0.0125 = 1.025 -> goal stop, payout 0.025.
    check(run_account(cfg, days, 0.04, 0.01, pol(room=0.5)),
          "payout", 0.025, 1, 0)
    # Naive: t1 bal 0.96 (>= 0.95), t2 bal 0.96 + 0.16 = 1.12 -> payout 0.12.
    check(run_account(cfg, days, 0.04, 0.01), "payout", 0.12, 1, 0)


def test_44_rule_aware_sizing_uses_the_higher_daily_floor():
    # daily 0.02 -> DF 0.98 > MF 0.95, EF 0.98.
    # size = min(0.04, 0.5*(1.00-0.98)) = 0.01. Trade +3R: bal 1.03.
    # profit 0.03 >= wgoal 0.01 -> payout 0.03.
    cfg = mk(maxdd=0.05, daily=0.02)
    check(run_account(cfg, [[(3, 0, 3)]], 0.04, 0.01, pol(room=0.5)),
          "payout", 0.03, 1, 0)


def test_45_rule_aware_sits_out_when_size_is_too_small():
    big = (100, 0, 100)
    days = [[LOSS, LOSS, LOSS, big], [big]]
    cfg = mk(maxdd=0.05)
    # risk 0.04 (minimum size 0.004), room_frac 0.5, MF 0.95.
    # t1: size 0.025   -> bal 0.975   (room 0.025)
    # t2: size 0.0125  -> bal 0.9625  (room 0.0125)
    # t3: size 0.00625 -> bal 0.95625 (room 0.00625)
    # t4: size 0.003125 < 0.004 -> no more trades today; +100R not taken.
    # Day 2: size 0.003125 < 0.004 again -> sits out. Censored.
    check(run_account(cfg, days, 0.04, 0.0, pol(room=0.5)),
          "censored", 0.0, 2, 0)
    # Naive: t1 bal 0.96; t2 worst 0.92 < 0.95 -> fail_maxdd on day 1.
    check(run_account(cfg, days, 0.04, 0.0), "fail_maxdd", 0.0, 1, 0)


def test_46_sat_out_day_does_not_count_as_traded():
    days = [[(1, 0, 5.95)], [WIN], [WIN]]
    cfg = mk(maxdd=0.05, trail_mode=3, f_min_qdays=2)
    # risk 0.01, room_frac 1.0, 2 qualifying days needed (no threshold).
    # Day 1: MF 0.95, size = min(0.01, 0.05) = 0.01. Best open equity
    #   1.0595, bal 1.01. high 1.0595 -> MF 1.0095; 1.01 >= 1.0095 ok.
    #   q = 1, no payout.
    # Day 2: room = 1.01 - 1.0095 = 0.0005 -> size 0.0005 < 0.001 -> sits
    #   out; not a traded day, q stays 1 -> no payout.
    # Day 3: same. Censored after 3 days.
    check(run_account(cfg, days, 0.01, 0.0, pol(room=1.0)),
          "censored", 0.0, 3, 0)
    # Naive: day 2 trade taken (MAE 0, no breach; MFE 1.02 < high), bal 1.02,
    #   q = 2 -> payout 0.02 on day 2.
    check(run_account(cfg, days, 0.01, 0.0), "payout", 0.02, 2, 0)


def test_47_rule_aware_day_stop_in_the_funded_stage():
    days = [[(1.5, 0, 1.5), WIN, WIN, WIN], [WIN, (1.2, 0, 1.2), WIN],
            [WIN, WIN, WIN]]
    cfg = mk(f_cons=0.5, f_min_profit=0.04)
    # risk 0.01, wgoal 0 -> min_profit 0.04. dstop_mult 1:
    # need = max(0.04, 0 * max(-1, 0)) = 0.04; dstop = 1 * 0.5 * 0.04 = 0.02.
    # Day 1: t1 day 0.015 (< 0.02), t2 day 0.025 >= 0.02 -> stop.
    #   profit 0.025 < 0.04 -> no payout. best 0.025.
    # Day 2: t1 day 0.01, t2 day 0.022 >= 0.02 -> stop. profit 0.047.
    #   0.025 <= 0.5*0.047 = 0.0235? no -> no payout.
    # Day 3: t1 day 0.01, profit 0.057 >= 0.04, best max(0.025, 0.01) = 0.025
    #   <= 0.0285 -> goal stop. EOD payout 0.057 on day 3.
    check(run_account(cfg, days, 0.01, 0.0, pol(dstop=1.0)),
          "payout", 0.057, 3, 0)
    # Naive (no day stop):
    # Day 1: 0.015, 0.025, 0.035, 0.045; at 0.045 (>= 0.04) today's 0.045 is
    #   the best day: 0.045 <= 0.0225? no. All four taken. best 0.045.
    # Day 2: 0.055 (0.045 <= 0.0275 no), 0.067 (<= 0.0335 no),
    #   0.077 (<= 0.0385 no). No payout.
    # Day 3: 0.087 (0.045 <= 0.0435 no), 0.097 (0.045 <= 0.0485 yes) -> stop.
    #   Payout 0.097 on day 3.
    check(run_account(cfg, days, 0.01, 0.0), "payout", 0.097, 3, 0)


def test_48_day_stop_need_includes_the_qualifying_day_requirement():
    day = [(1.1, 0, 1.1)] * 6
    days = [day] * 5
    cfg = mk(f_cons=0.4, f_min_profit=0.02, f_min_qdays=5, f_qthresh=0.01)
    # risk 0.005 -> each trade +0.0055. dstop_mult 1, qstop 0.
    # need = max(0.02, 5 * 0.01) = 0.05; dstop = 1 * 0.4 * 0.05 = 0.02.
    # Days 1-4: 0.0055, 0.011, 0.0165, 0.022 >= 0.02 -> stop after 4 trades.
    #   (No goal stop earlier: at most 3 + 1 = 4 qualifying days < 5.)
    #   Each day profit 0.022 >= 0.01 * ds (ds <= 1.066) -> qualifies.
    #   After day 4: profit 0.088, q 4, best 0.022. No payout (q < 5).
    # Day 5: ds 1.088, threshold 0.01088.
    #   t1: today 0.0055 < 0.01088 -> q-so-far 4 -> continue.
    #   t2: today 0.011 >= 0.01088 -> q-so-far 5; profit 0.099 >= 0.02;
    #       max(0.022, 0.011) = 0.022 <= 0.4*0.099 = 0.0396 -> goal stop.
    #   EOD: q 5, payout 0.099 on day 5.
    # (If need were min_profit alone, dstop would be 0.008 and the payout
    #  would be 0.055.)
    check(run_account(cfg, days, 0.005, 0.0, pol(dstop=1.0)),
          "payout", 0.099, 5, 0)


def test_49_qstop_stops_the_day_once_it_qualifies():
    days = [[WIN, WIN, WIN]] * 3
    cfg = mk(f_cons=0.6, f_min_qdays=2, f_qthresh=0.01)
    # risk 0.006, wgoal 0.035 -> min_profit 0.035. dstop_mult 10:
    # need = max(0.035, 2*0.01) = 0.035; dstop = 10 * 0.6 * 0.035 = 0.21.
    # qstop 1:
    # Day 1: q 0 < 2 -> dstop = min(0.21, 0.01 * 1.0) = 0.01.
    #   t1 day 0.006; t2 day 0.012 >= 0.01 -> stop. Qualifies, q 1, best 0.012.
    # Day 2: ds 1.012, dstop = 0.01012. t1 0.006; t2 0.012 -> stop.
    #   profit 0.024 < 0.035, q 2. No payout.
    # Day 3: q 2 is not < 2 -> dstop 0.21. t1 profit 0.030 (< 0.035);
    #   t2 profit 0.036 >= 0.035, q ok, max(0.012, 0.012) <= 0.6*0.036 =
    #   0.0216 -> goal stop. Payout 0.036 on day 3.
    check(run_account(cfg, days, 0.006, 0.035, pol(dstop=10.0, qstop=1)),
          "payout", 0.036, 3, 0)
    # qstop 0: dstop 0.21 throughout.
    # Day 1: 0.006, 0.012, 0.018 (profit < 0.035, no goal stop). q 1,
    #   best 0.018.
    # Day 2: 0.024, 0.030, 0.036 >= 0.035; q-so-far 1 + 1 (today 0.018 >=
    #   0.01018) = 2; max(0.018, 0.018) <= 0.0216 -> payout 0.036 on day 2.
    check(run_account(cfg, days, 0.006, 0.035, pol(dstop=10.0, qstop=0)),
          "payout", 0.036, 2, 0)


def test_50_rule_aware_day_stop_in_the_evaluation():
    days = [[(2, 0, 2), (1.5, 0, 1.5), (3, 0, 3)],
            [(1.2, 0, 1.2), (2, 0, 2), (3, 0, 3)],
            [(0.5, 0, 0.5), (9, 0, 9)],
            [WIN]]
    cfg = mk(n_eval=1, tgt1=0.06, eval_cons=0.5)
    # risk 0.01. dstop_mult 1 -> dstop = 1 * 0.5 * 0.06 = 0.03.
    # Day 1: t1 day 0.02; t2 day 0.035 >= 0.03 -> stop. profit 0.035 < 0.06.
    # Day 2: t1 day 0.012; t2 day 0.032 >= 0.03 -> stop. profit 0.067.
    #   best max(0.035, 0.032) = 0.035 <= 0.0335? no -> not passed.
    # Day 3: target met, consistency not -> trades. t1 profit 0.072 -> goal
    #   stop. 0.035 <= 0.036 -> passed on day 3.
    # Day 4 (funded, f_cons off so no day stop): +0.01 -> payout 0.01.
    check(run_account(cfg, days, 0.01, 0.0, pol(dstop=1.0)),
          "payout", 0.01, 4, 1)
    # Naive:
    # Day 1: 0.02, 0.035, 0.065 >= 0.06 -> stop. best 0.065 <= 0.0325? no.
    # Day 2: t1 profit 0.077 -> stop. 0.065 <= 0.0385? no.
    # Day 3: t1 profit 0.082 -> stop. 0.065 <= 0.041? no.
    # Day 4: profit 0.092. 0.065 <= 0.046? no. Censored in phase 0.
    check(run_account(cfg, days, 0.01, 0.0), "censored", 0.0, 4, 0)


# ============================================================================
# Scenarios that pin a reading recorded in AMBIGUITIES.md. If the fast
# simulator disagrees on one of these, compare readings before calling it a bug.

def test_A1_dstop_mult_switched_off_means_no_day_stop_even_if_need_is_zero():
    days = [[WIN, WIN], [(1.5, 0, 1.5)], [(1.5, 0, 1.5)]]
    cfg = mk(f_cons=0.5)
    # AMBIGUITY 1. risk 0.01, wgoal 0 -> need 0. dstop_mult 1e9 = off ->
    # dstop infinite (the product 1e9 * 0.5 * 0 = 0 is NOT used).
    # Day 1: t1 profit 0.01: 0.01 <= 0.005? no. t2 profit 0.02: 0.02 <= 0.01?
    #   no. best 0.02, no payout.
    # Day 2: profit 0.035: 0.02 <= 0.0175? no.
    # Day 3: profit 0.05: 0.02 <= 0.025? yes -> payout 0.05 on day 3.
    # (With dstop = 0 the trader would stop after t1 on day 1 and be paid
    #  0.04 on day 3.)
    check(run_account(cfg, days, 0.01, 0.0, pol()), "payout", 0.05, 3, 0)


def test_A3_qstop_applies_even_when_funded_consistency_is_off():
    days = [[WIN, WIN, WIN]] * 3
    cfg = mk(f_min_qdays=2, f_qthresh=0.01)        # f_cons 1.0 = off
    # AMBIGUITY 3. risk 0.006, wgoal 0.035, dstop_mult off, qstop 1.
    # Day 1: dstop = min(inf, 0.01 * 1.0) = 0.01: t1 0.006, t2 0.012 -> stop.
    #   q 1.
    # Day 2: dstop 0.01012: stop after t2. profit 0.024, q 2.
    # Day 3: no qstop (q = 2). t1 profit 0.030, t2 profit 0.036 >= 0.035 ->
    #   goal stop. Payout 0.036 on day 3.
    # (If the qstop clause were nested under the consistency condition there
    #  would be no day stop and the payout would come on day 2.)
    check(run_account(cfg, days, 0.006, 0.035, pol(qstop=1)),
          "payout", 0.036, 3, 0)


def test_A4_funded_goal_stop_does_not_require_positive_profit():
    # AMBIGUITY 4. Neutral cfg, wgoal 0. t1 is flat: profit 0 >= min_profit
    # 0, q ok, consistency off -> goal stop, so the +1R trade is not taken.
    # EOD: profit 0 is not > 0 -> no payout -> censored.
    res = run_account(mk(), [[(0, 0, 0), WIN]], 0.01, 0.0)
    check(res, "censored", 0.0, 1, 0)


def test_A5_token_day_counts_as_traded_even_with_no_candidates():
    days = [[(6, 0, 6)], [], [WIN]]
    # AMBIGUITY 5. n_eval 1, tgt1 0.05, eval_min_days 2, risk 0.01.
    # Day 1: profit 0.06, traded 1.
    # Day 2: target met -> token day, counts as traded although the
    #   candidate list is empty. traded 2 -> passed.
    # Day 3 (funded): +0.01 -> payout 0.01.
    res = run_account(mk(n_eval=1, tgt1=0.05, eval_min_days=2), days, 0.01, 0.0)
    check(res, "payout", 0.01, 3, 1)


def test_A6_soft_daily_stop_does_not_feed_mfe_into_the_mode3_high():
    days = [[(-4, -4, 4)], [LOSS]]
    cfg = mk(daily=0.03, daily_soft=1, maxdd=0.05, trail_mode=3)
    # AMBIGUITY 6. risk 0.01.
    # Day 1: DF 0.97 >= MF 0.95. 1 - 0.04 = 0.96 < 0.97 -> soft stop,
    #   bal = 0.97. high is left at 1.0 (the trade's MFE of 1.04 is ignored).
    # Day 2: DF 0.94, MF 0.95. 0.97 - 0.01 = 0.96 >= 0.95 -> survives.
    #   Censored. (If high had become 1.04, MF would be 0.99 -> fail_maxdd.)
    check(run_account(cfg, days, 0.01, 0.0), "censored", 0.0, 2, 0)


# ------------------------------------------------------------- plain runner

if __name__ == "__main__":
    tests = sorted((name, fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (name, exc))
    print("%d tests, %d failed" % (len(tests), failed))
    sys.exit(1 if failed else 0)
