# Account rules: written specification for an independent reference implementation

This document describes, in words, how one simulated prop-firm account evolves.
Write a plain-Python reference implementation from this text only. Do not look
at any other code in the parent folder. Clarity matters more than speed.

## Units

All money is a fraction of the initial balance. The account starts at balance
1.0 (think $100,000). A profit of 0.02 means 2% of the initial balance.

## Inputs

`run_account(cfg, days, risk, wgoal, policy=None)` where

- `cfg` is a dict with these keys:
  - `n_eval` (0, 1 or 2): number of evaluation phases before the funded stage.
  - `tgt1`, `tgt2`: profit targets of phase 1 and phase 2 (fractions).
  - `eval_min_days`: minimum number of traded days in each evaluation phase.
  - `eval_cons`: evaluation consistency limit c. The rule is "best day <= c x
    total profit". A value >= 0.999 means the rule is off.
  - `daily`: daily loss limit (fraction). A huge value (1e9) means off.
  - `daily_base`: 0 means the day's floor is `day_start_balance - daily`
    (the limit is a fraction of the initial balance). 1 means the floor is
    `day_start_balance * (1 - daily)`.
  - `daily_soft`: 1 means a breach of the daily floor does not fail the
    account; the position is closed exactly at the daily floor and trading
    stops for that day. 0 means a breach fails the account.
  - `maxdd`: maximum loss distance (fraction). 1e9 means off.
  - `trail_mode`: how the maximum-loss floor moves.
    - 0: static. Floor = 1.0 - maxdd, always.
    - 1: floor = (highest closed balance so far, updated after every closed
      trade) - maxdd.
    - 2: floor = (highest end-of-day balance so far) - maxdd. It only moves
      at the end of a day.
    - 3: floor = (highest equity so far, including the best open profit
      reached inside a trade) - maxdd.
    The "highest ... so far" value starts at 1.0.
  - `trail_lock`: 1 means the trailing floor never rises above 1.0 (the
    initial balance). Only relevant for trail_mode 1, 2, 3.
  - `f_min_profit`: minimum profit (fraction) for a payout in the funded stage.
  - `f_min_qdays`: number of qualifying days needed for a payout.
  - `f_qthresh`: a day qualifies if a trade was taken that day and the day's
    profit is at least `f_qthresh x day_start_balance`. A negative value means
    every day with a trade qualifies.
  - `f_cons`: funded-stage consistency limit, same form as `eval_cons`.
  - `split`: trader's profit share.
  - `f_frac`: fraction of the profit that can be withdrawn at once.
  - `cap`: maximum payout (fraction of the initial balance). 1e9 means none.
- `days`: a list, one entry per trading day, in order. Each entry is a list
  of candidate trades for that day in the order the trader would take them.
  A candidate trade is a tuple `(R, MAE, MFE)` in units of the amount risked:
  `R` is the final result, `MAE` (<= 0, and <= R) is the worst open result
  reached during the trade, `MFE` (>= R) is the best open result reached.
  A day's list can be empty.
- `risk`: the amount risked per trade (fraction of the initial balance), so a
  trade changes the balance by `size * R`, where size is `risk` unless the
  policy below shrinks it.
- `wgoal`: the trader only asks for a payout once profit is at least
  `max(f_min_profit, wgoal)`. Call this number `min_profit`.
- `policy`: None for the naive trader, or a dict
  `{"room_frac": x, "dstop_mult": m, "qstop": 0 or 1}` for the rule-aware
  trader (see below). 1e9 for `room_frac` or `dstop_mult` switches that part off.

## Output

A dict with
- `outcome`: "payout", "fail_daily", "fail_maxdd", or "censored" (the list of
  days ran out).
- `payout`: the payout amount (0 unless outcome is "payout").
- `days_used`: number of days processed, counting the day on which the account
  failed or was paid (equal to len(days) if censored).
- `phase`: number of evaluation phases passed.

## State

- `phase` (starts 0). The account is in the funded stage when `phase >= n_eval`.
- `balance` (1.0), `high` (1.0, the "highest ... so far" value for the trailing
  floor), `traded_days` (0), `qualifying_days` (0), `best_day` (0.0, the largest
  single-day profit so far in the current stage; never below 0).

When an evaluation phase is passed, the next stage starts with a fresh
account: balance 1.0, high 1.0, and the three counters back to zero.

## One day

1. Record `day_start = balance`. Compute the daily floor from `day_start`
   (see `daily_base`). It stays fixed all day.
2. Evaluation only: if profit (`balance - 1.0`) already meets the current
   target AND the evaluation consistency rule is currently satisfied (or off),
   the trader does not use any candidate trade today. The day still counts as
   a traded day (the trader places a token trade with no effect on balance).
   If the target is met but the consistency rule is NOT satisfied, the trader
   keeps trading normally.
3. Otherwise go through the day's candidate trades in order. For each one:
   a. Compute the maximum-loss floor from the current state (see
      `trail_mode`, `trail_lock`). The effective floor is the higher of the
      daily floor and the maximum-loss floor.
   b. Size. Naive trader: size = risk. Rule-aware trader:
      size = min(risk, room_frac x (balance - effective floor)); if that is
      below 0.1 x risk, the trader takes no more trades today (and does not
      take this one).
   c. Breach test: if `balance + size x MAE` is below the effective floor
      (strictly below; use a tolerance of 1e-12, i.e. breach when
      `balance + size*MAE < floor - 1e-12`):
      - if `daily_soft` is 1 and the daily floor is >= the maximum-loss
        floor: set balance to the daily floor, the day counts as traded, and
        no more trades are taken today. The account survives.
      - otherwise the account fails immediately: "fail_daily" if the daily
        floor is >= the maximum-loss floor, else "fail_maxdd".
   d. If no breach: the best open equity of the trade is
      `balance + size x MFE`. Then `balance += size x R`. The day counts as
      traded. Update `high`: for trail_mode 1 with the new balance; for
      trail_mode 3 with the best open equity of the trade.
      For trail_mode 3 only: after updating `high`, recompute the
      maximum-loss floor; if the new balance is below it (same tolerance), the
      account fails with "fail_maxdd" (the trade gave back more than the
      trailing distance after making a new equity high).
   e. Rule-aware day stop: if `balance - day_start >= dstop`, stop trading for
      the day. `dstop` is computed once at the start of the day (using the
      state at that moment), like this. Start with dstop = infinity.
      - Funded stage: if the funded consistency rule is on and dstop_mult is
        not switched off: `need = max(min_profit, f_min_qdays x max(f_qthresh, 0))`
        and `dstop = dstop_mult x f_cons x need`. Then, if `qstop` is 1 and
        `f_qthresh > 0` and `qualifying_days < f_min_qdays`:
        `dstop = min(dstop, f_qthresh x day_start)`.
      - Evaluation: if the evaluation consistency rule is on and dstop_mult is
        not switched off: `dstop = dstop_mult x eval_cons x target`.
      The naive trader has no day stop.
   f. Goal stop (both trader types). Evaluation: if profit now meets the
      target, stop trading for the day. Funded: if all payout conditions would
      be met at this moment, stop trading for the day. "At this moment" means:
      profit >= min_profit; the number of qualifying days, counting today as
      one if today's profit so far qualifies, is >= f_min_qdays; and the
      funded consistency rule holds (or is off) using
      max(best_day, today's profit so far) as the best day.
4. End of day (only reached if the account did not fail):
   - today's profit = balance - day_start. If the day counted as traded,
     traded_days += 1. best_day = max(best_day, today's profit).
     If the day counted as traded and qualifies (see `f_qthresh`),
     qualifying_days += 1.
   - trail_mode 2: update `high` with the end-of-day balance.
   - Evaluation: the phase is passed if profit >= target AND
     traded_days >= eval_min_days AND (consistency off OR
     best_day <= eval_cons x profit). On passing, phase += 1 and the account
     is reset as described above.
   - Funded: the payout happens if profit >= min_profit AND profit > 0 AND
     qualifying_days >= f_min_qdays AND (consistency off OR
     best_day <= f_cons x profit). Payout amount =
     min(split x f_frac x profit, cap). The run ends with outcome "payout".

The current target is `tgt1` in phase 0 and `tgt2` in any later evaluation phase.

A day on which no trade was taken (empty list, or the rule-aware trader sat
out) does not count as traded.

## Deliverables

- `reference.py` with `run_account` as specified, plain Python, no numpy needed.
- `test_reference.py` (pytest style, runnable with `python3 -m pytest -q` or
  plain `python3 test_reference.py`) with at least 25 hand-worked scenarios.
  Each scenario must state in a comment the expected result worked out by hand
  from this text, and assert it. Cover at least: static vs each trailing mode;
  the lock; hard and soft daily limit with both bases; a breach decided by
  which floor is higher; qualifying days with and without a threshold;
  consistency in the evaluation (including target met but consistency not yet
  satisfied) and in the funded stage; minimum days; payout cap and f_frac;
  the wgoal; two-phase reset; the rule-aware sizing, sit-out and day stops;
  trail_mode 3 with the give-back failure.
