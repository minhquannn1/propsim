# Ambiguities found in SPEC.md and the reading chosen in reference.py

Each point gives the text in question, the reading implemented (always the most
literal one I could find), the alternative, and when the two readings differ.
Points 1, 3, 4, 5 and 6 are pinned by a dedicated test (`test_A1`, `test_A3`,
`test_A4`, `test_A5`, `test_A6`), so a disagreement with the fast simulator on
one of those tests is a difference of reading before it is a bug.

## Points that can change a result

1. **What "switched off" means for `dstop_mult` (Inputs, step 3e).**
   The spec says "1e9 ... switches that part off" and step 3e says "if ...
   dstop_mult is not switched off".
   Chosen: `dstop_mult >= 1e9` is off, and then the consistency day stop is not
   computed at all (dstop stays infinite).
   Alternative: no special case, just compute `1e9 x cons x need`.
   They differ when `need` (or the evaluation target) is 0: the product is then
   0 and the trader would stop after the first trade of every non-losing day.
   The exact threshold (`>= 1e9` rather than `== 1e9` or `> 1e8`) is my choice.

2. **What "switched off" means for `room_frac` (Inputs, step 3b).**
   Step 3b gives the size formula with no "unless switched off" condition
   (unlike step 3e).
   Chosen: no special case. `size = min(risk, room_frac x (balance - floor))` is
   always evaluated, so 1e9 switches sizing off only through the size of the
   number.
   Alternative: `room_frac >= 1e9` means size = risk, no sit-out test.
   They differ only when `balance - effective floor` is zero or negative within
   the 1e-12 tolerance (the balance sits exactly on a floor): the chosen
   reading sits out, the alternative takes the trade at full risk.

3. **Scope of the `qstop` clause (step 3e, funded).**
   "Funded stage: if the funded consistency rule is on and dstop_mult is not
   switched off: need = ..., dstop = .... Then, if qstop is 1 and ...".
   Chosen: the "Then, if qstop ..." sentence is independent of the consistency
   condition. It applies in the funded stage whenever `qstop == 1`,
   `f_qthresh > 0` and `qualifying_days < f_min_qdays`, even with `f_cons` off
   or `dstop_mult` off (this is why dstop starts at infinity and the clause
   uses `min`).
   Alternative: the qstop clause is nested and only applies when the
   consistency day stop was computed.

4. **Funded goal stop and "profit > 0" (step 3f against step 4).**
   Step 3f says "if all payout conditions would be met at this moment" and then
   defines "at this moment" with three conditions that do not include the
   `profit > 0` condition of step 4.
   Chosen: the explicit three-condition definition. The goal stop can therefore
   fire with profit exactly 0 (when `min_profit` is 0) or negative (when
   `min_profit` is negative), and the day then ends with no payout.
   Alternative: also require `profit > 0` for the goal stop.

5. **Token day with an empty candidate list (step 2 against the last
   paragraph).**
   Step 2 says the day "still counts as a traded day" when the trader sits on a
   met target. The last paragraph says a day with an empty list does not count
   as traded.
   Chosen: step 2 wins. In the evaluation, a day that starts with the target
   met and consistency satisfied (or off) counts as traded whether or not the
   candidate list is empty. An empty list counts as not traded in every other
   case.

6. **Soft daily stop and the trailing `high` (step 3c).**
   On a soft breach the text only says: set balance to the daily floor, count
   the day as traded, stop for the day.
   Chosen: nothing else happens. `high` is not updated from that trade, so in
   trail_mode 3 the stopped trade's MFE is ignored, and steps 3d, 3e and 3f are
   skipped for it.
   Alternative: feed the stopped trade's best open equity into `high` in mode 3.

7. **Evaluation goal stop ignores consistency (step 2 against step 3f).**
   Step 2 says that with the target met but consistency not satisfied the
   trader "keeps trading normally". Step 3f says, for the evaluation, "if
   profit now meets the target, stop trading for the day", with no consistency
   condition.
   Chosen: both, literally. On such a day the trader takes candidate trades
   until, after a trade, profit is at or above the target, which usually means
   exactly one trade per day while the balance stays above the target.

8. **Stage on the day a phase is passed (step 4).**
   Chosen: the stage (evaluation or funded) is fixed at the start of the day.
   On the day the last evaluation phase is passed only the evaluation check
   runs; the reset happens at the end of that day and the funded stage starts
   the next day. No funded payout check is made on the passing day.

9. **Exact comparisons and floating point.**
   The 1e-12 tolerance is applied only to the two breach tests (steps 3c and
   3d), as stated. Every other comparison (`profit >= target`,
   `best_day <= c x profit`, `balance - day_start >= dstop`, the qualifying
   threshold, `size < 0.1 x risk`, `profit > 0`) is exact, on values computed
   as written: profit is `balance - 1.0`, today's profit is
   `balance - day_start`. Cases that are exact ties in decimal arithmetic (for
   example two equal days against `eval_cons = 0.5`) are decided by rounding
   and may come out differently in an implementation that accumulates profit
   in another order. The tests avoid such ties, except where binary-exact
   numbers are used.

## Smaller points (stated for completeness, unlikely to matter)

10. `daily_is_binding` on an exact tie: the spec says "daily floor >= the
    maximum-loss floor", so a tie is treated as daily (`fail_daily`, or the
    soft stop). Implemented as written.
11. "Consistency off" is implemented as `c >= 0.999`, for both `eval_cons` and
    `f_cons`. With the rule on and total profit zero or negative the test
    `best_day <= c x profit` is applied literally (true only if both sides are
    0 or the product is at least best_day).
12. A target of 0 in an evaluation phase is taken literally: a fresh account
    has profit 0 >= 0, so the trader places token days from day 1 and the phase
    passes as soon as `eval_min_days` is reached.
13. `f_qthresh == 0` is not "negative", so a traded day then qualifies only if
    its profit is >= 0. Qualifying days are also counted during the evaluation;
    this has no effect because the counter is reset on passing.
14. `best_day` is updated at the end of every day, traded or not (a day without
    trades has profit 0, so nothing changes). It is never fed intraday; step 3f
    uses `max(best_day, today's profit so far)` without storing it.
15. In trail_mode 3 `high` is updated only with the best open equity
    `balance_before + size x MFE` (this is >= the new balance because
    MFE >= R). The give-back check is run after every booked trade, not only
    when a new high was made; the two are equivalent because the MAE test has
    already passed against the old floor.
16. `trail_lock` is ignored in trail_mode 0. In trail_mode 2 `high` is updated
    at the end of every day, including days without trades.
17. The day stop `dstop` uses the state at the start of the day: `day_start`,
    the `qualifying_days` count before today, and the current phase's target.
    In the funded stage with `f_cons` off there is no consistency day stop even
    if `dstop_mult` is set.
18. Rule-aware sit-out: "below 0.1 x risk" is strict, so a size exactly equal
    to `0.1 x risk` is traded. If earlier trades were taken that day, the day
    still counts as traded.
19. Output conventions not spelled out: `payout` is 0.0 for every outcome other
    than "payout"; `phase` is the number of phases passed at the moment the run
    ends; with an empty `days` list the result is "censored" with
    `days_used = 0`.
20. `policy` is assumed to contain all three keys when it is not None, and
    step 2 (sitting on a met target) applies to both trader types.
