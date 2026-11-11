"""Plain-Python reference implementation of the account rules in SPEC.md.

Written from the specification text only. Clarity is preferred over speed.
Every block is labelled with the step of the spec ("One day", steps 1 to 4)
that it implements. Points where the text had to be interpreted are listed in
AMBIGUITIES.md and marked here with "AMBIGUITY n".
"""

TOL = 1e-12            # tolerance of the two breach tests (steps 3c and 3d)
CONS_OFF = 0.999       # a consistency limit >= this value means "rule off"
SWITCH_OFF = 1e9       # dstop_mult >= this value means "switched off" (AMBIGUITY 1)
INFINITY = float("inf")


# ---------------------------------------------------------------- small rules

def consistency_on(limit):
    """A consistency limit c is active unless c >= 0.999."""
    return limit < CONS_OFF


def consistency_ok(limit, best_day, profit):
    """True if the rule "best day <= c x total profit" holds, or is off."""
    if not consistency_on(limit):
        return True
    return best_day <= limit * profit


def daily_floor_of(cfg, day_start):
    """Daily loss floor for a day that started at balance `day_start`."""
    if cfg["daily_base"] == 1:
        return day_start * (1.0 - cfg["daily"])
    return day_start - cfg["daily"]


def maxloss_floor_of(cfg, high):
    """Maximum-loss floor given the current "highest ... so far" value."""
    if cfg["trail_mode"] == 0:
        return 1.0 - cfg["maxdd"]
    floor = high - cfg["maxdd"]
    if cfg["trail_lock"] == 1 and floor > 1.0:
        floor = 1.0
    return floor


def day_qualifies(cfg, traded, day_profit, day_start):
    """A day qualifies if a trade was taken and the profit meets the threshold."""
    if not traded:
        return False
    if cfg["f_qthresh"] < 0:
        return True
    return day_profit >= cfg["f_qthresh"] * day_start


def current_target(cfg, phase):
    """tgt1 in phase 0, tgt2 in any later evaluation phase."""
    return cfg["tgt1"] if phase == 0 else cfg["tgt2"]


def day_stop_level(cfg, policy, funded, target, min_profit, qualifying_days,
                   day_start):
    """Step 3e: the rule-aware trader's day stop, computed once per day."""
    dstop = INFINITY
    if policy is None:
        return dstop                      # the naive trader has no day stop
    mult = policy["dstop_mult"]
    mult_on = mult < SWITCH_OFF           # AMBIGUITY 1
    if funded:
        if consistency_on(cfg["f_cons"]) and mult_on:
            need = max(min_profit,
                       cfg["f_min_qdays"] * max(cfg["f_qthresh"], 0.0))
            dstop = mult * cfg["f_cons"] * need
        # AMBIGUITY 3: the qstop clause is applied whether or not the clause
        # above was active (that is why dstop starts at infinity).
        if (policy["qstop"] == 1 and cfg["f_qthresh"] > 0
                and qualifying_days < cfg["f_min_qdays"]):
            dstop = min(dstop, cfg["f_qthresh"] * day_start)
    else:
        if consistency_on(cfg["eval_cons"]) and mult_on:
            dstop = mult * cfg["eval_cons"] * target
    return dstop


def result(outcome, payout, days_used, phase):
    return {"outcome": outcome, "payout": payout, "days_used": days_used,
            "phase": phase}


# ------------------------------------------------------------------ main loop

def run_account(cfg, days, risk, wgoal, policy=None):
    """Simulate one account over `days`. See SPEC.md for the meaning of inputs."""
    n_eval = cfg["n_eval"]
    trail_mode = cfg["trail_mode"]
    min_profit = max(cfg["f_min_profit"], wgoal)

    # State
    phase = 0
    balance = 1.0
    high = 1.0
    traded_days = 0
    qualifying_days = 0
    best_day = 0.0

    for day_index, trades in enumerate(days):
        days_used = day_index + 1
        funded = phase >= n_eval          # stage is fixed for the whole day
        target = None if funded else current_target(cfg, phase)

        # ---- Step 1: day start and daily floor (fixed all day)
        day_start = balance
        daily_floor = daily_floor_of(cfg, day_start)

        traded = False

        # ---- Step 2: evaluation only, target already met and consistency fine
        sit_on_target = False
        if not funded:
            profit = balance - 1.0
            if profit >= target and consistency_ok(cfg["eval_cons"], best_day,
                                                   profit):
                sit_on_target = True
                traded = True             # token trade, no effect on balance

        # ---- Step 3: go through the candidate trades
        if not sit_on_target:
            dstop = day_stop_level(cfg, policy, funded, target, min_profit,
                                   qualifying_days, day_start)

            for (r_mult, mae, mfe) in trades:
                # 3a. floors
                maxloss_floor = maxloss_floor_of(cfg, high)
                floor = max(daily_floor, maxloss_floor)

                # 3b. size
                if policy is None:
                    size = risk
                else:
                    size = min(risk, policy["room_frac"] * (balance - floor))
                    if size < 0.1 * risk:
                        break             # no more trades today, not this one

                # 3c. breach test on the worst open result
                if balance + size * mae < floor - TOL:
                    daily_is_binding = daily_floor >= maxloss_floor
                    if cfg["daily_soft"] == 1 and daily_is_binding:
                        balance = daily_floor
                        traded = True
                        break             # account survives, day is over
                    if daily_is_binding:
                        return result("fail_daily", 0.0, days_used, phase)
                    return result("fail_maxdd", 0.0, days_used, phase)

                # 3d. no breach: book the trade
                best_open_equity = balance + size * mfe
                balance += size * r_mult
                traded = True
                if trail_mode == 1:
                    high = max(high, balance)
                elif trail_mode == 3:
                    high = max(high, best_open_equity)
                    if balance < maxloss_floor_of(cfg, high) - TOL:
                        return result("fail_maxdd", 0.0, days_used, phase)

                # 3e. rule-aware day stop (dstop is infinity for the naive trader)
                if balance - day_start >= dstop:
                    break

                # 3f. goal stop, both trader types
                profit = balance - 1.0
                if not funded:
                    if profit >= target:
                        break
                else:
                    profit_so_far = balance - day_start
                    qdays_now = qualifying_days
                    if day_qualifies(cfg, True, profit_so_far, day_start):
                        qdays_now += 1
                    best_now = max(best_day, profit_so_far)
                    if (profit >= min_profit
                            and qdays_now >= cfg["f_min_qdays"]
                            and consistency_ok(cfg["f_cons"], best_now, profit)):
                        break

        # ---- Step 4: end of day
        day_profit = balance - day_start
        if traded:
            traded_days += 1
        best_day = max(best_day, day_profit)
        if day_qualifies(cfg, traded, day_profit, day_start):
            qualifying_days += 1

        if trail_mode == 2:
            high = max(high, balance)

        profit = balance - 1.0
        if not funded:
            if (profit >= target
                    and traded_days >= cfg["eval_min_days"]
                    and consistency_ok(cfg["eval_cons"], best_day, profit)):
                phase += 1
                balance = 1.0
                high = 1.0
                traded_days = 0
                qualifying_days = 0
                best_day = 0.0
        else:
            if (profit >= min_profit
                    and profit > 0
                    and qualifying_days >= cfg["f_min_qdays"]
                    and consistency_ok(cfg["f_cons"], best_day, profit)):
                payout = min(cfg["split"] * cfg["f_frac"] * profit, cfg["cap"])
                return result("payout", payout, days_used, phase)

    return result("censored", 0.0, len(days), phase)
