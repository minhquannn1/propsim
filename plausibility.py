"""Stage-by-stage rates with all rules on, for comparison with published figures."""
import numpy as np, pandas as pd
from engine import simulate, N_EVAL
from rules import ARCHETYPES
from core import TRADERS, libs, NOB, NOR, SEED

L = libs()
rows = []
for akey, A in ARCHETYPES.items():
    cfg = A["cfg"]; ne = int(cfg[N_EVAL])
    for tkey, tr in TRADERS.items():
        for risk in [0.0025, 0.005, 0.01, 0.015]:
            for smart in (0, 1):
                if tr["ttype"] == 0:
                    lib, b = (L["R"], L["M"], L["F"], L["E"]), NOB
                else:
                    x = tr["lib"]; lib, b = NOR, (L["R" + x], L["M" + x], L["F" + x], L["h" + x])
                o, p, u, s = simulate(cfg, tr["ttype"], risk, tr["edge"], 2, *lib, *b, 50_000, 250, SEED, 20, 0.02, smart)
                funded = s >= ne
                rows.append(dict(archetype=akey, trader=tkey, risk=risk, trader_mode="rule_aware" if smart else "naive",
                                 pass_phase1=(s >= 1).mean() if ne >= 1 else np.nan,
                                 funded=funded.mean() if ne >= 1 else 1.0,
                                 paid=(o == 1).mean(),
                                 paid_given_funded=(o == 1).sum() / max(funded.sum(), 1),
                                 failed=((o == 2) | (o == 3)).mean(), still_open=(o == 0).mean()))
pd.DataFrame(rows).to_csv("plausibility.csv", index=False)
print("done")
