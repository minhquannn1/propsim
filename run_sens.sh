#!/bin/sh
# Robustness runs at 0.5% and 1% risk, 20,000 paths each
C="RISKS=0.005,0.01 SIMS=20000"
python3 run_final.py _wgoal1 $C WGOAL=0.01
python3 run_final.py _wgoal4 $C WGOAL=0.04
python3 run_final.py _spread2 $C LIBS=_spread2
python3 run_final.py _block5 $C BLOCK=5
python3 run_final.py _block60 $C BLOCK=60
python3 run_final.py _h125 $C H=125
python3 run_final.py _h500 $C H=500
echo ALLDONE
