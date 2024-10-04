#!/bin/bash 

#OAR -n attacker
#OAR -q production 
#OAR -l gpu=1,walltime=36:00:00
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

cd ~/scratch/decentralizepy_grid5000
chmod +x setup_control.sh
./setup_control.sh
source /tmp/custom_python/bin/activate;

python attacks/perform_attacks.py $@