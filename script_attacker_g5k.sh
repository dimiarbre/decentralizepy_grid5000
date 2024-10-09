#!/bin/bash 

#OAR -n attacker
#OAR -q production 
#OAR -l gpu=1,walltime=36:00:00
#OAR -O logs/OAR_%jobid%.out
#OAR -E logs/OAR_%jobid%.err

set -e 

cd ~/scratch/decentralizepy_grid5000


module load cuda; 
module load singularity;

export DATASETS_DIR=~/datasets;
export DECENTRALIZEPY_DIR=~/scratch/decentralizepy_grid5000;
export CONTAINER_FILE=$DECENTRALIZEPY_DIR/attacker_container.sif;

./script_attacker.sh "$@"