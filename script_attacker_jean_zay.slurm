#!/bin/bash 

#SBATCH --job-name=Attacker                     # Job name
#SBATCH --output=logs/attackerMovieLens.%j.out  # logs (%j = job ID)
#SBATCH --error=logs/attackerMovieLens.%j.err   # logs (%j = job ID)
#SBATCH --hint=nomultithread                    # no hyperthreading
#SBATCH --account=ktn@v100                      # V100 account
#SBATCH --constraint=v100-16g                   # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1                               # allocate 1 node (one machine)
#SBATCH --ntasks-per-node=1                     # allocate 1 task (one processus)
#SBATCH --gres=gpu:1                            # allocate 1 GPU
#SBATCH --cpus-per-task=10                      # allocate 10 CPU per task
#SBATCH --qos=qos_gpu-t3                        # Default QoS, allows 20h
#SBATCH --time=20:00:00                         # Max job duration

set -x
module purge

cd ~/work/decentralizepy_grid5000

module load cuda; 
module load singularity;

./script_attacker.sh $@