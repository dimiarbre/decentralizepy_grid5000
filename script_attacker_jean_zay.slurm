#!/bin/bash 

#SBATCH --job-name=Attacker                     # Job name
#SBATCH --output=logs/attackerMovieLens.%j.out  # logs (%j = job ID)
#SBATCH --error=logs/attackerMovieLens.%j.err   # logs (%j = job ID)
#SBATCH --hint=nomultithread                    # no hyperthreading
#SBATCH --account=ggl@v100                      # V100 account
#SBATCH --constraint=v100-16g                   # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1                               # allocate 1 node (one machine)
#SBATCH --ntasks-per-node=1                     # allocate 1 task (one processus)
#SBATCH --gres=gpu:1                            # allocate 1 GPU
#SBATCH --cpus-per-task=10                      # allocate 10 CPU per task
#SBATCH --qos=qos_gpu-t3                        # Default QoS, allows 20h
#SBATCH --time=20:00:00                         # Max job duration

module purge

cd ~/work/decentralizepy_grid5000

# Script to deploy on a jupyter notebook, for a control node that will launch jobs & perform attacks.
module load cuda; 
module load singularity;


# Default values
NB_WORKERS=""
BATCH_SIZE=""
NB_AGENTS=""
NB_MACHINES=""
EXPERIMENT_DIR=""

# Usage function to show how to use the script
usage() {
    echo "Usage: $0 --experiment_dir <EXPERIMENT_DIR> --nb_workers <NB_WORKERS> --batch_size <BATCH_SIZE> --nb_agents <NB_AGENTS> --nb_machines <NB_MACHINES>"
    exit 1
}

# Parse the arguments using a while loop
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment_dir) EXPERIMENT_DIR="$2"; shift ;;
        --nb_workers) NB_WORKERS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --nb_agents) NB_AGENTS="$2"; shift ;;
        --nb_machines) NB_MACHINES="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Check if all required parameters are set
if [[ -z "$EXPERIMENT_DIR" || -z "$NB_WORKERS" || -z "$BATCH_SIZE" || -z "$NB_AGENTS" || -z "$NB_MACHINES" ]]; then
    echo "Error: Missing required parameters."
    usage
fi

# Build and run the Apptainer (or Singularity) container command
srun singularity exec \
    --bind "$EXPERIMENT_DIR:/experiments_to_attack" my_container.sif \
    --bind $SCRATCH/datasets:/datasets \
    --bind $WORK/decentralizepy_grid5000:/decentralizepy_grid5000 \
    $SINGULARITY_ALLOWED_DIR/attacker_container.sif \
    --nb_workers "$NB_WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --nb_agents "$NB_AGENTS" \
    --nb_machines "$NB_MACHINES" \
    /experiments_to_attack/
