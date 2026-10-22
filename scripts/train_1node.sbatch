#!/bin/bash
# One-node Apollo training: 4 x H100, one task per GPU (Lightning DDP over Slurm tasks).
# Submit through scripts/submit_train.sh, which also creates slurm-logs/.
#
# Always starts with --resume auto: if this run's last.ckpt exists the job continues
# from it (weights, optimisers, schedulers, epoch, early-stopping state), otherwise it
# starts fresh. Resubmitting or chaining the same job therefore just continues training.
#SBATCH --job-name=apollo-1node
#SBATCH --account=def-gtzan_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

CONFIG=${CONFIG:-configs/apollo_1node.yaml}
SIF=${SIF:-$SLURM_SUBMIT_DIR/containers/apollo.sif}
cd "$SLURM_SUBMIT_DIR"
module load apptainer

echo "job $SLURM_JOB_ID on $(hostname) | config $CONFIG | commit $(git rev-parse --short HEAD) | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# One task per GPU: Lightning reads SLURM_* to set ranks; .env is loaded by train.py.
srun apptainer exec --nv "$SIF" python train.py --conf_dir "$CONFIG" --resume auto
