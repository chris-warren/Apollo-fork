#!/bin/bash
# Evaluate a finished run on the packed test set with test.py, on one H100.
#
#   sbatch --export=ALL,RUN=outputs/exps/Apollo-1node-4xH100,MAX=250 --time=1:00:00 scripts/test_1gpu.sbatch
#   sbatch --export=ALL,RUN=outputs/exps/Apollo-1node-4xH100 scripts/test_1gpu.sbatch
#
# RUN: the run folder (its config.yaml and best_model.pth are used). MAX: optional, only the
# first MAX clips per dataset and bitrate (the development subset; 250 = 3,000 clips).
# Results: <RUN>/results/full/ or <RUN>/results/first<MAX>_per_bitrate/ (metrics.csv per clip,
# metrics_breakdown.csv per dataset, bitrate and category, all and aligned-only clips).
# Estimates are not saved (about 127 GB for the full set); add --save-estimates via ARGS.
#SBATCH --job-name=apollo-test
#SBATCH --account=def-gtzan_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail

RUN=${RUN:?set RUN=<run folder>, e.g. outputs/exps/Apollo-1node-4xH100}
SIF=${SIF:-$SLURM_SUBMIT_DIR/containers/apollo.sif}
cd "$SLURM_SUBMIT_DIR"
module load apptainer
# APOLLO_DATA from .env, read by the same loader train.py and test.py use.
DATA=$(apptainer exec "$SIF" python -c "import os; from dotenv_loader import load_env; load_env(verbose=False); print(os.environ['APOLLO_DATA'])")
TEST_H5=$DATA/test/test.hdf5
[[ -f $TEST_H5 ]] || { echo "no test set at $TEST_H5"; exit 1; }

echo "job $SLURM_JOB_ID on $(hostname) | run $RUN | max per bitrate ${MAX:-all} | commit $(git rev-parse --short HEAD) | $(date)"
apptainer exec --nv "$SIF" python test.py --conf_dir "$RUN/config.yaml" \
    --test_h5 "$TEST_H5" ${MAX:+--max-per-bitrate "$MAX"} ${ARGS:-}
echo "done $(date)"
