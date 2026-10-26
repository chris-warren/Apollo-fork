#!/bin/bash
# Submit Apollo training (run from the repository root on a login node).
#
#   bash scripts/submit_train.sh [config] [--chain N] [extra sbatch options...]
#
#   bash scripts/submit_train.sh configs/apollo_1node_test.yaml --time=0:30:00
#   bash scripts/submit_train.sh configs/apollo_1node.yaml --chain 4
#
# --chain N submits N jobs, each starting after the previous one ends (for any reason),
# so a run longer than the job time limit continues from its last checkpoint. Cancel the
# remaining jobs with scancel once training has finished (early stopping).
set -euo pipefail

CONFIG=configs/apollo_1node.yaml
if [[ $# -gt 0 && $1 != --* ]]; then CONFIG=$1; shift; fi
CHAIN=1
if [[ ${1:-} == --chain ]]; then CHAIN=$2; shift 2; fi

[[ -f $CONFIG ]] || { echo "no such config: $CONFIG"; exit 1; }
[[ -f .env ]] || { echo "no .env here (copy .env.example and set APOLLO_DATA)"; exit 1; }
SIF=${SIF:-$PWD/containers/apollo.sif}
[[ -f $SIF ]] || { echo "no container at $SIF (build it: bash containers/build.sh containers/apollo.sif)"; exit 1; }
mkdir -p outputs/slurm

prev=""
for ((i = 1; i <= CHAIN; i++)); do
    dep=${prev:+--dependency=afterany:$prev}
    prev=$(sbatch --parsable $dep --export=ALL,CONFIG="$CONFIG",SIF="$SIF" "$@" scripts/train_1node.sbatch)
    echo "submitted job $prev ($i/$CHAIN) with $CONFIG${dep:+, after $dep}"
done
echo "logs: outputs/slurm/apollo-1node-<jobid>.out (+ -gpu.csv)    run: outputs/exps/<exp.name>/    status: squeue -u \$USER"
