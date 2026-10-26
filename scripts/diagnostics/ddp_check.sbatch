#!/bin/bash
# Which multi-GPU communication setting works on this node? One short 4-GPU job that runs
# scripts/diagnostics/ddp_check.py under several settings, one after another.
#
#   mkdir -p outputs/slurm && sbatch scripts/diagnostics/ddp_check.sbatch
#
# Results: outputs/manual/<date>_ddp_check_<jobid>/ (one log per setting, NCCL topology
# dumps, nvidia-smi topology) and a summary at the end of outputs/slurm/ddp-check-<id>.out.
#SBATCH --job-name=ddp-check
#SBATCH --account=def-gtzan_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -uo pipefail  # no -e: a failing setting must not stop the others

cd "$SLURM_SUBMIT_DIR"
module load apptainer
SIF=${SIF:-$SLURM_SUBMIT_DIR/containers/apollo.sif}
OUT=outputs/manual/$(date +%F)_ddp_check_$SLURM_JOB_ID
mkdir -p "$OUT"
export DDP_CHECK_MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "job $SLURM_JOB_ID on $(hostname) | commit $(git rev-parse --short HEAD) | results in $OUT"

nvidia-smi topo -m > "$OUT/nvidia-smi-topo.txt" 2>&1
nvidia-smi nvlink -s > "$OUT/nvidia-smi-nvlink.txt" 2>&1
nvidia-smi -q | grep -iE "driver version|cuda version" | head -2

port=29500
run() {  # run <name> <backend> [ENV=VALUE ...]
    local name=$1 backend=$2; shift 2
    port=$((port + 1))
    env "$@" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV NCCL_TOPO_DUMP_FILE="$OUT/topo-$name.xml" \
        timeout 240 srun apptainer exec --nv "$SIF" python scripts/diagnostics/ddp_check.py "$backend" "$port" \
        > "$OUT/$name.log" 2>&1
    local status=$?
    local line
    line=$(grep -m1 "^OK " "$OUT/$name.log" || grep -m1 -E "Error|error|Last error" "$OUT/$name.log" || echo "no output")
    printf '%-26s exit %-4s %s\n' "$name" "$status" "$line" | tee -a "$OUT/summary.txt"
}

run nccl_default        nccl
run nccl_p2p_off        nccl NCCL_P2P_DISABLE=1
run nccl_p2p_level_phb  nccl NCCL_P2P_LEVEL=PHB
run nccl_p2p_shm_off    nccl NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
run gloo                gloo

echo "== summary (also $OUT/summary.txt)"
cat "$OUT/summary.txt"
