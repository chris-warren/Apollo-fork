"""Check GPU-to-GPU communication across Slurm tasks (one task per GPU).

    srun apptainer exec --nv $SIF python scripts/diagnostics/ddp_check.py nccl 29500

Does what DDP training does first: start a process group, broadcast a Python object
(Lightning's first collective), then all-reduce a gradient-sized tensor and time it.
Prints one OK line from rank 0, or fails with the backend's error. Run by
scripts/diagnostics/ddp_check.sbatch under several NCCL settings and with gloo.
"""
import os
import socket
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

# Apollo generator (16.54 M) + discriminator (2.77 M) parameters: the gradient volume
# DDP all-reduces in each backward pass.
N_GRAD = 19_310_000


def main():
    backend, port = sys.argv[1], sys.argv[2]
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ.update(MASTER_ADDR=os.environ.get("DDP_CHECK_MASTER", socket.gethostname()),
                      MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(world))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:  # lets the script be exercised on a CPU-only machine with gloo
        device = torch.device("cpu")
    dist.init_process_group(backend, timeout=timedelta(seconds=120))

    obj = [f"hello from rank 0 on {socket.gethostname()}" if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)  # the call that failed in training
    assert obj[0].startswith("hello from rank 0"), obj

    check = torch.full((1024,), float(rank + 1), device=device)
    dist.all_reduce(check)
    expected = world * (world + 1) / 2
    assert torch.allclose(check, torch.full_like(check, expected)), (check[0].item(), expected)

    grad = torch.ones(N_GRAD, device=device)
    for _ in range(3):  # warm-up
        dist.all_reduce(grad)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    reps = 10
    for _ in range(reps):
        dist.all_reduce(grad)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - start) / reps * 1000
    if rank == 0:
        print(f"OK backend={backend} world={world} device={device.type}: broadcast and all-reduce "
              f"correct; {N_GRAD * 4 / 1e6:.0f} MB all-reduce {ms:.1f} ms", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
