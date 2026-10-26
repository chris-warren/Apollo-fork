# Running Apollo on DRAC

Everything is prepared on a login node, so GPU time is spent only on the run.

The container (`containers/apollo.def`) holds the environment: torch 2.0.0, torchaudio
2.0.1, FFmpeg 4.2.2, the packages pinned in `look2hear.yml`, and wandb. The code is not
inside it; it runs from your git checkout, so code changes never need a rebuild. A
`.sif` is a single file (a conda environment would be tens of thousands), so it barely
touches the file quota.

## 1. Build the container (login node, no root, about 20–30 min)

```bash
cd ~/projects/def-<pi>/<user>
git clone git@github.com:chris-warren/Apollo-fork.git && cd Apollo-fork
git checkout data-prep-splits
module load apptainer
bash containers/build.sh ~/projects/def-<pi>/<user>/containers/apollo.sif
```

`build.sh` first tries a normal unprivileged build. If the cluster's Apptainer does not
allow that, it builds the same definition through a writable sandbox instead; either
way the result is the same image. Temporary files go to `$SCRATCH` and are deleted.

## 2. Settings and preflight check (login node)

```bash
cp .env.example .env     # set APOLLO_DATA to your apollo_v1 folder; keep WANDB_MODE=offline
SIF=~/projects/def-<pi>/<user>/containers/apollo.sif
apptainer exec $SIF python scripts/preflight.py
```

Every check should pass; the GPU check just reports that no GPU is visible on a login
node. Fix any FAIL before requesting a GPU (`PREFLIGHT_TRACEBACK=1` shows details).

## 3. Smoke test (inside the allocation)

Everything a run produces goes in the gitignored `outputs/` folder of the repo:
`outputs/exps/<exp.name>/` (training runs), `outputs/slurm/` (batch-job logs) and
`outputs/manual/<date>_<label>/` (anything run by hand, like this).

```bash
salloc --account=def-gtzan_gpu --gpus-per-node=h100:1 --cpus-per-task=12 --mem=64G --time=1:00:00
module load apptainer
cd ~/Apollo-fork
OUT=outputs/manual/$(date +%F)_smoke_1xh100 && mkdir -p $OUT
apptainer exec --nv $SIF python scripts/preflight.py 2>&1 | tee $OUT/preflight.log   # 6/6 incl. GPU
apptainer exec --nv $SIF python train.py --conf_dir configs/apollo_smoke.yaml 2>&1 | tee $OUT/train.log
# second shell on the same node (srun --jobid=<id> --overlap --pty bash), from ~/Apollo-fork:
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 | tee $OUT/gpu.csv
```

On DRAC `/scratch` is bound into containers automatically and your home directory is
always visible; add `-B <dir>` only for data or code on another file system. `--nv`
exposes the GPU.

## 4. Full run: one node, 4 x H100

`configs/apollo_1node.yaml` = `configs/apollo.yaml` with 4 GPUs x batch 2 = 8 samples per
optimiser step (the paper: 8 GPUs x batch 1). `training_step` evaluates the GAN losses
per sample, so this is the paper's step exactly (`tests/test_training_equivalence.py`).
From the smoke test: about 46 min and 3.1 GPU-hours per epoch.

```bash
# from the repository root, on a login node
bash scripts/submit_train.sh configs/apollo_1node_test.yaml --time=0:30:00   # ~10 min test first
bash scripts/submit_train.sh configs/apollo_1node.yaml --chain 4             # then the run
squeue -u $USER;  tail -f outputs/slurm/apollo-1node-<jobid>.out
```

Every job starts with `train.py --resume auto`: it continues from the run's
`outputs/exps/<name>/checkpoints/last.ckpt` if there is one, so `--chain N` (N jobs of 3 days,
each starting when the previous ends) carries a run past the job time limit; a job that
stops mid-epoch loses only that epoch. Cancel leftover chained jobs (`scancel`) once early
stopping has ended training. To rerun the test from scratch, delete
`outputs/exps/Apollo-1node-test` first. Each job also logs GPU utilisation and memory
every 10 s to `outputs/slurm/apollo-1node-<jobid>-gpu.csv`. The test config writes a
profiler report per GPU process to `outputs/exps/Apollo-1node-test/` showing how much
time training spent waiting for data (`train_dataloader_next`).

## H100s: set APOLLO_SDPA_BACKEND=math

torch 2.0's fused "memory-efficient" attention kernel crashes on H100 (sm_90) with
`CUDA error: an illegal instruction was encountered` (it can also surface later as
`cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`). FlashAttention does not run in fp32, so
on H100s Apollo's attention must use torch's plain math implementation:
`APOLLO_SDPA_BACKEND=math` in `.env` (the default in `.env.example`). `train.py`,
`test.py` and `scripts/preflight.py` apply it; the preflight fails on an H100 without it.
It computes the same `softmax(QK^T/sqrt(d))V` in fp32: on a GPU where both paths run,
generator outputs differed by 7.7e-7 and gradients by 2.6e-7 (relative). On the
authors' GPUs (RTX 4090, fp32) torch used the memory-efficient kernel; leave the
setting at `default` on GPUs where that kernel works.

## Known gap: ViSQOL (final evaluation only)

`look2hear.yml` lists `visqol==3.3.3`, which is not on PyPI; the authors must have
built Google's ViSQOL from source. It is needed only by `test.py`
(`look2hear/metrics/wrapper.py` imports it to compute the ViSQOL score), not by
training, validation or the smoke test, so the image leaves it out for now. It has to
be added before the final evaluation.
