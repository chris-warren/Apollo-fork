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
apptainer exec -B /project -B /scratch $SIF python scripts/preflight.py
```

Every check should pass; the GPU check just reports that no GPU is visible on a login
node. Fix any FAIL before requesting a GPU (`PREFLIGHT_TRACEBACK=1` shows details).

## 3. Smoke test (inside the allocation)

```bash
salloc --account=def-<pi> --gpus-per-node=h100:1 --cpus-per-task=12 --mem=64G --time=1:30:00
cd ~/projects/def-<pi>/<user>/Apollo-fork
apptainer exec --nv -B /project -B /scratch $SIF python scripts/preflight.py   # now incl. GPU
apptainer exec --nv -B /project -B /scratch $SIF python train.py --conf_dir configs/apollo_smoke.yaml
# second shell on the same node:
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
```

`-B /project -B /scratch` makes the project and scratch file systems visible inside the
container (`~/projects/...` links there); `--nv` exposes the GPU.

## Known gap: ViSQOL (final evaluation only)

`look2hear.yml` lists `visqol==3.3.3`, which is not on PyPI; the authors must have
built Google's ViSQOL from source. It is needed only by `test.py`
(`look2hear/metrics/wrapper.py` imports it to compute the ViSQOL score), not by
training, validation or the smoke test, so the image leaves it out for now. It has to
be added before the final evaluation.
