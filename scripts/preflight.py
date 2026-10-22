"""Check the environment, data and config before spending GPU time.

    apptainer exec --nv apollo.sif python scripts/preflight.py [--config configs/apollo_smoke.yaml]

Runs on a login node (GPU checks are skipped when there is no GPU) and inside an
allocation (GPU checks run). Prints PASS/FAIL per check; exit status 1 if any fail.
"""
import argparse
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EXPECTED = {"torch": "2.0.0", "torchaudio": "2.0.1", "pytorch_lightning": "2.2.1"}
results = []


def check(name):
    def wrap(fn):
        start = time.time()
        try:
            detail = fn()
            results.append(True)
            print(f"PASS  {name}" + (f": {detail}" if detail else "") + f"  ({time.time() - start:.1f}s)", flush=True)
        except Exception as e:  # report and keep going, so one run shows every problem
            results.append(False)
            print(f"FAIL  {name}: {type(e).__name__}: {e}", flush=True)
            if os.environ.get("PREFLIGHT_TRACEBACK"):
                traceback.print_exc()
        return fn
    return wrap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(REPO, "configs", "apollo_smoke.yaml"))
    args = p.parse_args()
    state = {}

    @check("package versions")
    def _():
        import importlib
        found = {m: importlib.import_module(m).__version__.split("+")[0] for m in EXPECTED}
        wrong = {m: v for m, v in found.items() if v != EXPECTED[m]}
        assert not wrong, f"expected {EXPECTED}, found {found}"
        import wandb  # noqa: F401  (train.py imports it)
        return ", ".join(f"{m} {v}" for m, v in found.items())

    @check("MP3 codec (released codec_simu: sox/LAME encode, FFmpeg 4 decode)")
    def _():
        import torch
        import torchaudio
        assert torchaudio._extension._FFMPEG_INITIALIZED, "torchaudio found no FFmpeg 4 libraries"
        from look2hear.datas.musdb_moisesdb_datamodule import codec_simu
        x = torch.randn(2, 3 * 44100) * 0.1
        for kbps in (24, 128):
            y = codec_simu(x, sr=44100, options={"bitrate": kbps * 1000})
            assert tuple(y.shape) == tuple(x.shape), (kbps, tuple(y.shape))
        return "24 and 128 kbit/s round trips OK"

    @check(".env and config")
    def _():
        from omegaconf import OmegaConf
        from dotenv_loader import configure_attention_backend, load_env
        load_env(verbose=False)
        state["sdpa"] = configure_attention_backend(verbose=False)
        cfg = OmegaConf.load(args.config)
        OmegaConf.resolve(cfg)
        for key in ("train_dir", "eval_dir"):
            assert os.path.exists(cfg.datas[key]), f"datas.{key} missing: {cfg.datas[key]}"
        state["cfg"] = cfg
        mode = os.environ.get("WANDB_MODE", "online")
        assert mode in ("offline", "disabled") or cfg.logger.get("offline"), \
            "wandb would try to go online; set WANDB_MODE=offline in .env"
        return (f"{os.path.basename(args.config)}, data at {os.environ.get('APOLLO_DATA')}, "
                f"WANDB_MODE={mode}, APOLLO_SDPA_BACKEND={state['sdpa']}")

    @check("training data, eval sets and leakage")
    def _():
        import io
        from contextlib import redirect_stdout
        import h5py
        from data_prep.check_leakage import LOADER_STEMS, find_problems
        root = os.path.dirname(state["cfg"].datas.train_dir)
        with redirect_stdout(io.StringIO()):
            problems = find_problems(root)
        assert not problems, f"{len(problems)} problem(s), first: {problems[0]}"
        train_dir = state["cfg"].datas.train_dir
        n_files = sum(len([f for f in os.listdir(os.path.join(train_dir, s)) if f.endswith(".hdf5")])
                      for s in LOADER_STEMS)
        with h5py.File(state["cfg"].datas.eval_dir, "r") as f:
            n_val, val_shape = f["ori"].shape[0], f["ori"].shape[1:]
        test_h5 = os.path.join(root, "test", "test.hdf5")
        n_test = h5py.File(test_h5, "r")["ori"].shape[0] if os.path.exists(test_h5) else 0
        return f"{n_files} training HDF5 files, val {n_val} clips {val_shape}, test {n_test} clips, no leakage"

    @check("one real training sample (HDF5 read, stem mixing, MP3)")
    def _():
        import hydra
        dm = hydra.utils.instantiate(state["cfg"].datas)
        dm.setup("fit")
        ori, codec = dm.data_train[0]
        expected = (2, int(state["cfg"].datas.segments * state["cfg"].datas.sr))
        assert tuple(ori.shape) == expected and tuple(codec.shape) == expected, (ori.shape, codec.shape)
        workers, cpus = state["cfg"].datas.num_workers, len(os.sched_getaffinity(0))
        note = f"; WARNING: num_workers={workers} > {cpus} CPUs" if workers > cpus else ""
        return f"ori/codec {expected}, {cpus} CPUs for {workers} workers{note}"

    @check("GPU (one full training step: generator, discriminator, losses)")
    def _():
        import hydra
        import torch
        if not torch.cuda.is_available():
            return "no GPU visible (expected on a login node; run again inside the allocation)"
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        assert arch in torch.cuda.get_arch_list(), f"{arch} not in this torch build: {torch.cuda.get_arch_list()}"
        # torch 2.0's fused memory-efficient attention crashes on Hopper (illegal instruction).
        assert not (cap[0] >= 9 and torch.backends.cuda.mem_efficient_sdp_enabled()), \
            f"{arch} GPU: set APOLLO_SDPA_BACKEND=math in .env (see .env.example)"
        cfg = state["cfg"]
        model = hydra.utils.instantiate(cfg.model).cuda()
        disc = hydra.utils.instantiate(cfg.discriminator).cuda()
        loss_g, loss_d = hydra.utils.instantiate(cfg.loss_g), hydra.utils.instantiate(cfg.loss_d)
        # 0.5 s at batch 1 (about 4 GiB) fits any GPU; the loss path (complex STFT
        # magnitudes) also makes torch compile kernels at run time, which needs NVRTC.
        seconds = 0.5
        ori, codec = (torch.randn(1, 2, int(seconds * cfg.datas.sr), device="cuda") * 0.1 for _ in range(2))
        torch.cuda.reset_peak_memory_stats()
        output = model(codec)
        est, _ = disc(output.detach(), sample_rate=cfg.datas.sr)
        tgt, _ = disc(ori, sample_rate=cfg.datas.sr)
        loss_d(tgt, est).backward()
        est, est_fm = disc(output, sample_rate=cfg.datas.sr)
        _, tgt_fm = disc(ori, sample_rate=cfg.datas.sr)
        loss_g(est, est_fm, tgt_fm, output, ori).backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        seg = cfg.datas.segments * cfg.datas.batch_size
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        return (f"{torch.cuda.get_device_name(0)} ({arch}, {total:.0f} GiB): step on {seconds} s OK, "
                f"{peak:.1f} GiB peak; the config's {seg:g} s per step needs about {peak * seg / seconds:.0f} GiB")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
