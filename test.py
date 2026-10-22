###
# Author: Kai Li
# Date: 2024-01-22 01:16:22
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2024-01-24 00:05:10
###
import csv
import h5py
import numpy as np
import torch
import torchaudio
import json
from typing import Any, Dict, List, Optional, Tuple
import os
from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import look2hear.system
import look2hear.datas
import look2hear.losses
import look2hear.models
from look2hear.metrics import MetricsTracker
from look2hear.utils import RankedLogger, instantiate, print_only
from dotenv_loader import configure_attention_backend, load_env
import warnings
warnings.filterwarnings("ignore")

class PackedTestSet():
    """Test clips packed by data_prep/make_eval_set.py: datasets `ori`, `codec`, `clip_id`.

    With max_per_bitrate, only the first N clips of each dataset/bitrate are used; these
    are exactly the clips a smaller set made with the same seed would contain."""
    def __init__(self, path, manifest, max_per_bitrate=0):
        self.h5 = h5py.File(path, "r")
        clip_ids = [c.decode() if isinstance(c, bytes) else str(c) for c in self.h5["clip_id"][:]]
        self.indices = list(range(len(clip_ids)))
        if max_per_bitrate:
            with open(manifest, newline="") as f:
                rows = list(csv.DictReader(f))
            seen, self.indices = {}, []
            for r in rows:
                key = (r["dataset"], r["bitrate_kbps"])
                seen[key] = seen.get(key, 0) + 1
                if seen[key] <= max_per_bitrate:
                    self.indices.append(int(r["index"]))
        self.clip_ids = [clip_ids[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return torch.from_numpy(self.h5["ori"][i]), torch.from_numpy(self.h5["codec"][i]), self.clip_ids[idx]

def breakdown(metrics_csv, manifest_csv, out_csv):
    """Mean metrics per dataset, bitrate and category (paper Fig. 2 / Table II), over all
    clips and, as "aligned:<group>", over clips whose MP3 input was aligned with the target."""
    with open(manifest_csv, newline="") as f:
        meta = {r["clip_id"]: r for r in csv.DictReader(f)}
    with open(metrics_csv, newline="") as f:
        scores = [r for r in csv.DictReader(f) if r["snt_id"] in meta]
    groups = {}
    for r in scores:
        m = meta[r["snt_id"]]
        for key in [("all", "all"), ("dataset", m["dataset"]), ("bitrate_kbps", m["bitrate_kbps"]),
                    ("category", m["category"]), ("dataset+bitrate", f"{m['dataset']}_{m['bitrate_kbps']}k"),
                    ("dataset+category", f"{m['dataset']}_{m['category']}")]:
            groups.setdefault(key, []).append(r)
            if m.get("aligned") == "1":
                groups.setdefault(("aligned:" + key[0], key[1]), []).append(r)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "value", "n", "sdr", "si-snr", "visqol"])
        for (group, value), rs in sorted(groups.items()):
            w.writerow([group, value, len(rs)] + [round(sum(float(r[k]) for r in rs) / len(rs), 4)
                                                  for k in ("sdr", "si-snr", "visqol")])
    print(f"Breakdown by dataset/bitrate/category written to {out_csv}")

def test(cfg: DictConfig, test_h5: str, manifest: str, max_per_bitrate: int = 0) -> None:
    data_val = PackedTestSet(test_h5, manifest, max_per_bitrate)
    
    cfg.model.pop("_target_", None)
    model = look2hear.models.Apollo.from_pretrain(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), **cfg.model).cuda()
    model.eval()
    results_dir = os.path.join(cfg.exp.dir, cfg.exp.name, "results/")
    os.makedirs(results_dir, exist_ok=True)
    metrics = MetricsTracker(save_file=results_dir+"metrics.csv")
    length = len(data_val)
    # All estimates go into one file rather than one WAV per clip (file-count limit on DRAC).
    with h5py.File(results_dir+"test_estimates.hdf5", "w") as est_file:
        clip_shape = data_val.h5["ori"].shape[1:]
        ests_ds = est_file.create_dataset("ests", (len(data_val),) + clip_shape, "float32", chunks=(1,) + clip_shape)
        est_file.create_dataset("clip_id", data=np.array(data_val.clip_ids, dtype=h5py.string_dtype()))
        for idx in range(len(data_val)):
            ori_wav, codec_wav, key = data_val[idx]
            
            ori_wav = ori_wav.cuda()
            codec_wav = codec_wav.unsqueeze(0).cuda()
            with torch.no_grad():
                ests = model(codec_wav)
                ests_ds[idx] = ests.squeeze(0).cpu().numpy()
                metrics(ori_wav, ests, key)
                
            if idx % 10 == 0:
                dicts = metrics.update()
                print(f"Processed {idx}/{length} samples: SDR: {dicts['sdr']}, SI-SNR: {dicts['si-snr']}, VISQOL: {dicts['visqol']}")
    
    metrics.final()       
    breakdown(results_dir+"metrics.csv", manifest, results_dir+"metrics_breakdown.csv")
    print("Finished testing")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="Exps/Apollo/config.yaml",
        help="Full path to save best validation model",
    )
    parser.add_argument("--test_h5", required=True, help="packed test set, e.g. .../apollo_v1/test/test.hdf5")
    parser.add_argument("--manifest", help="test manifest CSV (default: test_manifest.csv next to --test_h5)")
    parser.add_argument("--max-per-bitrate", type=int, default=0,
                        help="quick run: only the first N clips per dataset and bitrate (0 = all)")
    
    args = parser.parse_args()
    # Same machine settings as train.py (APOLLO_SDPA_BACKEND must match the training run).
    load_env()
    configure_attention_backend()
    cfg = OmegaConf.load(args.conf_dir)
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    test(cfg, args.test_h5, args.manifest or os.path.join(os.path.dirname(args.test_h5), "test_manifest.csv"),
         args.max_per_bitrate)
    