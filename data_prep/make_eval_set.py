"""Build a fixed, packed validation or test set from held-out songs only.

    python -m data_prep.make_eval_set --root <out>/apollo_v1 --split test \
        --musdb-root <musdb18hq> --moises-root <moisesdb/moisesdb_v0.1> --per-bitrate 2500

Each clip is 6 s from ONE held-out song: a subset of its stems, time-aligned at their
original levels, so the mixture is real music. Clips are balanced over the four
categories of the paper's Table II and over the six bitrates, per dataset. The MP3
step is the released training code's codec_simu, unchanged, and both signals are
peak-normalised as in training. codec_simu re-aligns the decoded MP3 with match2(),
which leaves some clips out of step with the target; each clip's remaining offset is
measured and stored in the manifest (`residual_lag_samples`, `aligned`) so results can
be reported on all clips and on aligned clips only.

Output: <root>/eval/val.hdf5 or <root>/test/test.hdf5 (float32 datasets `ori` and
`codec` of shape (N, 2, 264600), plus `clip_id`) and a <split>_manifest.csv. Each
dataset/bitrate has its own seeded random stream, so the first N clips per bitrate of
a larger set are exactly the clips of a smaller one (see test.py --max-per-bitrate).
"""
import argparse
import csv
import os
import random
from multiprocessing import Pool

import h5py
import numpy as np
import soundfile as sf

from data_prep.preprocess import SESSION_LENGTH, SR
from data_prep.runlog import record_run
from data_prep.splits import DATASETS, SPLITS_DIR, load_tracks

BITRATES = (24, 32, 48, 64, 96, 128)  # kbit/s, as in the paper
CATEGORIES = ("vocals", "single_stem", "multi_stem", "multi_stem_vocals")  # Table II
CLIP = SR * SESSION_LENGTH
HOP = CLIP // 2
DEFAULT_SEEDS = {"val": 20251019, "test": 20251020}
MANIFEST_FIELDS = ["clip_id", "index", "dataset", "track_id", "track_name", "category", "stems",
                   "start_sample", "bitrate_kbps", "residual_lag_samples", "aligned", "seed"]
MAX_LAG = 4000  # samples; the MP3 delay is 1,105


def activity(x, sr=SR, win=441, session_length=SESSION_LENGTH):
    """Which 6 s windows (3 s hop) VAD() would keep, as a boolean array.

    Same threshold and >50% rule as data_prep.preprocess.VAD (tested to agree); for a
    silent stem it returns all False, like VAD returning no windows.
    """
    x_len = x.shape[0] // sr * sr
    x = x[:x_len]
    pow_ = np.sum(np.power(x.reshape(-1, win, x.shape[1]), 2), (1, 2))
    valid = pow_[pow_ > 1e-3]
    num_session = max(0, (x_len - sr * session_length) // (sr * session_length // 2))
    if len(valid) == 0:
        return np.zeros(num_session, bool)
    threshold = np.quantile(valid, 0.15)
    frames = sr * session_length // win
    hop = frames // 2
    above = pow_ > threshold
    return np.array([above[s * hop:s * hop + frames].mean() > 0.5 for s in range(num_session)], bool)


def load_stem(files, length=None, start=0):
    """Sum a stem's source files (MoisesDB stems can have several); zero-pad to `length`."""
    parts = []
    for f in files:
        frames = -1 if length is None else length
        x, sr = sf.read(f, start=start, frames=frames, dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{f}: expected {SR} Hz, got {sr}")
        parts.append(x)
    n = length if length is not None else max(len(p) for p in parts)
    out = np.zeros((n, 2), np.float32)
    for p in parts:
        out[:len(p)] += p
    return out


def song_activity(track):
    """{stem: bool array over the song's window grid}; stems padded to the longest one."""
    n = max(sf.info(f).frames for files in track.stems.values() for f in files)
    return {stem: activity(load_stem(files, n)) for stem, files in track.stems.items() if stem != "mixture"}


def pick_stems(rng, category, stems):
    """Choose a stem subset for a category, or None if the song cannot provide one."""
    others = sorted(s for s in stems if s != "vocals")
    has_vocals = "vocals" in stems
    if category == "vocals":
        return ["vocals"] if has_vocals else None
    if category == "single_stem":
        return [rng.choice(others)] if others else None
    if category == "multi_stem":
        return sorted(rng.sample(others, rng.randint(2, len(others)))) if len(others) >= 2 else None
    if category == "multi_stem_vocals":
        return ["vocals"] + sorted(rng.sample(others, rng.randint(1, len(others)))) if has_vocals and others else None
    raise ValueError(category)


def plan_clips(tracks, masks, dataset, n_per_bitrate, seed, max_tries=1000):
    """Deterministically choose (song, stems, window) for every clip of one dataset."""
    plan = []
    for b in BITRATES:
        rng = random.Random(f"{seed}-{dataset}-{b}")
        for i in range(n_per_bitrate):
            category = CATEGORIES[i % len(CATEGORIES)]
            for _ in range(max_tries):
                t = rng.choice(tracks)
                stems = pick_stems(rng, category, masks[t.track_id])
                if stems is None:
                    continue
                common = np.logical_and.reduce([masks[t.track_id][s] for s in stems])
                if common.any():
                    start = int(rng.choice(np.flatnonzero(common))) * HOP
                    plan.append(dict(dataset=dataset, track=t, category=category, stems=stems,
                                     start_sample=start, bitrate_kbps=b))
                    break
            else:
                raise RuntimeError(f"{dataset}: no song has an active window for {category}")
    return plan


def residual_lag(ori, codec, max_lag=MAX_LAG):
    """Offset (samples) by which `codec` trails `ori`: peak of the plain cross-correlation
    of the channel sums within +-max_lag. 0 means aligned."""
    a, b = ori.sum(0), codec.sum(0)
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)
    k = int(np.argmax(np.r_[c[:max_lag + 1], c[-max_lag:]]))
    return k if k <= max_lag else k - (2 * max_lag + 1)


def render(item):
    from look2hear.datas.musdb_moisesdb_datamodule import codec_simu
    import torch

    t = item["track"]
    ori = sum(load_stem(t.stems[s], CLIP, item["start_sample"]) for s in item["stems"])
    ori = torch.from_numpy(np.ascontiguousarray(ori.T))
    codec = codec_simu(ori, sr=SR, options={"bitrate": item["bitrate_kbps"] * 1000})
    max_scale = max(ori.abs().max(), codec.abs().max())  # as MusdbMoisesdbDataset
    if max_scale > 0:
        ori, codec = ori / max_scale, codec / max_scale
    ori, codec = ori.numpy().astype(np.float32), codec.numpy().astype(np.float32)
    return ori, codec, residual_lag(ori, codec)


def _init_worker():
    import torch
    torch.set_num_threads(1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="output root, e.g. .../processed/apollo_v1")
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--musdb-root")
    p.add_argument("--moises-root")
    p.add_argument("--per-bitrate", type=int, required=True, help="clips per bitrate per dataset")
    p.add_argument("--seed", type=int, help=f"default: {DEFAULT_SEEDS}")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--splits-dir", default=SPLITS_DIR, help="folder holding <dataset>.csv split files")
    p.add_argument("--export-wav", type=int, default=0, metavar="K",
                   help="also write K clips per dataset/bitrate/category as WAVs, for listening")
    p.add_argument("--export-dir", help="where --export-wav writes (keep it off the server)")
    args = p.parse_args(argv)
    if args.export_wav and not args.export_dir:
        p.error("--export-wav needs --export-dir")
    seed = DEFAULT_SEEDS[args.split] if args.seed is None else args.seed

    roots = {"musdb18hq": args.musdb_root, "moisesdb": args.moises_root}
    plan = []
    for dataset in DATASETS:
        if not roots[dataset]:
            continue
        tracks = [t for t in load_tracks(dataset, roots[dataset], args.splits_dir) if t.split == args.split]
        print(f"{dataset}: measuring stem activity in {len(tracks)} {args.split} songs", flush=True)
        with Pool(args.workers) as pool:
            masks = dict(zip([t.track_id for t in tracks], pool.map(song_activity, tracks)))
        plan += plan_clips(tracks, masks, dataset, args.per_bitrate, seed)

    subdir = "eval" if args.split == "val" else "test"
    out_dir = os.path.join(args.root, subdir)
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"{args.split}.hdf5")
    tmp = final + ".tmp"
    rows = []
    with h5py.File(tmp, "w") as f, Pool(args.workers, initializer=_init_worker) as pool:
        ori_ds = f.create_dataset("ori", (len(plan), 2, CLIP), "float32", chunks=(1, 2, CLIP))
        codec_ds = f.create_dataset("codec", (len(plan), 2, CLIP), "float32", chunks=(1, 2, CLIP))
        ids = []
        for i, (item, (ori, codec, lag)) in enumerate(zip(plan, pool.imap(render, plan, chunksize=4))):
            ori_ds[i], codec_ds[i] = ori, codec
            t = item["track"]
            clip_id = f"{item['dataset']}_{item['bitrate_kbps']}k_{i:05d}"
            ids.append(clip_id)
            rows.append(dict(clip_id=clip_id, index=i, dataset=item["dataset"], track_id=t.track_id,
                             track_name=t.name, category=item["category"], stems="+".join(item["stems"]),
                             start_sample=item["start_sample"], bitrate_kbps=item["bitrate_kbps"],
                             residual_lag_samples=lag, aligned=int(lag == 0), seed=seed))
            if args.export_wav and sum(r["dataset"] == item["dataset"] and r["bitrate_kbps"] == item["bitrate_kbps"]
                                       and r["category"] == item["category"] for r in rows) <= args.export_wav:
                d = os.path.join(args.export_dir, args.split, clip_id)
                os.makedirs(d, exist_ok=True)
                sf.write(os.path.join(d, "ori_wav.wav"), ori.T, SR, subtype="FLOAT")
                sf.write(os.path.join(d, "codec_wav.wav"), codec.T, SR, subtype="FLOAT")
            if (i + 1) % 100 == 0 or i + 1 == len(plan):
                print(f"{i + 1}/{len(plan)} clips", flush=True)
        f.create_dataset("clip_id", data=np.array(ids, dtype=h5py.string_dtype()))
        f.attrs["split"], f.attrs["seed"], f.attrs["sample_rate"] = args.split, seed, SR
    os.replace(tmp, final)

    manifest = os.path.join(out_dir, f"{args.split}_manifest.csv")
    with open(manifest + ".tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(manifest + ".tmp", manifest)

    details = {"split": args.split, "seed": seed, "clips per bitrate per dataset": args.per_bitrate,
               "total clips": len(rows), "file": os.path.relpath(final, args.root)}
    for dataset in DATASETS:
        rs = [r for r in rows if r["dataset"] == dataset]
        if rs:
            cats = {c: sum(r["category"] == c for r in rs) for c in CATEGORIES}
            details[dataset] = f"{len(rs)} clips from {len({r['track_id'] for r in rs})} songs, by category {cats}"
            details[f"{dataset} aligned clips"] = {b: f"{sum(r['aligned'] for r in rs if r['bitrate_kbps'] == b)}/"
                                                      f"{sum(r['bitrate_kbps'] == b for r in rs)}" for b in BITRATES}
    record_run(args.root, f"make_eval_set ({args.split})", details)
    for k, v in details.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
