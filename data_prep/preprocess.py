"""Build the training HDF5 store from the *train split only*.

    python -m data_prep.preprocess --root <out>/apollo_v1 \
        --musdb-root <musdb18hq> --moises-root <moisesdb/moisesdb_v0.1>

Writes <root>/hdf5_train/<stem>/<dataset>__<track>__<file>.hdf5 (one file per source
WAV, dataset "data" of shape (n_windows, 2, 6 * 44100)) and <root>/hdf5_train/manifest.csv,
the layout MusdbMoisesdbDataset reads. Differences from Apollo-data-preprocess are
listed in data_prep/README.md.
"""
import argparse
import csv
import os
from multiprocessing import Pool

import h5py
import numpy as np
import soundfile as sf

from data_prep.runlog import record_run
from data_prep.splits import DATASETS, SPLITS_DIR, load_tracks


# VAD() is copied unchanged from Apollo-data-preprocess (commit bb37a7b,
# musdb_preprocess.py / moisesdb_preprocess.py), Kai Li, CC BY-NC-SA 4.0:
# https://github.com/JusperLee/Apollo-data-preprocess
def VAD(x, sr=44100, win=441, session_length=6):
    # x shape: (T, 2)
    # VAD
    x_len = x.shape[0] // sr * sr
    x = x[:x_len]
    x_p = x.reshape(-1, win, 2)
    x_pow = np.sum(np.power(x_p, 2), (1, 2)).reshape(-1,)
    # skip all silent segments when calculating threshold
    x_valid_pow = x_pow[x_pow > 1e-3]
    if len(x_valid_pow) > 0:
        threshold = np.quantile(x_valid_pow, 0.15)
        valid_segment = []

        num_session = (x_len - sr * session_length) // (sr * session_length // 2)

        for s in range(num_session):
            this_segment = x[s * sr * session_length // 2:s * sr * session_length // 2 + sr * session_length,:]
            this_segment_pow = this_segment.reshape(-1, win, 2)
            this_segment_pow = np.sum(np.power(this_segment_pow, 2), (1,2)).reshape(-1,)
            pow_ratio = np.sum(this_segment_pow > threshold) / len(this_segment_pow)
            if pow_ratio > 0.5:
                valid_segment.append(this_segment)

        return valid_segment
    else:
        return []


SR = 44100
SESSION_LENGTH = 6
MANIFEST_FIELDS = ["dataset", "track_id", "track_name", "stem", "source_file", "hdf5_file", "n_windows"]


def output_name(dataset: str, track_id: str, source: str) -> str:
    return f"{dataset}__{track_id}__{os.path.splitext(os.path.basename(source))[0]}.hdf5"


def process_file(job):
    """VAD one source WAV and write its windows; returns a manifest row."""
    dataset, track_id, track_name, stem, source, out_dir, dtype = job
    rel = os.path.join(stem, output_name(dataset, track_id, source))
    final = os.path.join(out_dir, rel)
    row = dict(dataset=dataset, track_id=track_id, track_name=track_name, stem=stem,
               source_file=source, hdf5_file=rel, n_windows=0)
    if os.path.exists(final):  # resume: finished files are never rewritten
        with h5py.File(final, "r") as f:
            row["n_windows"] = f["data"].shape[0]
        return row
    x, sr = sf.read(source, always_2d=True)  # float64, as upstream
    if sr != SR or x.shape[1] != 2:
        raise ValueError(f"{source}: expected {SR} Hz stereo, got {sr} Hz x {x.shape[1]}")
    windows = VAD(x, sr=sr, session_length=SESSION_LENGTH)
    if not windows:  # upstream wrote an empty HDF5 here, which breaks training
        row["hdf5_file"] = ""
        return row
    data = np.stack(windows).transpose(0, 2, 1).astype(dtype)  # (n, 2, T), as upstream's x[s].T
    tmp = final + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("data", data=data, chunks=(1,) + data.shape[1:])
    os.replace(tmp, final)
    row["n_windows"] = data.shape[0]
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="output root, e.g. .../processed/apollo_v1")
    p.add_argument("--musdb-root", help="folder holding MUSDB18-HQ train/ and test/")
    p.add_argument("--moises-root", help="MoisesDB folder holding one folder per song (moisesdb_v0.1)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--splits-dir", default=SPLITS_DIR, help="folder holding <dataset>.csv split files")
    p.add_argument("--limit-tracks", type=int, default=0, help="debug: only the first N train tracks per dataset")
    args = p.parse_args(argv)

    roots = {"musdb18hq": args.musdb_root, "moisesdb": args.moises_root}
    if not any(roots.values()):
        p.error("give --musdb-root and/or --moises-root")
    out_dir = os.path.join(args.root, "hdf5_train")
    os.makedirs(out_dir, exist_ok=True)

    jobs, held_out = [], {}
    for dataset in DATASETS:
        if not roots[dataset]:
            continue
        tracks = load_tracks(dataset, roots[dataset], args.splits_dir)  # raises on any split mismatch
        train = [t for t in tracks if t.split == "train"]
        held_out[dataset] = len(tracks) - len(train)
        if args.limit_tracks:
            train = train[:args.limit_tracks]
        for t in train:
            for stem, files in t.stems.items():
                os.makedirs(os.path.join(out_dir, stem), exist_ok=True)
                jobs += [(dataset, t.track_id, t.name, stem, f, out_dir, args.dtype) for f in files]

    for dp, _, fs in os.walk(out_dir):  # leftovers from an interrupted run
        for f in fs:
            if f.endswith(".tmp"):
                os.remove(os.path.join(dp, f))

    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(process_file, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} files", flush=True)

    rows.sort(key=lambda r: (r["dataset"], r["track_id"], r["stem"], r["source_file"]))
    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(manifest + ".tmp", manifest)

    written = [r for r in rows if r["hdf5_file"]]
    details = {"dtype": args.dtype, "held-out tracks skipped": held_out}
    for dataset in DATASETS:
        rs = [r for r in written if r["dataset"] == dataset]
        if rs or roots[dataset]:
            details[f"{dataset}"] = (f"{len({r['track_id'] for r in rs})} train tracks, {len(rs)} HDF5 files, "
                                     f"{sum(r['n_windows'] for r in rs)} windows")
    details["source files with no non-silent window (not written)"] = len(rows) - len(written)
    if args.limit_tracks:
        details["limit-tracks"] = args.limit_tracks
    record_run(args.root, "preprocess (hdf5_train)", details)
    for k, v in details.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
