"""Check that no held-out song reached training, and that eval sets use only their split.

    python -m data_prep.check_leakage --root <out>/apollo_v1

Training files are checked from their own names (<dataset>__<track>__<file>.hdf5) as
well as from manifest.csv, so a stray file that the manifest does not know about is
caught too. Exits with status 1 on any violation.
"""
import argparse
import csv
import os
import sys

from data_prep.splits import DATASETS, SPLITS_DIR, load_split_file

# Stem folders MusdbMoisesdbDataset samples from; an empty or missing one crashes training.
LOADER_STEMS = ("bass", "bowed_strings", "drums", "guitar", "other", "other_keys", "other_plucked",
                "percussion", "piano", "vocals", "wind", "mixture")


def find_problems(root, splits_dir=SPLITS_DIR):
    """Print what is present and return a list of violations (empty = no leakage)."""
    splits = {d: load_split_file(d, splits_dir) for d in DATASETS}
    problems = []

    def split_of(dataset, track_id):
        return splits.get(dataset, {}).get(track_id)

    train_dir = os.path.join(root, "hdf5_train")
    train_tracks = {d: set() for d in DATASETS}
    if os.path.isdir(train_dir):
        on_disk = set()
        for dp, _, fs in os.walk(train_dir):
            for f in fs:
                if f == "manifest.csv":
                    continue
                rel = os.path.relpath(os.path.join(dp, f), train_dir)
                on_disk.add(rel)
                parts = f[:-len(".hdf5")].split("__") if f.endswith(".hdf5") else []
                if len(parts) != 3 or parts[0] not in DATASETS:
                    problems.append(f"hdf5_train: unexpected file {rel}")
                    continue
                split = split_of(parts[0], parts[1])
                if split != "train":
                    problems.append(f"hdf5_train: {rel} comes from a {split or 'unlisted'} track")
                train_tracks[parts[0]].add(parts[1])
        with open(os.path.join(train_dir, "manifest.csv"), newline="") as f:
            listed = {r["hdf5_file"] for r in csv.DictReader(f) if r["hdf5_file"]}
        for rel in sorted(on_disk ^ listed):
            problems.append(f"hdf5_train: {rel} is {'not in manifest' if rel in on_disk else 'missing on disk'}")
        for stem in LOADER_STEMS:
            d = os.path.join(train_dir, stem)
            if not os.path.isdir(d) or not any(n.endswith(".hdf5") for n in os.listdir(d)):
                problems.append(f"hdf5_train: stem folder {stem}/ is missing or empty (the loader samples it)")
        for d in DATASETS:
            n_listed = sum(s == "train" for s in splits[d].values())
            print(f"hdf5_train: {d}: {len(train_tracks[d])} of {n_listed} train tracks present")
    else:
        print("hdf5_train: not built yet")

    eval_songs = {}
    for split, sub in (("val", "eval"), ("test", "test")):
        manifest = os.path.join(root, sub, f"{split}_manifest.csv")
        if not os.path.exists(manifest):
            print(f"{split}: not built yet")
            continue
        with open(manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            s = split_of(r["dataset"], r["track_id"])
            if s != split:
                problems.append(f"{split}: clip {r['clip_id']} uses {r['track_id']!r} from the {s or 'unlisted'} split")
        eval_songs[split] = {(r["dataset"], r["track_id"]) for r in rows}
        print(f"{split}: {len(rows)} clips from {len(eval_songs[split])} songs")

    for split, songs in eval_songs.items():
        for d, t in sorted(songs):
            if t in train_tracks[d]:
                problems.append(f"{split} song {d}/{t} also appears in hdf5_train")
    if "val" in eval_songs and "test" in eval_songs:
        for d, t in sorted(eval_songs["val"] & eval_songs["test"]):
            problems.append(f"song {d}/{t} is in both val and test")
    musdb_test = {t for t, s in splits["musdb18hq"].items() if s == "test"}
    print(f"MUSDB18-HQ official test tracks in training data: {len(musdb_test & train_tracks['musdb18hq'])} of {len(musdb_test)}")

    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--splits-dir", default=SPLITS_DIR, help="folder holding <dataset>.csv split files")
    args = p.parse_args(argv)
    problems = find_problems(args.root, args.splits_dir)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for msg in problems:
            print("  " + msg)
        sys.exit(1)
    print("\nOK: no leakage found")


if __name__ == "__main__":
    main()
