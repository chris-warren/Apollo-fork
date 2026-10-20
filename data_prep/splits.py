"""Train/val/test splits for MUSDB18-HQ and MoisesDB, checked against what is on disk.

The split files live in data_prep/splits/ (sources in data_prep/splits/README.md).
Every track found on disk must appear in exactly one split, and every listed track
must be on disk; anything else raises SplitError so a partial or renamed dataset
can never silently change what ends up in training.
"""
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

SPLITS = ("train", "val", "test")
DATASETS = ("musdb18hq", "moisesdb")
SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
MUSDB_STEMS = ("mixture", "bass", "drums", "other", "vocals")


class SplitError(ValueError):
    pass


@dataclass
class Track:
    dataset: str
    track_id: str  # MUSDB folder name, or MoisesDB song UUID
    name: str  # human-readable "Artist - Title"
    split: str
    path: str
    # stem name -> source WAV files; a MoisesDB stem can have several (e.g. kick, snare)
    stems: Dict[str, List[str]] = field(default_factory=dict)


def load_split_file(dataset: str, splits_dir: str = SPLITS_DIR) -> Dict[str, str]:
    """Return {track_id: split} from data_prep/splits/<dataset>.csv."""
    if dataset not in DATASETS:
        raise SplitError(f"unknown dataset {dataset!r}")
    key = "track" if dataset == "musdb18hq" else "song_id"
    out = {}
    with open(os.path.join(splits_dir, f"{dataset}.csv"), newline="") as f:
        for row in csv.DictReader(f):
            tid, split = row[key], row["split"]
            if split not in SPLITS:
                raise SplitError(f"{dataset}: {tid!r} has unknown split {split!r}")
            if tid in out:
                raise SplitError(f"{dataset}: {tid!r} is listed more than once")
            out[tid] = split
    return out


def _musdb_tracks(root: str, splits: Dict[str, str]) -> List[Track]:
    tracks = []
    for folder in ("train", "test"):
        base = os.path.join(root, folder)
        if not os.path.isdir(base):
            raise SplitError(f"musdb18hq: missing folder {base}")
        for tid in sorted(os.listdir(base)):
            path = os.path.join(base, tid)
            if not os.path.isdir(path):
                continue
            split = splits.get(tid)
            if split is None:
                raise SplitError(f"musdb18hq: track on disk but not in any split: {tid!r}")
            # The official test folder must map to our test split and vice versa.
            if (folder == "test") != (split == "test"):
                raise SplitError(f"musdb18hq: {tid!r} is in {folder}/ but split file says {split}")
            stems = {}
            for s in MUSDB_STEMS:
                p = os.path.join(path, f"{s}.wav")
                if not os.path.isfile(p):
                    raise SplitError(f"musdb18hq: {tid!r} is missing {s}.wav")
                stems[s] = [p]
            tracks.append(Track("musdb18hq", tid, tid, split, path, stems))
    return tracks


def _moisesdb_tracks(root: str, splits: Dict[str, str]) -> List[Track]:
    tracks = []
    for tid in sorted(os.listdir(root)):
        path = os.path.join(root, tid)
        if not os.path.isdir(path):
            continue
        split = splits.get(tid)
        if split is None:
            raise SplitError(f"moisesdb: song on disk but not in any split: {tid!r}")
        with open(os.path.join(path, "data.json")) as f:
            meta = json.load(f)
        stems = {}
        for stem in meta["stems"]:
            d = os.path.join(path, stem["stemName"])
            files = [os.path.join(d, f"{t['id']}.{t.get('extension', 'wav')}") for t in stem["tracks"]]
            missing = [p for p in files if not os.path.isfile(p)]
            if missing:
                raise SplitError(f"moisesdb: {tid!r} is missing {missing}")
            # 23 songs list the same stemName more than once; their files belong together.
            stems.setdefault(stem["stemName"], []).extend(files)
        stems = {name: sorted(files) for name, files in stems.items() if files}
        name = f"{meta.get('artist', '?')} - {meta.get('song', '?')}"
        tracks.append(Track("moisesdb", tid, name, split, path, stems))
    return tracks


def load_tracks(dataset: str, root: str, splits_dir: str = SPLITS_DIR) -> List[Track]:
    """List every track of `dataset` under `root`, verified against its split file.

    `root` is the MUSDB18-HQ folder holding train/ and test/, or the MoisesDB folder
    holding one folder per song UUID (moisesdb_v0.1/).
    """
    splits = load_split_file(dataset, splits_dir)
    if dataset == "musdb18hq":
        tracks = _musdb_tracks(root, splits)
    else:
        tracks = _moisesdb_tracks(root, splits)
    missing = sorted(set(splits) - {t.track_id for t in tracks})
    if missing:
        raise SplitError(f"{dataset}: {len(missing)} listed tracks not found under {root}: {missing[:5]}")
    return tracks
