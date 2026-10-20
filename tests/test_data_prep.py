import ast
import csv
import hashlib
import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import soundfile as sf

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:  # the released codec needs torchaudio 2.0.x with FFmpeg 4 libraries (look2hear.yml)
    import torch
    import torchaudio
    from look2hear.datas.musdb_moisesdb_datamodule import codec_simu
    if not torchaudio._extension._FFMPEG_INITIALIZED:
        codec_simu = None
except (ImportError, AttributeError):  # pragma: no cover
    codec_simu = None

from data_prep import check_leakage, make_eval_set, preprocess
from data_prep.splits import SplitError, load_split_file, load_tracks

SR = 44100


def tone(seconds, freq=440.0, amp=0.3, seed=0):
    """Stereo noise-modulated tone, loud enough for VAD to keep."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * rng.random(t.size))
    return np.stack([x, 0.9 * x], 1).astype(np.float32)


def write_wav(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, SR, subtype="PCM_16")


class Fixture:
    """Tiny MUSDB-like and MoisesDB-like datasets plus matching split files."""

    def __init__(self, base, seconds=16):
        self.musdb = os.path.join(base, "musdb18hq")
        self.moises = os.path.join(base, "moisesdb_v0.1")
        self.splits = os.path.join(base, "splits")
        os.makedirs(self.splits)
        musdb = {"Train A": ("train", "train"), "Train B": ("train", "train"),
                 "Val A": ("train", "val"), "Test A": ("test", "test")}
        for i, (name, (folder, _)) in enumerate(musdb.items()):
            for j, stem in enumerate(("mixture", "bass", "drums", "other", "vocals")):
                x = tone(seconds, 110 * (j + 1), seed=10 * i + j)
                if name == "Train B" and stem == "bass":
                    x = np.zeros_like(x)  # a silent stem must not produce an HDF5 file
                write_wav(os.path.join(self.musdb, folder, name, f"{stem}.wav"), x)
        with open(os.path.join(self.splits, "musdb18hq.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track", "split"])
            w.writerows((n, s) for n, (_, s) in musdb.items())
        moises = {"song-train": "train", "song-val": "val", "song-test": "test"}
        for i, (sid, _) in enumerate(moises.items()):
            stems = []
            for j, stem in enumerate(("vocals", "guitar", "drums")):
                tracks = []
                for k in range(2 if stem == "drums" else 1):
                    tid = f"{sid}-{stem}-{k}"
                    n = seconds - (1 if k else 0)  # unequal stem lengths, as in MoisesDB
                    write_wav(os.path.join(self.moises, sid, stem, f"{tid}.wav"), tone(n, 150 * (j + 1) + k, seed=i))
                    tracks.append({"id": tid, "extension": "wav", "trackType": stem})
                if stem == "drums":  # MoisesDB sometimes lists one stem name twice
                    stems += [{"stemName": stem, "tracks": tracks[:1]}, {"stemName": stem, "tracks": tracks[1:]}]
                else:
                    stems.append({"stemName": stem, "tracks": tracks})
            with open(os.path.join(self.moises, sid, "data.json"), "w") as f:
                json.dump({"artist": "Artist", "song": sid, "stems": stems}, f)
        with open(os.path.join(self.splits, "moisesdb.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["song_id", "split"])
            w.writerows(moises.items())


class SplitTests(unittest.TestCase):
    def test_shipped_split_files_are_complete(self):
        musdb = load_split_file("musdb18hq")
        moises = load_split_file("moisesdb")
        count = lambda d, s: sum(v == s for v in d.values())
        self.assertEqual([count(musdb, s) for s in ("train", "val", "test")], [86, 14, 50])
        self.assertEqual([count(moises, s) for s in ("train", "val", "test")], [144, 48, 48])

    def test_rejects_unlisted_missing_and_misplaced_tracks(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d, seconds=1)
            self.assertEqual(len(load_tracks("musdb18hq", fx.musdb, fx.splits)), 4)
            os.rename(os.path.join(fx.musdb, "train", "Train A"), os.path.join(fx.musdb, "train", "Renamed"))
            with self.assertRaisesRegex(SplitError, "not in any split"):
                load_tracks("musdb18hq", fx.musdb, fx.splits)
            os.rename(os.path.join(fx.musdb, "train", "Renamed"), os.path.join(fx.musdb, "test", "Train A"))
            with self.assertRaisesRegex(SplitError, "is in test/"):
                load_tracks("musdb18hq", fx.musdb, fx.splits)
            os.remove(os.path.join(fx.moises, "song-val", "data.json"))
            os.rename(os.path.join(fx.moises, "song-val"), os.path.join(d, "elsewhere"))
            with self.assertRaisesRegex(SplitError, "not found"):
                load_tracks("moisesdb", fx.moises, fx.splits)

    def test_repeated_moisesdb_stem_names_keep_all_files(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d, seconds=1)
            for t in load_tracks("moisesdb", fx.moises, fx.splits):
                self.assertEqual(len(t.stems["drums"]), 2, t.track_id)

    def test_rejects_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "moisesdb.csv"), "w") as f:
                f.write("song_id,split\nx,train\nx,test\n")
            with self.assertRaisesRegex(SplitError, "more than once"):
                load_split_file("moisesdb", d)


class VadTests(unittest.TestCase):
    def test_activity_matches_upstream_vad(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.1, (40 * SR, 2))
        x[5 * SR:14 * SR] *= 1e-3  # quiet stretch
        x[25 * SR:27 * SR] = 0
        kept = preprocess.VAD(x)
        mask = make_eval_set.activity(x)
        starts = [s * 3 * SR for s in np.flatnonzero(mask)]
        self.assertEqual(len(kept), mask.sum())
        for seg, start in zip(kept, starts):
            np.testing.assert_array_equal(seg, x[start:start + 6 * SR])
        self.assertTrue(0 < mask.sum() < mask.size)

    def test_silent_input(self):
        self.assertEqual(preprocess.VAD(np.zeros((10 * SR, 2))), [])
        self.assertFalse(make_eval_set.activity(np.zeros((10 * SR, 2))).any())


class PickStemsTests(unittest.TestCase):
    def test_categories(self):
        rng = random.Random(0)
        stems = ["bass", "drums", "other", "vocals"]
        for _ in range(50):
            self.assertEqual(make_eval_set.pick_stems(rng, "vocals", stems), ["vocals"])
            single = make_eval_set.pick_stems(rng, "single_stem", stems)
            self.assertEqual(len(single), 1)
            self.assertNotIn("vocals", single)
            multi = make_eval_set.pick_stems(rng, "multi_stem", stems)
            self.assertGreaterEqual(len(multi), 2)
            self.assertNotIn("vocals", multi)
            mv = make_eval_set.pick_stems(rng, "multi_stem_vocals", stems)
            self.assertIn("vocals", mv)
            self.assertGreaterEqual(len(mv), 2)
        self.assertIsNone(make_eval_set.pick_stems(rng, "vocals", ["bass", "drums"]))
        self.assertIsNone(make_eval_set.pick_stems(rng, "multi_stem", ["bass", "vocals"]))


@unittest.skipIf(h5py is None, "h5py not installed")
class PreprocessTests(unittest.TestCase):
    def test_only_train_tracks_float32_and_no_empty_files(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            root = os.path.join(d, "out")
            args = ["--root", root, "--musdb-root", fx.musdb, "--moises-root", fx.moises,
                    "--splits-dir", fx.splits, "--workers", "2"]
            with redirect_stdout(io.StringIO()):
                preprocess.main(args)
                preprocess.main(args)  # rerun resumes without rewriting
            files = sorted(os.path.relpath(os.path.join(dp, f), root)
                           for dp, _, fs in os.walk(os.path.join(root, "hdf5_train")) for f in fs)
            hdf5 = [f for f in files if f.endswith(".hdf5")]
            self.assertTrue(hdf5)
            self.assertFalse([f for f in files if f.endswith(".tmp")])
            for f in hdf5:
                name = os.path.basename(f)
                self.assertTrue(name.startswith(("musdb18hq__Train ", "moisesdb__song-train__")), name)
                with h5py.File(os.path.join(root, f), "r") as h:
                    self.assertEqual(h["data"].dtype, np.float32)
                    self.assertEqual(h["data"].shape[1:], (2, 6 * SR))
                    self.assertGreater(h["data"].shape[0], 0)
            self.assertNotIn("hdf5_train/bass/musdb18hq__Train B__bass.hdf5", files)
            with open(os.path.join(root, "hdf5_train", "manifest.csv")) as f:
                rows = list(csv.DictReader(f))
            silent = [r for r in rows if r["track_id"] == "Train B" and r["stem"] == "bass"]
            self.assertEqual([r["n_windows"] for r in silent], ["0"])
            with open(os.path.join(root, "README.md")) as f:
                self.assertIn("Generation runs", f.read())
            with redirect_stdout(io.StringIO()):
                problems = check_leakage.find_problems(root, fx.splits)
            # The fixture has only a few stems, so only missing-stem warnings may remain.
            self.assertEqual([p for p in problems if "stem folder" not in p], [])

    def test_leakage_check_catches_a_test_track(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d, seconds=1)
            stem_dir = os.path.join(d, "out", "hdf5_train", "vocals")
            os.makedirs(stem_dir)
            with h5py.File(os.path.join(stem_dir, "musdb18hq__Test A__vocals.hdf5"), "w") as h:
                h["data"] = np.zeros((1, 2, 10), np.float32)
            with open(os.path.join(d, "out", "hdf5_train", "manifest.csv"), "w") as f:
                f.write("hdf5_file\nvocals/musdb18hq__Test A__vocals.hdf5\n")
            with redirect_stdout(io.StringIO()):
                problems = check_leakage.find_problems(os.path.join(d, "out"), fx.splits)
            self.assertTrue(any("comes from a test track" in p for p in problems), problems)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sha256 of the training-path code at upstream commit e84bcac (JusperLee/Apollo). The
# baseline must train exactly as released; any change here is a deliberate experiment
# and belongs on its own branch, with this table updated in the same commit.
UPSTREAM_SHA256 = {
    ("look2hear/datas/musdb_moisesdb_datamodule.py", "compute_mch_rms_dB"): "318b2351e4499dea5d1869f09fa49a116c61d91d20adde431cfe585744c112b2",
    ("look2hear/datas/musdb_moisesdb_datamodule.py", "match2"): "88a65b7c426505493cac30aa9d8a686a8d7ec15a2c72de9a15e3a150e0cecfbf",
    ("look2hear/datas/musdb_moisesdb_datamodule.py", "codec_simu"): "ac2241ad191eee92fb5c4eb9129edf415ef3dde7f628372041fa563bda2da40c",
    ("look2hear/datas/musdb_moisesdb_datamodule.py", "get_wav_files"): "1f2faa279d99f913faf2b4f907129f8d0e4aab0be15c927b57c5a6099b38597f",
    ("look2hear/datas/musdb_moisesdb_datamodule.py", "MusdbMoisesdbDataset"): "d5a25268dd5cf1eb52090114b5a43cae9c3015ce8896497967c34f5b8a344967",
    ("look2hear/datas/musdb_moisesdb_datamodule.py", "MusdbMoisesdbDataModule"): "f5c988cc232b993e4f33e932ebc338c4a91f145c5b1eba91cb803b5c45b98e5b",
    ("look2hear/system/audio_litmodule.py", "training_step"): "c9e155cca6e27a63e030913eb469b473bc8429578e63ca1dbc7ffabe03784438",
    ("look2hear/system/audio_litmodule.py", "configure_optimizers"): "e99ab35e5b94d852446a668577b8cf29406fbae2181476e9e2313212aca4c12a",
    ("look2hear/system/audio_litmodule.py", "on_validation_epoch_end"): "3d047ce3a0f8f5c4b522f3467b800538e615e8b7146efe83f0970b4c8e5c23d8",
    ("look2hear/models/apollo.py", "Apollo"): "a2b4834527f71cc06191f209293402310e7a4d440b2de2cecca6b32b698db28a",
    ("look2hear/models/apollo.py", "BSNet"): "3277f0165ee4939e7712326d5ba15c936a5c23bd7f6b816a4374fce1b868a4fc",
    ("look2hear/models/apollo.py", "Roformer"): "c664151294d5c2f69bef7ac5ecc7f4e4189762900e80c444c20a9317ac621a82",
    ("look2hear/models/apollo.py", "ICB"): "2a7f497314a94a162ea8daa74ac4dcc5d9b4c79fca12d81e3fda0c6fb5a854b5",
    ("look2hear/models/apollo.py", "ConvActNorm1d"): "fe6ec0e6cba5296a3fc717451c4f9dbe390f7af12712f34b092615aeae1bd70e",
    ("look2hear/models/apollo.py", "RMSNorm"): "c4a48a86a1d829fff1e5c569e6e93af7e4f10223024fbb9544ad93e51ed41d0e",
    ("look2hear/losses/gan_losses.py", None): "88637a865b25ae1f9f3a21a9e01a7815be9b97e342491cd7cf1aea730c7a9c1b",
    ("look2hear/discriminators/frequencydis.py", None): "7f0c92a1e312fe91cd0c857efb846e3b3ab4b9b8d106734168e3a03b68ee8e0b",
    ("train.py", None): "07c739cf12f0039f672917d7f39b3d309a19d6487a44b08baef98f8072147ab0",
}


class FidelityTests(unittest.TestCase):
    """The baseline trains with the released code; only eval I/O and run-blocking fixes differ."""

    def test_training_path_matches_upstream(self):
        cache = {}
        for (path, name), expected in UPSTREAM_SHA256.items():
            if path not in cache:
                with open(os.path.join(REPO, path)) as f:
                    src = f.read()
                nodes = {n.name: ast.get_source_segment(src, n) for n in ast.walk(ast.parse(src))
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                cache[path] = (src, nodes)
            src, nodes = cache[path]
            code = src if name is None else nodes.get(name, "")
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(), expected,
                             f"{path}:{name or '(whole file)'} differs from upstream e84bcac")

    def test_released_codec_import_kept(self):
        with open(os.path.join(REPO, "look2hear/datas/musdb_moisesdb_datamodule.py")) as f:
            self.assertIn("from torchaudio.functional import apply_codec", f.read())


class ResidualLagTests(unittest.TestCase):
    def test_measures_known_shift(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.1, (2, 3 * SR)).astype(np.float32)
        for shift in (0, 1, 1105, -37):
            y = np.roll(x, shift, axis=1)
            self.assertEqual(make_eval_set.residual_lag(x, y), shift)


@unittest.skipIf(codec_simu is None or h5py is None,
                 "released codec needs torch 2.0 + torchaudio 2.0.1 with FFmpeg 4 (see look2hear.yml)")
class CodecTests(unittest.TestCase):
    def test_released_codec_simu_runs(self):
        x = torch.from_numpy(np.random.default_rng(0).normal(0, 0.1, (2, 3 * SR)).astype(np.float32))
        for kbps in (24, 128):
            y = codec_simu(x, sr=SR, options={"bitrate": kbps * 1000})
            self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_make_eval_set_end_to_end(self):
        from look2hear.datas.musdb_moisesdb_datamodule import MusdbMoisesdbEval
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            root = os.path.join(d, "out")
            args = ["--root", root, "--musdb-root", fx.musdb, "--moises-root", fx.moises, "--splits-dir",
                    fx.splits, "--split", "test", "--per-bitrate", "4", "--workers", "2"]
            with redirect_stdout(io.StringIO()):
                make_eval_set.main(args)
            with open(os.path.join(root, "test", "test_manifest.csv")) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2 * 6 * 4)
            self.assertEqual({r["track_id"] for r in rows}, {"Test A", "song-test"})
            self.assertEqual({r["category"] for r in rows}, set(make_eval_set.CATEGORIES))
            self.assertTrue(all(r["aligned"] in ("0", "1") for r in rows))
            ds = MusdbMoisesdbEval(os.path.join(root, "test", "test.hdf5"))
            self.assertEqual(len(ds), 48)
            ori, codec, clip_id = ds[5]
            self.assertEqual(clip_id, rows[5]["clip_id"])
            self.assertEqual(tuple(ori.shape), (2, 6 * SR))
            self.assertLessEqual(max(ori.abs().max(), codec.abs().max()).item(), 1.0 + 1e-6)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(check_leakage.find_problems(root, fx.splits), [])


if __name__ == "__main__":
    unittest.main()
