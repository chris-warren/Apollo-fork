# Apollo data preparation (`apollo_v1`)

> **Branch `exp/codec-alignment`.** One change from the baseline (`data-prep-splits`):
> `codec_simu` cuts the MP3 path's fixed 1,105-sample delay instead of re-aligning with
> `match2()`, so every training pair is time-aligned (40/40 real clips at each of the six
> bitrates, versus about 85–93 % with the released code). The MP3 encoder and decoder,
> the per-worker bitrate behaviour and everything else are unchanged, so a model trained
> here differs from the baseline only in alignment. Evaluate both on the same
> `apollo_v1` test set; the `aligned:*` rows of `metrics_breakdown.csv` separate clips
> that the released codec left misaligned.

This folder (`data_prep/` in the Apollo repo) builds leak-free training data and fixed
validation/test sets for Apollo from MUSDB18-HQ and MoisesDB. A copy of this file is
written next to the generated data, with a log of every generation run appended at
the end (command, git commit, seed, counts).

It replaces the authors' separate preprocessing repo,
[Apollo-data-preprocess](https://github.com/JusperLee/Apollo-data-preprocess)
(commit `bb37a7b`, 2025-03-09; "upstream" below). Every behaviour that differs from
upstream or from the released Apollo data code is listed under
[Deviations](#deviations-from-upstream), each with its reason.

**Baseline first.** The first goal is to reproduce Apollo as released. Where the
released code differs from the paper, the released code is followed; nothing that
changes what the model trains on is altered (see [Kept as released](#kept-as-released)).
`tests/test_data_prep.py::FidelityTests` checks that the training path — data loader,
MP3 codec, model, losses, discriminator, training step, and `train.py`'s `train()` —
is byte-identical to upstream commit `e84bcac`. (`train.py`'s startup code additionally
loads `.env`, resolves the config so the saved `config.yaml` records real paths, and
checks that the data paths exist.) Improvements are separate experiments on their own
branches (e.g. `exp/codec-alignment`), measured against this baseline.

## Commands

```bash
# 1. training data (train split only)
python -m data_prep.preprocess --root <data>/processed/apollo_v1 \
    --musdb-root <musdb18hq> --moises-root <moisesdb>/moisesdb_v0.1
# 2. validation and test sets (held-out songs only)
python -m data_prep.make_eval_set --root <data>/processed/apollo_v1 --split val  --per-bitrate 25 \
    --musdb-root <musdb18hq> --moises-root <moisesdb>/moisesdb_v0.1
python -m data_prep.make_eval_set --root <data>/processed/apollo_v1 --split test --per-bitrate 2500 \
    --musdb-root <musdb18hq> --moises-root <moisesdb>/moisesdb_v0.1
# 3. leakage check (exit status 1 on any violation)
python -m data_prep.check_leakage --root <data>/processed/apollo_v1
```

Training config: `datas.train_dir: <...>/apollo_v1/hdf5_train` and
`datas.eval_dir: <...>/apollo_v1/eval/val.hdf5`. Testing:
`python test.py --conf_dir <config> --test_h5 <...>/apollo_v1/test/test.hdf5`
(add `--max-per-bitrate 250` for the quick development subset).

**Environment.** Preprocessing needs only `numpy`, `soundfile` and `h5py`. Eval-set
generation and training need the released MP3 path, i.e. the versions pinned in
`look2hear.yml`: torch 2.0.0, torchaudio 2.0.1 **and FFmpeg 4.x shared libraries**
(`ffmpeg=4.2.2` from conda). torchaudio 2.0.1 encodes MP3 with its bundled sox + LAME
(the line `formats: mp3 can't encode MPEG audio ... to 16-bit` it prints is a harmless
sox warning) but decodes MP3 only through FFmpeg, and it can load only FFmpeg 4. With
FFmpeg 5/6 or none, `apply_codec` fails with `Failed to load audio from <BytesIO>`.
Newer torchaudio deprecated and later removed `apply_codec`, so it is not an option.
A working environment without root:

```bash
micromamba create -p <env> -c defaults -c conda-forge python=3.10 ffmpeg=4.2.2 lame=3.100
<env>/bin/python -m pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
<env>/bin/python -m pip install "numpy<2" soundfile h5py pytorch-lightning==2.2.1 hydra-core==1.3.2 "setuptools<70"
```

## Output layout

```
apollo_v1/
  README.md                         this file + generation log
  hdf5_train/<stem>/<dataset>__<track>__<file>.hdf5   "data": (n_windows, 2, 264600) float32
  hdf5_train/manifest.csv           source file and window count of every HDF5 file
  eval/val.hdf5, eval/val_manifest.csv       "ori", "codec": (N, 2, 264600) float32, "clip_id"
  test/test.hdf5, test/test_manifest.csv
```

`<stem>` is one of the 11 MoisesDB stem names or MUSDB's `bass`, `drums`, `other`,
`vocals`, `mixture`. Each manifest row of the eval sets records the song, stems,
start sample, bitrate, category and seed of the clip, so every clip can be rebuilt.

## Splits

| Dataset | Train | Val | Test | Source |
|---|---|---|---|---|
| MUSDB18-HQ | 86 | 14 | 50 | Official `train/` and `test/` folders; validation = the 14 `validation_tracks` of SigSep's [`musdb/configs/mus.yaml`](https://github.com/sigsep/sigsep-mus-db/blob/master/musdb/configs/mus.yaml) |
| MoisesDB | 144 | 48 | 48 | Folds of Banquet (Watcharasupat & Lerch, ISMIR 2024), [`reproducibility/splits.csv`](https://github.com/kwatcharasupat/query-bandit/blob/79ed5bb75e5c3a40cd319d9d990cee913fc65c26/reproducibility/splits.csv) (MIT licence): folds 1–3 train, 4 val, 5 test, as in that repository's code |

The files are `splits/musdb18hq.csv` and `splits/moisesdb.csv` (the latter keeps the
Banquet fold number). `splits.py` refuses to run if any track on disk is not in
exactly one split, or a listed track is missing.

**Why these splits.** The Apollo paper (arXiv 2409.08514v2, 7 Jan 2025) says only that
the model was "trained and tested on the combined MUSDB18-HQ and MoisesDB" and tested
"on the combined test set ... (with 5000 samples for each case)"; it does not list songs,
and validation is mentioned only for early stopping (patience 20). MoisesDB has no
official split; the Banquet folds are published, genre-stratified, and reused by later
MoisesDB work (144/48/48).

**Caveat.** The Banquet folds are stratified by genre, not grouped by artist: 30 of
MoisesDB's 47 artists have songs in more than one fold. No *song* is shared between
splits, but some *artists* are.

## Deviations from upstream

1. **The MUSDB18-HQ test set is not used for training.** Upstream `musdb_preprocess.py`
   walks both `musdb18hq/train` and `musdb18hq/test` into the training HDF5, so a model
   trained on it has heard all 50 test songs. Here only train-split tracks are written.
2. **Validation and test songs are held out of training** (14 MUSDB, 96 MoisesDB).
   Upstream processes every song of both datasets.
3. **float32 instead of float64.** Upstream creates the dataset with `dtype=float`
   (float64). The loader converts every window to float32 anyway
   (`torch.FloatTensor`), so values are identical and the data is half the size.
4. **No empty HDF5 files.** Upstream (MUSDB script) creates the file before running VAD,
   so a stem with no active window leaves a file without `data`, which raises a
   `KeyError` whenever training picks it. Here such sources are skipped and listed in
   `manifest.csv` with `n_windows = 0`.
5. **File names** are `<dataset>__<track>__<file>.hdf5` instead of `sample<n>.hdf5`, so
   every file can be traced to its source (and checked by `check_leakage.py`). The
   number of files is unchanged — one per source WAV — so the loader's sampling (a
   uniformly random file of a stem, then a random window) is unchanged.
6. **Atomic, resumable writes.** Each file is written as `*.tmp` and renamed when
   complete; a rerun skips finished files. (The output drive may be unplugged.)
7. **Unchanged from upstream:** the `VAD()` function (copied verbatim, see
   `preprocess.py`), its parameters (10 ms frames, 15 % power quantile, >50 % active),
   6 s windows with a 3 s hop, 44.1 kHz stereo, and reading audio as float64.
8. **Validation and test sets are generated here.** Neither upstream nor the Apollo repo
   contains code for the `./eval` and `./codec-test` folders the training and test code
   read. Choices:
   - **6 s clips**, the length of the authors' example clips in `asserts/`.
   - **Within-song mixes.** Each clip is a subset of one held-out song's stems,
     time-aligned at their original levels (no random gain), so it is real music
     rather than the cross-song random mixes used for training.
   - **Four categories** mirroring the paper's Table II, assigned in rotation
     (`i % 4`, so with 25 clips the first category gets 7 and the others 6):
     `vocals` (vocals only), `single_stem` (one non-vocal stem), `multi_stem` (2 or
     more non-vocal stems), `multi_stem_vocals` (vocals + 1 or more others). The number
     of stems is drawn uniformly.
   - **Every chosen stem must be active** in the clip's window, by the same rule `VAD()`
     uses for training windows; windows lie on the same 3 s grid. Songs are drawn
     uniformly, so every held-out song can contribute.
   - MoisesDB stems made of several files are summed; stems of different lengths in a
     song (50 songs have them) are zero-padded to the longest.
   - **MP3 and normalisation as in training:** the released `codec_simu`, unchanged,
     at a fixed bitrate, then both signals divided by the larger of their peaks.
   - **Alignment flag.** The released `codec_simu` leaves about 7–15 % of clips out of
     step with the target (see below). Each clip's remaining offset is measured
     (plain cross-correlation of the channel sums, ±4,000 samples) and stored in the
     manifest as `residual_lag_samples` and `aligned`; `test.py` reports every metric
     over all clips (comparable to the paper) and over aligned clips only
     (`aligned:*` rows of `metrics_breakdown.csv`).
   - **Sizes:** validation 25 clips per bitrate per dataset (300; it runs every epoch
     and the paper gives no size); test **2,500 per bitrate per dataset = 5,000 per
     bitrate**, our reading of the paper's "5000 samples for each case" (Fig. 2 compares
     six bitrates): 30,000 clips. Because each dataset/bitrate has its own random
     stream, the first 250 per bitrate are exactly a 3,000-clip set — the quick
     development subset (`test.py --max-per-bitrate 250`). The 98 test songs hold about
     3,900 distinct 6 s windows, so clips reuse windows with different stem subsets.
   - **Fixed seeds** (val 20251019, test 20251020); the per-dataset, per-bitrate random
     streams are seeded independently, so changing one part does not reshuffle another.
9. **Packed eval files.** One HDF5 file per split instead of one folder per clip
   (`ori_wav.wav`, `codec_wav.wav`, plus `ests_wav.wav` written by `test.py`). The data
   is meant for a server with a 1,000,000-file quota: the folder layout would use about
   120,000 files for this 30,000-clip test set, the packed one uses 3 regardless of
   size. `MusdbMoisesdbEval` reads the packed file (and still reads
   the folder layout), and `test.py` writes all estimates to one `test_estimates.hdf5`.
10. **Fixes needed to run the released training/test code** (no effect on training):
    `validation_step` unpacked two values from batches that have three (clip id), which
    failed at the first validation; `test.py` called a model class that does not exist
    (`GullFullband`, now `Apollo`), read a hard-coded `./codec-test`, and did not put the
    model in eval mode (Apollo's only dropout layers are built with probability 0 and it
    has no batch norm, so this changes nothing numerically). `test.py` now also writes
    metrics per dataset, bitrate and category (`metrics_breakdown.csv`), matching the
    paper's Fig. 2 and Table II, and writes all estimates to one HDF5 file.

## Kept as released

These behaviours of the released code differ from the paper's description or are
plainly unintended, but they are what the authors trained with, so the baseline keeps
them. Each is a candidate experiment once the baseline exists.

- **MP3 misalignment.** `codec_simu` re-aligns the decoded MP3 with `match2()`
  (phase-transform cross-correlation, first channel only), then `torch.roll`s it. The
  codec delay is a constant 1,105 samples (25 ms), but on 60 random real 3 s clips
  per bitrate, `match2` left 9, 7, 5, 5, 4, 4 clips (24 … 128 kbit/s) misaligned —
  mostly by the full 1,105 samples (it returned 0), occasionally by 1–2 samples or by
  thousands. So about 7–15 % of training pairs are out of step with their targets.
- **Bitrate fixed per worker per epoch.** `codec_simu` writes the random bitrate back
  into the shared `codec_options` (an OmegaConf `DictConfig` from hydra), so each
  dataloader worker uses the bitrate it drew for its first sample for the rest of the
  epoch; workers are recreated (and reseeded) every epoch. With 8 workers per GPU an
  epoch therefore sees at most 8 bitrate draws per GPU — in a simulation, one epoch had
  no 24/32/48 kbit/s samples at all — while over many epochs every bitrate is equally
  likely. The paper describes a random bitrate per sample.
- **MP3 encoder settings** are sox's defaults for LAME at the given bitrate (encoded by
  torchaudio's bundled sox, decoded by FFmpeg 4), not an explicit `ffmpeg` command.

## Storage and file counts

| Output | Size | Files |
|---|---|---|
| `hdf5_train/` | 154 GB (1,748 HDF5 files, 77,629 windows) | 1,749 |
| `eval/` (300 clips) | 1.2 GB | 2 |
| `test/` (30,000 clips) | about 127 GB | 2 (+1 `test_estimates.hdf5` from `test.py`) |

Raw datasets are best kept as one `.tar` per dataset on the server (2 files instead of
5,341) and unpacked to local scratch only when re-preprocessing.
