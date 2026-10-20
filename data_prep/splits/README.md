# Split files

| File | Columns | Source |
|---|---|---|
| `musdb18hq.csv` | `track,split` | MUSDB18-HQ official `train/` (100) and `test/` (50) folders. The 14 `validation_tracks` of SigSep's [`musdb/configs/mus.yaml`](https://github.com/sigsep/sigsep-mus-db/blob/master/musdb/configs/mus.yaml) are `val`; the other 86 training tracks are `train`. |
| `moisesdb.csv` | `song_id,split,banquet_fold` | Banquet folds from [`kwatcharasupat/query-bandit` `reproducibility/splits.csv`](https://github.com/kwatcharasupat/query-bandit/blob/79ed5bb75e5c3a40cd319d9d990cee913fc65c26/reproducibility/splits.csv) (commit `79ed5bb`, MIT licence, © 2024 Karn Watcharasupat). Folds 1–3 → `train` (144), fold 4 → `val` (48), fold 5 → `test` (48), the defaults of that repository's `MoisesDB` dataset code. |

Banquet: K. N. Watcharasupat and A. Lerch, "A Stem-Agnostic Single-Decoder System for
Music Source Separation Beyond Four Stems," ISMIR 2024, arXiv:2406.18747.

The Banquet folds are stratified by genre, not grouped by artist: 30 of MoisesDB's 47
artists have songs in more than one fold (e.g. Firefly, 60 songs). Songs never cross
splits; artists can.
