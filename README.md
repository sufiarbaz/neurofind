# Neurofind: Re-identifying dendritic spines across time with an interactive Napari GUI

This project tracks dendritic spines across time frames of a microscopy stack using DINOv3 embeddings. The controlled experiments are performed using human-labeled data as query point. For a spine clicked in one frame in the napari-GUI, the model finds the most similar point in a later time frame.

**Hypothesis 1 (local search):** *"A smaller area around the selected point might give the same match as full candidate-points search but faster."*
It compares local search strategies against a full serch baseline, instead of comparing the query against every candidate, it restricts the search to a smaller region around the query point.

The strategies compared are:
- `full_search`: compare against all candidates (the baseline)
- `euclidean`: candidates inside a circle, sized as a % of the total imaged area
- `manhattan`: candidates inside a diamond of the same area

**Hypothesis 2 (grid search):** *"Dividing the image into a fixed grid and searching only the cell the query point falls into gives a trade-off between accuracy and search time, controlled by the grid size."* The image is divided into a fixed NxN grid; only the cell containing the click is searched.
Grid sizes tested: 2x2, 4x4, 8x8, 16x16, 32x32.

Every prediction, in both hypotheses, is scored by its distance (in micrometers) to the human-labeled target point.

"The experiments of both the hypotheses have been run across **3 stacks (A1, A5, B2) x 5 consecutive time frame pairs (0→1, 1→2, 2→3, 3→4, 4→5)**, totaling 427 tracked observations."

---

## Requirements
- Python 3.13
- The packages in `requirements.txt`
```
pip install -r requirements.txt
```
---

## Setup files and folders not in this repository
Some files are too large for GitHub or are third-party, so they are not committed. Get them as follows:

### 1. The DINOv3 repository
The DINOv3 is a complete repository. Download the `dinov3` folder from:
https://drive.google.com/file/d/1_hshul_L6XlYqSQvfVEGIgRMmWXVm5KU/view?usp=sharing

Place it at the project root:
```
dinov3
```

### 2. The trained model checkpoint
Download the `spine_embedder_ssl_dinov3_128_7_5`(~342 MB) from:
https://drive.google.com/file/d/1Sy0CjT9RPsIzBtRJd0aFc2TIAuZIPq0U/view?usp=sharing

Place it in:
```
models/spine_embedder_ssl_dinov3_128_7_5pth.sec
```

### 3. The raw TIFF stacks
Download `33648_A1_TS_dftcorr.tif` (~42 MB) from:
https://drive.google.com/file/d/10EOQ5xvHCBFRxqYRFBydbUhk0UW_n7vV/view?usp=sharing
Download `33648_A5_TS_dftcorr.tif` (~144 MB) from:
https://drive.google.com/file/d/1-DsSbYnTTEWkWQHVIQ2LBNG4udu9x1DM/view?usp=sharing
Download `33648_B2_TS_dftcorr.tif` (~132 MB) from:
https://drive.google.com/file/d/1qm5xxdMh7pTYx9sns91dzdoICs-4irDv/view?usp=sharing

Place them in `data/time_data_labeled/`:
```
data/time_data_labeled/33648_A1_TS_dftcorr.tif
data/time_data_labeled/33648_A5_TS_dftcorr.tif
data/time_data_labeled/33648_B2_TS_dftcorr.tif
```

The raw human-label CSVs are small and already committed to this repository (see
`.gitignore`), so they need no separate download:
```
data/time_data_labeled/33648_A1_TS_dftcorr_01.csv
data/time_data_labeled/33648_A5_TS_dftcorr_02.csv
data/time_data_labeled/33648_B2_TS_dftcorr_04.csv
```

### 4. The candidate embeddings
The large fingerprints file, `embeddings_dinov3_<STACK>.npy`, is not in the repository.
Get it one of two ways:
**Download** from below links
Download `embeddings_dinov3_A1.npy` (~124 MB) from:
https://drive.google.com/file/d/1u5R_eeNqI9vSCsAtW7yj2M3tmOogKWoY/view?usp=sharing
Download `embeddings_dinov3_A5.npy` (~444 MB) from:
https://drive.google.com/file/d/1QnstOhdov7cS8Lyc_Oa8PR4BiPSZdMaZ/view?usp=sharing
Download `embeddings_dinov3_B2.npy` (~259 MB) from:
https://drive.google.com/file/d/1avsQqacKAMOtEU_cKiRnTrMx9ewBWjYh/view?usp=sharing
**Or regenerate it yourself**, by running `src/base/compute_full_embeddings_hpc.py` 
on an HPC machine with a GPU (roughly 3-4 hours per stack; measured ~3.5 hours for
~910,000 candidates on an RTX 2080 Ti). This also regenerates the other three files,
so if you do this you do not need to rely on the committed copies.

For each stack (A1, A5, B2), four files live in `data/embeddings/`:
```
candidate_points_<STACK>.npy          -- small, already committed to this repository
candidate_frame_numbers_<STACK>.npy   -- small, already committed to this repository
embedding_rate_<STACK>.json           -- tiny, already committed to this repository
embeddings_dinov3_<STACK>.npy         -- NOT committed (hundreds of MB per stack)
```

---

## Which stack and frames a run uses
**One place controls this: `src/base/stack_settings.py`.** It holds the stack name, the
raw filenames, the two frame numbers, and that stack's calibration. To run on a
different stack or a different frame pair, edit this file — nothing else needs to
change. Both experiment configs and the GUI read from it.

If you don't know a new stack's calibration or pixel dimensions, run:
```
python -m src.base.read_stack_info
```
It reads the TIFF named in `stack_settings.py` and prints the numbers to paste back in.

---

## How to run
All commands are run from the `project root`, so that the local `dinov3` folder is found correctly.

The query/target pairs file (the ground truth the experiments score against) is created
automatically from the raw human labels the first time an experiment runs, if it isn't
there yet — see `src/experiments/shared/load_data.py`. You do not need to run anything
for this yourself.

If you want to create it ahead of time, or inspect it before running an experiment:
```
python -m src.base.organise_labeled_data
```
This creates, e.g.:
```
data/formatted_human_labels/A5_frame0_to_frame1_pairs.csv
```

### 1. Compute the candidate embeddings (once per stack, on the HPC)
If you want to use the pre-computed embeddings then just download it from above links (`See setup files and not in this repository`) section in this document. Otherwise you can run generate it by yourself by executing the below command from the project root.
```
python -m src.base.compute_full_embeddings_hpc <STACK_NAME> <STACK_FILENAME>
```
e.g. `python -m src.base.compute_full_embeddings_hpc A5 33648_A5_TS_dftcorr.tif`

If run with no arguments, it uses the stack named in `stack_settings.py` instead.
Takes roughly 3-4 hours per stack on a single GPU.

### 2. Run the Hypothesis 1 experiment (local search)
```
python -m src.experiments.local_search.run_experiment
```
Results are written to:
```
outputs/h1_local_search/<STACK>_frame<Q>_to_frame<T>_search_strategy_results.csv
```
Takes a few minutes to a few tens of minutes per stack/frame pair, depending on how
many candidates fall inside the search regions and whether a GPU is available.

### 3. Run the Hypothesis 2 experiment (grid search)
```
python -m src.experiments.grid_search.run_experiment
```
Results are written to:
```
outputs/h2_grid_search/<STACK>_frame<Q>_to_frame<T>_grid_search_results.csv
```

### 4. Interactive GUI
Opens a napari window with the query and target frames side by side. Click a spine in
the query frame to see the model's predicted match in the target frame.
```
python -m src.GUI.napari_gui
```

---

## Notes
- All coordinates and distances in the experiment outputs are in micrometers. Z is
  treated as a slice index (the TIFF has no physical-spacing value), so distances are
  measured in the XY plane only.
- Each stack's calibration is read from its own TIFF metadata and set in
  `stack_settings.py` — it is not assumed to be the same across stacks.
- `compute_full_embeddings_hpc.py` also measures how long the model takes to embed one
  candidate, and saves it (`embedding_rate_<STACK>.json`). The experiments use this to
  report `embedding_time_seconds` and `total_time_seconds` — the honest cost of a search
  if the fingerprints were not already pre-computed.

---

## Project layout
```
neurofind/
├── data/
│   ├── time_data_labeled/      #raw TIFF stacks and raw human-label CSVs
│   ├── formatted_human_labels/      #generated query/target pairs
│   └── embeddings/      #candidate points, fingerprints, frame numbers, rates
├── src/
│   ├── base/
│   │   ├── stack_settings.py      #which stack and frames everything uses
│   │   ├── utils.py      #the model class and shared helpers
│   │   ├── organise_labeled_data.py      #builds the pairs file
│   │   ├── compute_full_embeddings_hpc.py      #generates embeddings (HPC)
│   │   └── read_stack_info.py      #reads a new stack's calibration
│   ├── experiments/
│   │   ├── shared/      #tools used by both experiments (matching, scoring, ...)
│   │   ├── local_search/      #the H1 experiment package
│   │   └── grid_search/      #the H2 experiment package
│   └── GUI/
│       └── napari_gui.py      #interactive viewer
├── logs/
│   └── experiment_log.xlsx      #logs of experiments
├── models/      #trained model checkpoint (not committed)
├── dinov3/      #DINOv3 repository (not committed)
└── outputs/      #experiment results
```