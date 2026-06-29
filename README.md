# Neurofind: Re-identifying dendritic spines across time with an interactive Napari GUI

This project tracks dendritic spines across time frames of a microscopy stack using DINOv3 embeddings. For a spine clicked in one frame, the model finds the most similar point in a later time frame.

Hypothesis 1: "A smaller area around the selected point might give the same match as full candidate-points search but faster."
It compares local search strategies against a full serch baseline, instead of comparing the query against every candidate, it restricts the search to a smaller region around the query point and checks whether that gives the same match, faster.

The strategies compared are:
- full_search: compare against all candidates
- euclidean: candidates inside a circle of a set percentage of the image area
- manhattan: candidates inside a diamond of the same area
- knn: the N nearest candidates (N matched to euclidean's count)

Every prediction is scored by its distance (in micrometers) to the human-labeled target_point.

---

## Requirements
- Python 3.13
- The packages in `requierements.txt`
```
pip install -r requirements.txt
```
---

## Setup files and folders not in this repository
Some files are too larger for GitHub or are third-party, so they are not committed. Get them as follows:

### 1. The large embeddings file
Download `embeddings_dinov3_A5_128_7.npy` (~444 MB) from:
https://drive.google.com/file/d/1QnstOhdov7cS8Lyc_Oa8PR4BiPSZdMaZ/view?usp=sharing

Place it in:
```
data/embeddings/embeddings_dinov3_A5_128_7.npy
```

(The two smaller arrays, `candidate_points_A5.npy` and `candidate_volume_indices_A5.npy`, are included in the repository.)

### 2. The DINOv3 repository
The DINOv3 is a complete repository. Download the `dinov3` folder from:
https://www.studon.fau.de/studon/ilias.php?baseClass=ilrepositorygui&cmdNode=13a:ry&cmdClass=ilObjFileGUI&cmd=sendfile&ref_id=6937306

Place it at the projec root:
```
dinov3
```

### 3. The traned model checkpoint
Download the `spine_embedder_ssl_dinov3_128_7_5`(~342 MB) from:
https://www.studon.fau.de/studon/ilias.php?baseClass=ilrepositorygui&cmdNode=13a:ry&cmdClass=ilObjFileGUI&cmd=sendfile&ref_id=6996853

Place it in:
```
models/spine_embedder_ssl_dinov3_128_7_5pth.sec
```

### 4. The raw A5 TIFF stack
Download `33648_A5_TS_dftcorr.tif` (~144 MB) from:
https://drive.google.com/file/d/1-DsSbYnTTEWkWQHVIQ2LBNG4udu9x1DM/view?usp=sharing

Place it in:
```
data/time_data_labeled/33648_A5_TS_dftcorr.tif
```
---

## How to run
All commands are run from the project root, so that the local `dinvov3` folder is found correctly.

### 1. Generate the query/target pairs
This reads the raw human labels and produces the 40-track pairs file the experiment uses.

```
python experiments/organise_labeled_data.py
```

This creates:
```
data/formatted_human_labels/A5_frame0_to_frame1_pairs.csv
```

### 2. Run the Hypothesis 1 experiment
```
python -m experiments.local_search.run_experiment
```

Results are written to:
```
outputs/h1_local_search/A5_frame0_to_frame1_search_strategy_results.csv
```

### 3. Interactive GUI
Opens a napari window with the query and target frames. Click a spine in th query frame to see the model's predicted match in the target frame.
```
python GUI/napari_gui.py
```
---

## Notes
- All coordinates and distances in the experiment output are in micrometers. Z is treated as a slice index (the TIFF has no physical-spacing value), so distances are measured in the XY plane only.
- The microscope calibration is 0.138 um/pixels (7.246376 pixels/um), taken from the TIFF metadata.
- `compute_full_embeddings_hpc.py` generates the candidate embeddings on an HPC machine  with a GPU; the committed `.npy` files are its output. It is included to document how the embeddings were produced, not as a step you need to run.
---

## Project layout
```
Neurofind Project/
├── configs/                 experiment configuration (search_strategy.yaml)
├── data/
│   ├── time_data_labeled/   raw TIFF stack and raw human-label CSV
│   ├── formatted_human_labels/  generated query/target pairs
│   └── embeddings/          candidate points, volume indices, (embeddings via Drive)
├── experiments/
│   ├── organise_labeled_data.py        builds the pairs file
│   ├── compute_full_embeddings_hpc.py  generates embeddings (HPC)
│   └── local_search/        the H1 experiment package
├── GUI/
│   ├── utils.py             shared model + helper functions
│   └── napari_gui.py        interactive viewer
├── models/                  trained model checkpoint (from the link, not committed)
├── dinov3/                  DINOv3 repository (from the link, not committed)
└── outputs/                 experiment results
```