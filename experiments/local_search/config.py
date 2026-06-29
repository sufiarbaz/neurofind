"""
config.py

All file paths and experiment constants for the Hypothesis 1 - Local search live here and nowhere else.

If the project is moved to a different machine, this is the only file that should need editing.
"""

from pathlib import Path # build OS independent file paths
# Project root, found from this file's own location
# This file is at: Neurofind Project / experiments / local_search / config.py
# parents[0] = local_search/, parents[1] = experiments/, parents[2] = project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Folder containing the shared utils.py
GUI_DIR = BASE_DIR / "GUI"

# Folder with the raw TIFF stack and the raw human-label CSV
DATA_DIR = BASE_DIR / "data" / "time_data_labeled"

# Folder where experiment result CSVs are written
OUTPUT_DIR = BASE_DIR / "outputs"

# Folder with the precomputed candidate points, embeddings, and volume indices
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

# The raw A5 TIFF stack (the actual image pixel data).
A5_STACK_PATH = DATA_DIR / "33648_A5_TS_dftcorr.tif"

# The trained DINOv3 spine-embedding model
MODEL_PATH = BASE_DIR / "models" / "spine_embedder_ssl_dinov3_128_7_5pth.sec"

# The three precomputed embedding files (from compute_full_embeddings_hpc.py)
TARGET_CANDIDATES_PATH = EMBEDDINGS_DIR / "candidate_points_A5.npy"
TARGET_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings_dinov3_A5_128_7.npy"
TARGET_VOLUME_INDICES_PATH = EMBEDDINGS_DIR / "candidate_volume_indices_A5.npy"

# The query/target pairs file (produced by organise_labeled_data.py)
LABEL_PAIRS_PATH = BASE_DIR / "data" / "formatted_human_labels" / "A5_frame0_to_frame1_pairs.csv"

# Where this experiment writes its results.
H1_OUTPUT_DIR = OUTPUT_DIR / "h1_local_search"
OUTPUT_RESULTS_PATH = H1_OUTPUT_DIR / "A5_frame0_to_frame1_search_strategy_results.csv"

# Pixels per micrometer for the A5 stack
A5_PIXELS_PER_UM_X = 7.246376
A5_PIXELS_PER_UM_Y = 7.246376

A5_STACK_WIDTH_PX = 512
A5_STACK_HEIGHT_PX = 512

# Crop size (slices, height width) taken around each point for the model
CROP_SIZE_Z = 7
CROP_SIZE_Y = 128
CROP_SIZE_X = 128

# Percentages of total imaged area to search, for the euclidean and manhattan strategies. KNN is matched to euclidean's resulting candidate count.
AREA_PERCENTAGES_TO_TEST = [0.25, 0.40]

# The local search strategies to run, in addition to full search
LOCAL_STRATEGIES_TO_RUN = ["euclidean", "manhattan", "knn"]

# Set to a small integer to process only the first N query pairs
NUMBER_OF_PAIRS_TO_RUN = None

# Adds one blank row between tracks in the output CSV, for readability
ADD_BLANK_ROW_BETWEEN_TRACKS = True

# Columns that the results CSV will contain, all coordinates and distances are in MICROMETERS
OUTPUT_COLUMNS = [
    "track_id",   
    "strategy",
    "local_area_requested",
    "number_of_candidates",
    "query_point_z,y,x",
    "predicted_point_z,y,x",
    "target_point_z,y,x",
    "runtime_seconds",
    "distance_to_human_label",
    "area_clipped_fraction",
    "query_frame",
    "target_frame"
]
