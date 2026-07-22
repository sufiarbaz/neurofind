"""
config.py

All file paths and experiments constants for Hypothesis 2 - Grid search.

To change stacks or frames, change the relevant variables here and everything else is built from it.
"""

from pathlib import Path
from src.base import stack_settings

# choose stack and frames
STACK_NAME = stack_settings.STACK_NAME # short name for the stack, e.g., "A5" or "A1"
STACK_FILENAME = stack_settings.STACK_FILENAME # place this raw TIFF in: data/time_data_labeled/

QUERY_FRAME = stack_settings.QUERY_FRAME # the frame where the query comes from
TARGET_FRAME = stack_settings.TARGET_FRAME # the frame we search for the match

# pixels per micrometer and stack width & height. If you dont know these values, run src/base/read_stack_info.py on the TIFF.
PIXELS_PER_UM_X = stack_settings.PIXELS_PER_UM_X
PIXELS_PER_UM_Y = stack_settings.PIXELS_PER_UM_Y
STACK_WIDTH_PX = stack_settings.STACK_WIDTH_PX
STACK_HEIGHT_PX = stack_settings.STACK_HEIGHT_PX

BASE_DIR = Path(__file__).resolve().parents[3] # path to the project root

DATA_DIR = BASE_DIR / "data" / "time_data_labeled" # folder with the raw TIFF
OUTPUT_DIR = BASE_DIR / "outputs" # folder where results are written
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings" # folder with the precomputed embeddings

STACK_PATH = DATA_DIR / STACK_FILENAME # path to the stack file
MODEL_PATH = BASE_DIR / "models" / "spine_embedder_ssl_dinov3_128_7_5pth.sec" # place the trained model in models/ and update the model name here.

# The precomputed files. They lives in data/embeddings/
# They are produced by running src/base/compute_full_embeddings_hpc.py on the stack.
TARGET_CANDIDATES_PATH = EMBEDDINGS_DIR / f"candidate_points_{STACK_NAME}.npy"
TARGET_EMBEDDINGS_PATH = EMBEDDINGS_DIR / f"embeddings_dinov3_{STACK_NAME}.npy"
TARGET_VOLUME_INDICES_PATH = EMBEDDINGS_DIR / f"candidate_frame_numbers_{STACK_NAME}.npy"
EMBEDDING_RATE_PATH = EMBEDDINGS_DIR / f"embedding_rate_{STACK_NAME}.json" # how long the model took to make one fingerprint, measured during the HPC run.

# query and target pairs file
PAIRS_FILENAME = f"{STACK_NAME}_frame{QUERY_FRAME}_to_frame{TARGET_FRAME}_pairs.csv"
LABEL_PAIRS_PATH = BASE_DIR / "data" / "formatted_human_labels" / PAIRS_FILENAME

# experiment's results folder and file
H2_OUTPUT_DIR = OUTPUT_DIR / "h2_grid_search"
OUTPUT_RESULTS_PATH = H2_OUTPUT_DIR / f"{STACK_NAME}_frame{QUERY_FRAME}_to_frame{TARGET_FRAME}_grid_search_results.csv"

# crop size taken around each point before making its fingerprint (the embedding)
CROP_SIZE_Z = stack_settings.CROP_SIZE_Z
CROP_SIZE_Y = stack_settings.CROP_SIZE_Y
CROP_SIZE_X = stack_settings.CROP_SIZE_X

# grid resolutions to test, an NxN grid cuts the image into N columns and N rows
GRID_SIZES_TO_TEST = [2, 4, 8, 16, 32]

# run setting, set to a small number to run the first few pairs
NUMBER_OF_PAIRS_TO_RUN = None

# add blank row between tracks for better readability
ADD_BLANK_ROW_BETWEEN_TRACKS = True

# columns of the result file, all distances are in micrometers
OUTPUT_COLUMNS = [
    "track_id",
    "strategy",
    "grid_size",
    "number_of_candidates",
    "query_point_z,y,x",
    "predicted_point_z,y,x",
    "target_point_z,y,x",
    "runtime_seconds",
    "embedding_time_seconds",
    "total_time_seconds",
    "target_in_click_cell",
    "distance_to_human_label",
    "query_frame",
    "target_frame"
]





