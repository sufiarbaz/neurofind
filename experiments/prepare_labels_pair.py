from pathlib import Path # helps us build paths
from label_utils import load_clean_labels

BASE_DIR = Path(__file__).resolve().parents[1] # takes you to the root file
DATA_DIR = BASE_DIR / "data" / "time_data_labeled" # folder where original CSV files are stored.
OUTPUT_DIR = BASE_DIR / "outputs" # point to the output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # creates output folder is not exist 

A1_LABEL_PATH = DATA_DIR / "33648_A1_TS_dftcorr_01.csv" # points to A1 human-label CSV file, it will be the query stack
A5_LABEL_PATH = DATA_DIR / "33648_A5_TS_dftcorr_02.csv" # points to A5 human-label CSV file, it will be the target stack
OUTPUT_PATH = OUTPUT_DIR / "label_pairs_A1_A5_frame0.csv" # contain the matched A1-A5 human-label pairs.

# pixel per micrometer scale values from TIFF metadata
# A1 metadata: XResolution = (7235890, 1000000), YResolution = (7235890, 1000000)
A1_PIXELS_PER_UM_X = 7235890 / 1000000
A1_PIXELS_PER_UM_Y = 7235890 / 1000000

# A5 metadata: XResolution = (7246376, 1000000), YResolution = (7246376, 1000000)
A5_PIXELS_PER_UM_X = 7246376 / 1000000
A5_PIXELS_PER_UM_Y = 7246376 / 1000000

# read and clean both label CSVs using the shared method
a1_labels = load_clean_labels(A1_LABEL_PATH)
a5_labels = load_clean_labels(A5_LABEL_PATH)

a1_frame0 = a1_labels[a1_labels["FRAME"] == 0].copy()
a5_frame0 = a5_labels[a5_labels["FRAME"] == 0].copy()

pairs = a1_frame0.merge(
    a5_frame0,
    on="TRACK_ID",
    suffixes=("_A1", "_A5"),
)

pairs = pairs[
    [
        "TRACK_ID",
        "ID_A1",
        "ID_A5",
        "POSITION_X_A1",
        "POSITION_Y_A1",
        "POSITION_Z_A1",
        "POSITION_X_A5",
        "POSITION_Y_A5",
        "POSITION_Z_A5",
        "FRAME_A1",
        "FRAME_A5",
    ]
]

pairs = pairs.rename(
    columns={
        "TRACK_ID": "track_id",
        "ID_A1": "query_id",
        "ID_A5": "target_id",
        "POSITION_X_A1": "query_x",
        "POSITION_Y_A1": "query_y",
        "POSITION_Z_A1": "query_z",
        "POSITION_X_A5": "target_x",
        "POSITION_Y_A5": "target_y",
        "POSITION_Z_A5": "target_z",
        "FRAME_A1": "query_frame",
        "FRAME_A5": "target_frame",
    }
)

# Convert A1 physical coordinates from micrometers to pixel coordinates
pairs["query_px_x"] = (pairs["query_x"] * A1_PIXELS_PER_UM_X).round().astype(int)
pairs["query_px_y"] = (pairs["query_y"] * A1_PIXELS_PER_UM_Y).round().astype(int)
pairs["query_px_z"] = pairs["query_z"].round().astype(int)

# convert A5 physical coordinates from micrometers to pixel coordinates
pairs["target_px_x"] = (pairs["target_x"] * A5_PIXELS_PER_UM_X).round().astype(int)
pairs["target_px_y"] = (pairs["target_y"] * A5_PIXELS_PER_UM_Y).round().astype(int)
pairs["target_px_z"] = pairs["target_z"].round().astype(int)

pairs.to_csv(OUTPUT_PATH, index=False)

print("\nNumber of matched A1-A5 FRAME 0 pairs:")
print(len(pairs))

print("\nFirst rows:")
print(pairs.head())

print("\nColumns:")
print(pairs.columns)



