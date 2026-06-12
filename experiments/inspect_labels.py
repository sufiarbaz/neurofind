from pathlib import Path
from label_utils import load_clean_labels

# Get the main project folder.
# This file is inside experiments/, so parents[1] gives the project root.
BASE_DIR = Path(__file__).resolve().parents[1]

# Folder containing the labeled time-series data.
DATA_DIR = BASE_DIR / "data" / "time_data_labeled"

# A1 label CSV file.
A1_LABEL_PATH = DATA_DIR / "33648_A1_TS_dftcorr_01.csv"

# A5 label CSV file.
A5_LABEL_PATH = DATA_DIR / "33648_A5_TS_dftcorr_02.csv"

# Load and clean A1 labels.
a1_labels = load_clean_labels(A1_LABEL_PATH)

# Load and clean A5 labels.
a5_labels = load_clean_labels(A5_LABEL_PATH)

# Select FRAME 0 labels
# Current experiment loads stack[0], so labels must also come from FRAME 0.
a1_frame0 = a1_labels[a1_labels["FRAME"] == 0].copy()

# Select A5 labels from FRAME 0.
a5_frame0 = a5_labels[a5_labels["FRAME"] == 0].copy()

# Find shared TRACK_IDs
# Find track IDs that appear in both A1 frame 0 and A5 frame 0.
shared_frame0_track_ids = sorted(
    set(a1_frame0["TRACK_ID"]).intersection(set(a5_frame0["TRACK_ID"]))
)

# Print inspection output
print("Clean A1:")
print(a1_labels.head())
print("A1 shape:", a1_labels.shape)

print("\nClean A5:")
print(a5_labels.head())
print("A5 shape:", a5_labels.shape)

print("\nA1 FRAME 0 shape:", a1_frame0.shape)
print("A5 FRAME 0 shape:", a5_frame0.shape)

print("\nNumber of shared TRACK_IDs at FRAME 0:", len(shared_frame0_track_ids))
print("Shared FRAME 0 TRACK_IDs:", shared_frame0_track_ids)

print("\nExample FRAME 0 matching pairs:")

for track_id in shared_frame0_track_ids[:10]:
    print("\nTRACK_ID:", track_id)

    print("A1 FRAME 0:")
    print(a1_frame0[a1_frame0["TRACK_ID"] == track_id])

    print("A5 FRAME 0:")
    print(a5_frame0[a5_frame0["TRACK_ID"] == track_id])