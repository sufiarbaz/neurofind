"""
organise_labeled_data.py

Turns the raw, human-labeled CSV into the clean query/target pairs file.

Run this ONCE, before running any experiment. It writes the pairs file into data/formatted_human_labels/, and the experiments pick it up from there.

To change stack or frame, change the relevant variables. The output filename is built from it.

How to run it (from the project root):
    python -m src.base.organise_labeled_data
"""

from pathlib import Path  # build file paths that work on any machine
import pandas as pd  # read, clean and reshape the label CSV as a table
from src.base import stack_settings

# choose stack and frames
STACK_NAME = stack_settings.STACK_NAME # short name of the stack, e.g. "A5" or "A1"
RAW_LABELS_FILENAME = stack_settings.RAW_LABELS_FILENAME # place the raw label CSV in: data/time_data_labeled/

QUERY_FRAME = stack_settings.QUERY_FRAME # the frame where the query comes from
TARGET_FRAME = stack_settings.TARGET_FRAME # the frame where we search for the match

BASE_DIR = Path(__file__).resolve().parents[2] # path to the project root

RAW_LABELS_PATH = BASE_DIR / "data" / "time_data_labeled" / RAW_LABELS_FILENAME
FORMATTED_LABELS_DIR = BASE_DIR / "data" / "formatted_human_labels"

# output file names
PAIRS_FILENAME = f"{STACK_NAME}_frame{QUERY_FRAME}_to_frame{TARGET_FRAME}_pairs.csv"

# output file paths
PAIRS_OUTPUT_PATH = FORMATTED_LABELS_DIR / PAIRS_FILENAME

# Stage 1: read the raw CSV and clean it up into a proper table of numbers.
def clean_labels(path):
    # latin1 encoding is needed because the lab software's export is not valid UTF-8.
    labels = pd.read_csv(path, encoding="latin1")

    # The raw file has 3 description rows before the real data starts. Drop them.
    labels = labels.iloc[3:].copy()

    # Keep only the columns we need, throwing away everything else in the export.
    labels = labels[
        [
            "LABEL",
            "ID",
            "TRACK_ID",
            "POSITION_X",
            "POSITION_Y",
            "POSITION_Z",
            "POSITION_T",
            "FRAME",
        ]
    ]

    # These columns should hold numbers. After reading a CSV, everything is text, even things that look like numbers.
    numeric_columns = [
        "ID",
        "TRACK_ID",
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "POSITION_T",
        "FRAME",
    ]

    for column in numeric_columns:
        # errors="coerce" means: if a value cannot be turned into a number, put a blank (NaN) there instead of crashing the whole script.
        labels[column] = pd.to_numeric(labels[column], errors="coerce")

    # Drop any row where one of the important values could not be read as a number.
    labels = labels.dropna(
        subset=["ID", "TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME"]
    )

    # These three are counting numbers, so make them whole numbers.
    labels["ID"] = labels["ID"].astype(int)
    labels["TRACK_ID"] = labels["TRACK_ID"].astype(int)
    labels["FRAME"] = labels["FRAME"].astype(int)

    return labels


# Stage 2: keep only the tracks that were labeled in BOTH frames.
def keep_tracks_in_frame_pair(labels, query_frame, target_frame):
    query_labels = labels[labels["FRAME"] == query_frame].copy()
    target_labels = labels[labels["FRAME"] == target_frame].copy()

    # The track IDs that appear in each frame.
    query_track_ids = set(query_labels["TRACK_ID"].unique())
    target_track_ids = set(target_labels["TRACK_ID"].unique())

    # The tracks in both frames
    common_track_ids = sorted(query_track_ids.intersection(target_track_ids))

    # Keep rows that are in one of the two frames AND belong to a usable track.
    common_tracks = labels[
        (labels["FRAME"].isin([query_frame, target_frame]))
        & (labels["TRACK_ID"].isin(common_track_ids))
    ].copy()

    common_tracks = common_tracks.sort_values(by=["TRACK_ID", "FRAME"]).reset_index(drop=True)

    return common_tracks


# Stage 3: reshape into one row per track, with the query and target side by side.
def build_query_target_pairs(labels, query_frame, target_frame):
    query_rows = labels[labels["FRAME"] == query_frame].copy()
    target_rows = labels[labels["FRAME"] == target_frame].copy()

    # Join each track's query-frame row to its target-frame row, so both end up in one row.
    pairs = query_rows.merge(
        target_rows,
        on="TRACK_ID",
        suffixes=("_QUERY", "_TARGET"),
        validate="one_to_one",
    )

    # Keep only the columns the experiments need.
    pairs = pairs[
        [
            "TRACK_ID",
            "FRAME_QUERY",
            "FRAME_TARGET",
            "POSITION_X_QUERY",
            "POSITION_Y_QUERY",
            "POSITION_Z_QUERY",
            "POSITION_X_TARGET",
            "POSITION_Y_TARGET",
            "POSITION_Z_TARGET",
        ]
    ].copy()

    # Rename to clear names that state the units. 
    pairs = pairs.rename(
        columns={
            "TRACK_ID": "track_id",
            "FRAME_QUERY": "query_frame",
            "FRAME_TARGET": "target_frame",
            "POSITION_X_QUERY": "query_x_um",
            "POSITION_Y_QUERY": "query_y_um",
            "POSITION_Z_QUERY": "query_z_slice",
            "POSITION_X_TARGET": "target_x_um",
            "POSITION_Y_TARGET": "target_y_um",
            "POSITION_Z_TARGET": "target_z_slice",
        }
    )

    pairs = pairs.sort_values(by="track_id").reset_index(drop=True)

    return pairs


def main():
    # Make the output folder if it does not exist yet.
    FORMATTED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_labels = clean_labels(RAW_LABELS_PATH)

    frame_pair_tracks = keep_tracks_in_frame_pair(
        cleaned_labels, query_frame=QUERY_FRAME, target_frame=TARGET_FRAME
    )
    # frame_pair_tracks.to_csv(FRAME_PAIR_PATH, index=False)

    pairs = build_query_target_pairs(
        frame_pair_tracks, query_frame=QUERY_FRAME, target_frame=TARGET_FRAME
    )
    pairs.to_csv(PAIRS_OUTPUT_PATH, index=False)

    return PAIRS_OUTPUT_PATH

if __name__ == "__main__":
    output_path = main()
    print(f"Pairs file written: {output_path}")
