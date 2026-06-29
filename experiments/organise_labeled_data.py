# Turns the raw, human-labeled CSV into the query/target pairs file used by the Hypothesis 1 experiment.

from pathlib import Path # build file paths relative to the project folder
import pandas as pd # read, clean and reshape the label CSV files as tables (dataframes)

BASE_DIR = Path(__file__).resolve().parents[1] # project root folder, found from this script's own location
RAW_LABELS_PATH = BASE_DIR / "data" / "time_data_labeled" / "33648_A5_TS_dftcorr_02.csv"
FORMATTED_LABELS_DIR = BASE_DIR / "data" / "formatted_human_labels"
FRAME_PAIR_PATH = FORMATTED_LABELS_DIR / "A5_tracks_present_frame0_to_frame1.csv"
PAIRS_OUTPUT_PATH = FORMATTED_LABELS_DIR / "A5_frame0_to_frame1_pairs.csv"

QUERY_FRAME = 0
TARGET_FRAME = 1

# Clean the raw label CSV, load it and return a clean, correctly-typed dataframe.
def clean_labels(path):
    labels = pd.read_csv(path, encoding="latin1") # read the raw CSV; latin1 encoding is required because the file isn't a valid UTF-8
    labels = labels.iloc[3:].copy() # skip the first three non-data header rows
    labels = labels [
        [
            "LABEL",
            "ID",
            "TRACK_ID",
            "POSITION_X",
            "POSITION_Y",
            "POSITION_Z",
            "POSITION_T",
            "FRAME"
        ]
    ] # keep only the columns we need; the inner [....] is the list of column names, the outer [....] is pandas' "select these columns" syntax
    
    numeric_columns = [
        "ID",
        "TRACK_ID",
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "POSITION_T",
        "FRAME"
    ] # list of columns that hold numbers, so the next step knows which ones to convert from text to actual numbers.

    for column in numeric_columns:
        labels[column] = pd.to_numeric(labels[column], errors="coerce") # convert the columns to numbers

    labels = labels.dropna(
        subset = [
            "ID",
            "TRACK_ID",
            "POSITION_X",
            "POSITION_Y",
            "POSITION_Z",
            "FRAME"
        ]
    ) # drop any rows missing any of the column values

    # convert the columns into whole numbers (integers)
    labels["ID"] = labels["ID"].astype(int) 
    labels["TRACK_ID"] = labels["TRACK_ID"].astype(int)
    labels["FRAME"] = labels["FRAME"].astype(int)

    return labels

# Keep only tracks available in both the query and target frame
def keep_tracks_in_frame_pair(labels, query_frame, target_frame):
    query_labels = labels[labels["FRAME"] == query_frame].copy() # keep only the rows belonging to the query (frame 0)
    target_labels = labels[labels["FRAME"] == target_frame].copy() # keep only the rows belonging to the target (frame 1)

    query_track_ids = set(query_labels["TRACK_ID"].unique()) # collect the unique track IDs that appear in the query frame
    target_track_ids = set(target_labels["TRACK_ID"].unique()) # collect the unique track IDs that appear in the target frame

    common_track_ids = sorted(query_track_ids.intersection(target_track_ids)) # common track IDs between both

    common_tracks = labels[
        (labels["FRAME"].isin([query_frame, target_frame]))
        & (labels["TRACK_ID"].isin(common_track_ids))
    ].copy() # keep rows that are in either query or target frame AND belong to a track present in both frames

    common_tracks = common_tracks.sort_values(
        by=["TRACK_ID", "FRAME"]
    ).reset_index(drop=True) # sort data by track id and frame, then reset row numbers

    return common_tracks

# reshape into one row per track, query and target side by side
def build_query_target_pairs(labels, query_frame, target_frame):
    query_rows = labels[labels["FRAME"] == query_frame].copy() # take the query frame rows
    target_rows = labels[labels["FRAME"] == target_frame].copy() # take the target frame rows

    pairs = query_rows.merge(
        target_rows,
        on="TRACK_ID",
        suffixes=("_QUERY", "_TARGET"),
        validate="one_to_one"
    ) # join each track's query-frame row with its target-frame row into one row, tagging overlapping columns with _QUERY/_TARGET, and verifying each track appears exactly once on both sides

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
            "POSITION_Z_TARGET"
        ]
    ].copy() # keep only these columns, dropping all others

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
    ) # rename the columns

    pairs = pairs.sort_values(by="track_id").reset_index(drop=True) # sort rows bt track ID

    return pairs


# ---------------------------------------------------------------------
# Pipeline entry point: runs all 3 stages in order
# ---------------------------------------------------------------------

def main():
    FORMATTED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Stage 1: cleaning raw label CSV...")
    cleaned_labels = clean_labels(RAW_LABELS_PATH)
    print(f"  {len(cleaned_labels)} rows after cleaning.")

    print(f"\nStage 2: keeping tracks present in frame {QUERY_FRAME} "
          f"and frame {TARGET_FRAME}...")
    frame_pair_tracks = keep_tracks_in_frame_pair(
        cleaned_labels, query_frame=QUERY_FRAME, target_frame=TARGET_FRAME
    )
    frame_pair_tracks.to_csv(FRAME_PAIR_PATH, index=False)
    print(f"  {frame_pair_tracks['TRACK_ID'].nunique()} tracks remain.")
    print(f"  Saved to: {FRAME_PAIR_PATH}")

    print("\nStage 3: building query/target pairs table...")
    pairs = build_query_target_pairs(
        frame_pair_tracks, query_frame=QUERY_FRAME, target_frame=TARGET_FRAME
    )
    pairs.to_csv(PAIRS_OUTPUT_PATH, index=False)
    print(f"  {len(pairs)} query-target pairs.")
    print(f"  Saved to: {PAIRS_OUTPUT_PATH}")

    print("\nDone. Output columns:", pairs.columns.tolist())


if __name__ == "__main__":
    main()

    











