#  ---------- The script below extracts the structural layout and metadata from multi-dimensional TIFF microscopic image file ----------
# from pathlib import Path # helps build file paths
# import tifffile # read TIFF image data and TIFF metadata
# BASE_DIR = Path(__file__).resolve().parents[1] # main project folder
# DATA_DIR = BASE_DIR / "data" / "time_data_labeled" # path to .tif files
# A5_TIFF_PATH = DATA_DIR / "33648_A5_TS_dftcorr.tif" # points to A5 TIFF file
# def inspect_tiff(path): # starts a resuable method
#     print("\n" + "=" * 80) # prints a separator
#     print("File:")
#     print(path) # prints file path
#     with tifffile.TiffFile(path) as tif: # opens the TIFF file safely, with makes sure the files closes automatically afterward
#         print("\nNumber of pages:") # TIFF files can contain many pages, a page is often one 2D image slice
#         print(len(tif.pages))
#         series = tif.series[0]
#         print("\nSeries shape:") # tells the image dimensions
#         print(series.shape)
#         print("\nSeries axes:") # tells what each dimension means
#         print(series.axes)
#         print("\nImageJ metadata:")
#         print(tif.imagej_metadata) # some microscope TIFFs store calibration here, we are looking for values like spacing, unit, finterval, pixel size information
#         print("\nOME metadata exists?")
#         print(tif.ome_metadata is not None) # some TIFFs store metadata in OME-XML format. This line only checks whether such metadata exists
#         if tif.ome_metadata is not None:
#             print("\nFirst 1000 characters of OME metadata:")
#             print(tif.ome_metadata[:1000]) # if OME metadata exists, print only the first 1000 characters
#         first_page = tif.pages[0] # inspect the first 2D page of the TIFF
#         print("\nImportant TIFF tags from first page:") # this is just a label for the output
#         useful_tags = [
#             "ImageWidth",
#             "ImageLength",
#             "XResolution",
#             "YResolution",
#             "ResolutionUnit",
#             "ImageDescription",
#         ] # these are the metadata tags that may contain image size or pixel calibration
#         for tag_name in useful_tags:
#             if tag_name in first_page.tags:
#                 print(f"\n{tag_name}")
#                 print(first_page.tags[tag_name].value) # loop over useful tag names, if the tag exists, print its value
# inspect_tiff(A5_TIFF_PATH)

#  ---------- The script below loads, cleans and formats data of the  human-labelled-data. ----------
import pandas as pd # handle tabular data structures
# Helper function: load and clean label CSV
def load_clean_labels(path):
    # Read CSV file.
    # latin1 is used because the file is not valid UTF-8.
    df = pd.read_csv(path, encoding="latin1")
    # Remove first three non-data rows.
    # These rows contain descriptions / empty values, not actual label points.
    df = df.iloc[3:].copy()
    # Keep only columns needed for matching and coordinates.
    df = df[
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
    # Columns that should contain numbers.
    numeric_columns = [
        "ID",
        "TRACK_ID",
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "POSITION_T",
        "FRAME",
    ]
    # Convert numeric columns from text to numbers.
    # Invalid values become NaN.
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Remove rows where important values are missing.
    df = df.dropna(
        subset=[
            "ID",
            "TRACK_ID",
            "POSITION_X",
            "POSITION_Y",
            "POSITION_Z",
            "FRAME",
        ]
    )
    # Convert integer-like columns to int.
    df["ID"] = df["ID"].astype(int)
    df["TRACK_ID"] = df["TRACK_ID"].astype(int)
    df["FRAME"] = df["FRAME"].astype(int)
    # Return cleaned dataframe.
    return df

# ---------- The below code filter the human-labelled-data received by the lab. ----------
# ---------- It keeps only stack TRACK_IDs that appear in all six time frames. ----------
# from pathlib import Path
# from label_utils import load_clean_labels
# INPUT_PATH = Path("C:/Users/z0051rra/Downloads/Neurofind Project/data/time_data_labeled/33648_A5_TS_dftcorr_02.csv")
# OUTPUT_PATH = Path("C:/Users/z0051rra/Downloads/Neurofind Project/data/time_data_labeled/formatted_labels/A5_tracks_present_all_6_frames.csv")
# a5_labels = load_clean_labels(INPUT_PATH)
# frame_counts = (
#     a5_labels.groupby("TRACK_ID")["FRAME"]
#     .nunique()
# )
# complete_track_ids = frame_counts[
#     frame_counts == 6
# ].index
# complete_tracks = a5_labels[
#     a5_labels["TRACK_ID"].isin(complete_track_ids)
# ].copy()
# complete_tracks = complete_tracks.sort_values(
#     by=["TRACK_ID", "FRAME"]
# ).reset_index(drop=True)
# complete_tracks.to_csv(OUTPUT_PATH, index=False)
# print("Saved complete A5 tracks to:")
# print(OUTPUT_PATH)
# print("\nNumber of complete TRACK_IDs:")
# print(len(complete_track_ids))
# print("\nNumber of rows:")
# print(len(complete_tracks))
# print("\nFrames present:")
# print(sorted(complete_tracks["FRAME"].unique()))
# print("\nRows per frame:")
# print(complete_tracks.groupby("FRAME").size())
# print("\nRows per TRACK_ID:")
# print(complete_tracks.groupby("TRACK_ID").size().head())
# print("\nFirst rows:")
# print(complete_tracks.head(12))

# ---------- The below script gives the TRACK_IDs that are available in both frame 0 and frame 1 ----------
# from pathlib import Path
# # from label_utils import load_clean_labels
# INPUT_PATH = Path("C:/Users/z0051rra/Downloads/Neurofind Project/data/time_data_labeled/33648_A5_TS_dftcorr_02.csv")
# OUTPUT_PATH = Path("C:/Users/z0051rra/Downloads/Neurofind Project/data/time_data_labeled/formatted_labels/A5_tracks_present_frame0_frame1.csv")
# OUTPUT_PATH.parent.mkdir(
#     parents=True,
#     exist_ok=True,
# )
# # Load and clean the complete A5 human-label CSV.
# a5_labels = load_clean_labels(INPUT_PATH)
# # Keep only rows from frame 0 and frame 1.
# frame0_labels = a5_labels[
#     a5_labels["FRAME"] == 0
# ].copy()
# frame1_labels = a5_labels[
#     a5_labels["FRAME"] == 1
# ].copy()
# # Extract unique TRACK_ID values from each frame.
# frame0_track_ids = set(
#     frame0_labels["TRACK_ID"].unique()
# )
# frame1_track_ids = set(
#     frame1_labels["TRACK_ID"].unique()
# )
# # Find TRACK_ID values present in both frames.
# common_track_ids = sorted(
#     frame0_track_ids.intersection(
#         frame1_track_ids
#     )
# )
# # Keep frame-0 and frame-1 rows belonging to the shared tracks.
# common_tracks = a5_labels[
#     (a5_labels["FRAME"].isin([0, 1]))
#     &
#     (a5_labels["TRACK_ID"].isin(common_track_ids))
# ].copy()
# # Sort each track by frame.
# common_tracks = common_tracks.sort_values(
#     by=["TRACK_ID", "FRAME"]
# ).reset_index(drop=True)
# # Save the filtered result.
# common_tracks.to_csv(
#     OUTPUT_PATH,
#     index=False,
# )
# # Validation output.
# print("Saved A5 tracks common to frame 0 and frame 1:")
# print(OUTPUT_PATH)
# print("\nNumber of TRACK_IDs in frame 0:")
# print(len(frame0_track_ids))
# print("\nNumber of TRACK_IDs in frame 1:")
# print(len(frame1_track_ids))
# print("\nNumber of common TRACK_IDs:")
# print(len(common_track_ids))
# print("\nCommon TRACK_IDs:")
# print(common_track_ids)
# print("\nNumber of output rows:")
# print(len(common_tracks))
# print("\nRows per frame:")
# print(
#     common_tracks.groupby("FRAME").size()
# )
# print("\nRows per TRACK_ID:")
# print(
#     common_tracks.groupby("TRACK_ID").size().value_counts()
# )
# print("\nFirst rows:")
# print(
#     common_tracks.head(12)
# )

# ---------- The script below takes the stable TRACK_IDs and place the frame-0 and frame-1 position of each track into the same row. ----------
from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_PATH = "C:/Users/z0051rra/Downloads/Neurofind Project/data/time_data_labeled/formatted_human_labels/A5_tracks_present_frame0_frame1.csv"
OUTPUT_PATH = "C:/Users/z0051rra/Downloads/Neurofind Project/data/time_data_labeled/formatted_human_labels/A5_frame0_to_frame1_pairs.csv"
# Load the filtered A5 labels containing only tracks
# that are present in all six frames.
complete_tracks = pd.read_csv(INPUT_PATH)
# Frame 0 provides the query labels.
frame0 = complete_tracks[
    complete_tracks["FRAME"] == 0
].copy()
# Frame 1 provides the corresponding human target labels.
frame1 = complete_tracks[
    complete_tracks["FRAME"] == 1
].copy()
print("Frame 0 rows:", len(frame0))
print("Frame 1 rows:", len(frame1))
# Join frame-0 and frame-1 observations belonging to the same spine.
pairs = frame0.merge(
    frame1,
    on="TRACK_ID",
    suffixes=("_QUERY", "_TARGET"),
    validate="one_to_one",
)
# Keep only the columns needed for the experiment.
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
# Give the columns explicit query/target names and preserve the units.
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
# Sort the output for easier inspection.
pairs = pairs.sort_values(
    by="track_id"
).reset_index(drop=True)
# Save the explicit frame-0 to frame-1 pair table.
pairs.to_csv(OUTPUT_PATH, index=False)
print("\nSaved frame-pair file to:")
print(OUTPUT_PATH)
print("\nNumber of query-target pairs:")
print(len(pairs))
print("\nQuery frames:")
print(pairs["query_frame"].unique())
print("\nTarget frames:")
print(pairs["target_frame"].unique())
print("\nOutput columns:")
print(pairs.columns.tolist())
print("\nFirst rows:")
print(pairs.head())