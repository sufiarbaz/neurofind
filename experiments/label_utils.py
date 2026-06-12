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