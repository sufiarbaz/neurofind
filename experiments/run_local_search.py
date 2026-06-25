from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

BASE_DIR = Path("C:/Users/z0051rra/Downloads/Neurofind Project")

GUI_DIR = BASE_DIR / "GUI"
sys.path.append(str(GUI_DIR))

from utils import DINOv3Encoder, crop_around_point, compute_embedding


DATA_DIR = BASE_DIR / "data" / "time_data_labeled"
OUTPUT_DIR = BASE_DIR / "outputs"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# A5 is now both the query stack and target stack.
A5_STACK_PATH = "C:/Users/z0051rra/Downloads/Neurofind Project/data/time_data_labeled/33648_A5_TS_dftcorr.tif"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "spine_embedder_ssl_dinov3_128_7_5pth.sec"
)

# Use your full A5 candidate files here.
TARGET_CANDIDATES_PATH = (
    EMBEDDINGS_DIR
    / "candidate_points_A5.npy"
)

TARGET_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR
    / "embeddings_dinov3_A5_128_7.npy"
)

TARGET_VOLUME_INDICES_PATH = (
    EMBEDDINGS_DIR
    / "candidate_volume_indices_A5.npy"
)

# This is the pair table that contains frame-0 query labels
# and frame-1 human target labels.
LABEL_PAIRS_PATH = (
    DATA_DIR / "formatted_human_labels" / "A5_frame0_to_frame1_pairs.csv"
)

OUTPUT_RESULTS_PATH = (
    OUTPUT_DIR / "A5_frame0_to_frame1_local_search_results.csv"
)

# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

# Human-labelled query X and Y positions are in micrometers.
# TIFF and candidate-point coordinates are in pixels.
A5_PIXELS_PER_UM_X = 7.246376
A5_PIXELS_PER_UM_Y = 7.246376

# Local-search radii are defined in micrometers.
SEARCH_RADII_UM = [5, 10, 15, 20, 25, 30]

# Use 1 while debugging.
# Use None to process all 28 stable tracks.
NUMBER_OF_PAIRS_TO_RUN = None

# Adds one empty CSV row between consecutive tracks.
ADD_BLANK_ROW_BETWEEN_TRACKS = True


# ---------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "strategy",
    "track_id",
    "query_frame",
    "target_frame",
    "radius_um",
    "number_of_candidates",
    "query_point_um_z,y,x",
    "predicted_point_um_z,y,x",
    "runtime_seconds",
    "same_match_as_full_search",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_time_stack(path):
    """
    Load the complete A5 TIFF while preserving the time dimension.

    Expected shape:
        (T, Z, Y, X)
    """

    stack = tifffile.imread(path)

    if stack.ndim != 4:
        raise ValueError(
            "Expected a four-dimensional A5 stack with shape "
            f"(T, Z, Y, X), but received shape {stack.shape}."
        )

    return stack


def point_um_to_pixel(x_um, y_um, z_slice):
    """
    Convert a human-labelled query coordinate into TIFF coordinates.

    X and Y:
        micrometers -> pixels

    Z:
        remains a slice index

    Returns:
        (z, y, x)
    """

    x_px = int(round(x_um * A5_PIXELS_PER_UM_X))
    y_px = int(round(y_um * A5_PIXELS_PER_UM_Y))
    z_px = int(round(z_slice))

    return z_px, y_px, x_px


def point_pixel_to_um(point_px):
    """
    Convert an internal candidate coordinate into physical XY units.

    Input:
        (z_slice, y_pixel, x_pixel)

    Output:
        (z_slice, y_micrometers, x_micrometers)

    Z remains a slice index because physical Z spacing is not part
    of Hypothesis 1.
    """

    z_slice = int(point_px[0])
    y_um = float(point_px[1] / A5_PIXELS_PER_UM_Y)
    x_um = float(point_px[2] / A5_PIXELS_PER_UM_X)

    return z_slice, y_um, x_um


def format_point(point):
    """
    Store one complete coordinate tuple in one CSV column.
    """

    if point is None:
        return None

    return f"({point[0]}, {point[1]}, {point[2]})"


def filter_candidates_by_radius_um(
    candidate_points,
    center_point_px,
    radius_um,
):
    """
    Keep target candidates inside an XY radius measured in micrometers.

    candidate_points:
        NumPy array with shape (N, 3), ordered as (z, y, x).

    center_point_px:
        Query coordinate ordered as (z, y, x).

    Z is deliberately ignored when defining the local XY area.
    """

    dy_px = candidate_points[:, 1] - center_point_px[1]
    dx_px = candidate_points[:, 2] - center_point_px[2]

    dy_um = dy_px / A5_PIXELS_PER_UM_Y
    dx_um = dx_px / A5_PIXELS_PER_UM_X

    distances_um = np.sqrt(
        dy_um**2 + dx_um**2
    )

    mask = distances_um <= radius_um
    local_candidates = candidate_points[mask]

    return local_candidates, mask


def find_best_match(
    query_embedding,
    target_embeddings,
    candidate_points,
):
    """
    Compare one query embedding with the supplied target embeddings.

    Returns:
        best candidate coordinate as a NumPy array: (z, y, x)
    """

    if len(candidate_points) == 0:
        raise ValueError(
            "find_best_match received zero candidate points."
        )

    query_embedding_batch = query_embedding.unsqueeze(0)

    similarities = F.cosine_similarity(
        query_embedding_batch,
        target_embeddings,
        dim=1,
    )

    best_index = int(
        torch.argmax(similarities).item()
    )

    return np.asarray(candidate_points[best_index])


def empty_result_row():
    """
    Create one completely empty row for visual separation in the CSV.
    """

    return {
        column: None
        for column in OUTPUT_COLUMNS
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    print(
        "Starting A5 frame 0 to frame 1 "
        "full-search versus local-search experiment..."
    )

    results = []

    # -------------------------------------------------------------
    # Load experiment input table
    # -------------------------------------------------------------

    label_pairs = pd.read_csv(
        LABEL_PAIRS_PATH
    )

    print(
        "Number of available query rows:",
        len(label_pairs),
    )

    # Target-coordinate columns are deliberately not required.
    required_columns = {
        "track_id",
        "query_frame",
        "target_frame",
        "query_x_um",
        "query_y_um",
        "query_z_slice",
    }

    missing_columns = (
        required_columns
        - set(label_pairs.columns)
    )

    if missing_columns:
        raise ValueError(
            "The experiment-input CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if NUMBER_OF_PAIRS_TO_RUN is not None:
        label_pairs = label_pairs.iloc[
            :NUMBER_OF_PAIRS_TO_RUN
        ].copy()

    label_pairs = label_pairs.reset_index(
        drop=True
    )

    print(
        "Number of queries selected:",
        len(label_pairs),
    )

    # -------------------------------------------------------------
    # Load complete A5 candidate arrays
    # -------------------------------------------------------------

    all_candidate_points = np.load(
        TARGET_CANDIDATES_PATH
    )

    all_target_embeddings = np.load(
        TARGET_EMBEDDINGS_PATH
    )

    all_volume_indices = np.load(
        TARGET_VOLUME_INDICES_PATH
    )

    print(
        "All candidate points shape:",
        all_candidate_points.shape,
    )

    print(
        "All target embeddings shape:",
        all_target_embeddings.shape,
    )

    print(
        "All volume indices shape:",
        all_volume_indices.shape,
    )

    if not (
        len(all_candidate_points)
        == len(all_target_embeddings)
        == len(all_volume_indices)
    ):
        raise ValueError(
            "Candidate points, embeddings, and volume indices "
            "must have matching lengths."
        )

    # -------------------------------------------------------------
    # Load complete A5 TIFF
    # -------------------------------------------------------------

    a5_stack = load_time_stack(
        A5_STACK_PATH
    )

    print(
        "A5 stack shape:",
        a5_stack.shape,
    )

    # -------------------------------------------------------------
    # Load DINOv3 model
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    model = DINOv3Encoder()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)
    model.eval()

    # -------------------------------------------------------------
    # Process selected query tracks
    # -------------------------------------------------------------

    for pair_number, row in label_pairs.iterrows():
        track_id = int(
            row["track_id"]
        )

        query_frame = int(
            row["query_frame"]
        )

        target_frame = int(
            row["target_frame"]
        )

        query_x_um = float(
            row["query_x_um"]
        )

        query_y_um = float(
            row["query_y_um"]
        )

        query_z_slice = int(
            round(row["query_z_slice"])
        )

        query_point_um = (
            query_z_slice,
            query_y_um,
            query_x_um,
        )

        # Pixel coordinates are required internally for TIFF cropping
        # and candidate-radius filtering.
        query_point_px = point_um_to_pixel(
            x_um=query_x_um,
            y_um=query_y_um,
            z_slice=query_z_slice,
        )

        print("\n" + "=" * 70)
        print(
            f"Query {pair_number + 1}: "
            f"TRACK_ID {track_id}"
        )
        print("=" * 70)

        print(
            "Query frame:",
            query_frame,
        )

        print(
            "Target search frame:",
            target_frame,
        )

        print(
            "Query point "
            "(z-slice, y-µm, x-µm):",
            query_point_um,
        )

        print(
            "Internal query coordinate "
            "(z, y, x pixels):",
            query_point_px,
        )

        # ---------------------------------------------------------
        # Validate frames
        # ---------------------------------------------------------

        if not 0 <= query_frame < a5_stack.shape[0]:
            raise ValueError(
                f"Invalid query frame {query_frame}. "
                f"Valid range is 0 to "
                f"{a5_stack.shape[0] - 1}."
            )

        if not 0 <= target_frame < a5_stack.shape[0]:
            raise ValueError(
                f"Invalid target frame {target_frame}. "
                f"Valid range is 0 to "
                f"{a5_stack.shape[0] - 1}."
            )

        query_volume = a5_stack[
            query_frame
        ]

        z, y, x = query_point_px

        if not 0 <= z < query_volume.shape[0]:
            raise ValueError(
                f"Invalid query z={z}. "
                f"Valid range is 0 to "
                f"{query_volume.shape[0] - 1}."
            )

        if not 0 <= y < query_volume.shape[1]:
            raise ValueError(
                f"Invalid query y={y}. "
                f"Valid range is 0 to "
                f"{query_volume.shape[1] - 1}."
            )

        if not 0 <= x < query_volume.shape[2]:
            raise ValueError(
                f"Invalid query x={x}. "
                f"Valid range is 0 to "
                f"{query_volume.shape[2] - 1}."
            )

        # ---------------------------------------------------------
        # Select candidates belonging only to the target frame
        # ---------------------------------------------------------

        target_frame_mask = (
            all_volume_indices == target_frame
        )

        candidate_points = (
            all_candidate_points[
                target_frame_mask
            ]
        )

        target_embeddings_numpy = (
            all_target_embeddings[
                target_frame_mask
            ]
        )

        print(
            "Target-frame candidate count:",
            len(candidate_points),
        )

        if len(candidate_points) == 0:
            raise ValueError(
                f"No candidates found for "
                f"target frame {target_frame}."
            )

        target_embeddings = (
            torch.from_numpy(
                target_embeddings_numpy
            )
            .float()
            .cpu()
        )

        # ---------------------------------------------------------
        # Compute query embedding
        # ---------------------------------------------------------

        crop = crop_around_point(
            query_volume,
            query_point_px,
            size_z=7,
            size_y=128,
            size_x=128,
        )

        with torch.no_grad():
            query_embedding = compute_embedding(
                crop,
                model,
                device,
            )

        query_embedding = (
            query_embedding
            .float()
            .cpu()
            .squeeze()
        )

        print(
            "Query embedding shape:",
            query_embedding.shape,
        )

        if query_embedding.ndim != 1:
            raise ValueError(
                "Expected one-dimensional query embedding, "
                f"but received shape {query_embedding.shape}."
            )

        if (
            query_embedding.shape[0]
            != target_embeddings.shape[1]
        ):
            raise ValueError(
                "Query and target embedding dimensions "
                "do not match: "
                f"{query_embedding.shape[0]} versus "
                f"{target_embeddings.shape[1]}."
            )

        # ---------------------------------------------------------
        # Full-search baseline
        # ---------------------------------------------------------

        print("\nFull-search baseline")

        full_start_time = time.perf_counter()

        full_best_point_px = find_best_match(
            query_embedding=query_embedding,
            target_embeddings=target_embeddings,
            candidate_points=candidate_points,
        )

        full_runtime_seconds = (
            time.perf_counter()
            - full_start_time
        )

        full_predicted_point_um = (
            point_pixel_to_um(
                full_best_point_px
            )
        )

        print(
            "Full-search candidate count:",
            len(candidate_points),
        )

        print(
            "Full-search predicted point "
            "(z-slice, y-µm, x-µm):",
            full_predicted_point_um,
        )

        print(
            "Full-search runtime:",
            f"{full_runtime_seconds:.6f} seconds",
        )

        results.append(
            {
                "strategy": "full_search",
                "track_id": track_id,
                "query_frame": query_frame,
                "target_frame": target_frame,
                "radius_um": None,
                "number_of_candidates": int(
                    len(candidate_points)
                ),
                "query_point_um_z,y,x": format_point(
                    query_point_um
                ),
                "predicted_point_um_z,y,x": format_point(
                    full_predicted_point_um
                ),
                "runtime_seconds": float(
                    full_runtime_seconds
                ),
                # The full search is the comparison baseline.
                "same_match_as_full_search": None,
            }
        )

        # ---------------------------------------------------------
        # Local searches
        # ---------------------------------------------------------

        print("\nLocal searches")

        for radius_um in SEARCH_RADII_UM:
            # Local runtime includes:
            # 1. radius-based candidate filtering
            # 2. embedding comparison
            local_start_time = time.perf_counter()

            (
                local_candidates,
                local_mask,
            ) = filter_candidates_by_radius_um(
                candidate_points=candidate_points,
                center_point_px=query_point_px,
                radius_um=radius_um,
            )

            print(
                f"\nRadius: {radius_um} µm"
            )

            print(
                "Local candidate count:",
                len(local_candidates),
            )

            if len(local_candidates) == 0:
                local_runtime_seconds = (
                    time.perf_counter()
                    - local_start_time
                )

                print(
                    "No candidates found inside "
                    "this search radius."
                )

                results.append(
                    {
                        "strategy": "local_search",
                        "track_id": track_id,
                        "query_frame": query_frame,
                        "target_frame": target_frame,
                        "radius_um": radius_um,
                        "number_of_candidates": 0,
                        "query_point_um_z,y,x": format_point(
                            query_point_um
                        ),
                        "predicted_point_um_z,y,x": None,
                        "runtime_seconds": float(
                            local_runtime_seconds
                        ),
                        "same_match_as_full_search": False,
                    }
                )

                continue

            local_embeddings = (
                target_embeddings[
                    local_mask
                ]
            )

            local_best_point_px = find_best_match(
                query_embedding=query_embedding,
                target_embeddings=local_embeddings,
                candidate_points=local_candidates,
            )

            local_runtime_seconds = (
                time.perf_counter()
                - local_start_time
            )

            local_predicted_point_um = (
                point_pixel_to_um(
                    local_best_point_px
                )
            )

            same_match_as_full = bool(
                np.array_equal(
                    local_best_point_px,
                    full_best_point_px,
                )
            )

            print(
                "Local predicted point "
                "(z-slice, y-µm, x-µm):",
                local_predicted_point_um,
            )

            print(
                "Same match as full search:",
                same_match_as_full,
            )

            print(
                "Local-search runtime:",
                f"{local_runtime_seconds:.6f} seconds",
            )

            results.append(
                {
                    "strategy": "local_search",
                    "track_id": track_id,
                    "query_frame": query_frame,
                    "target_frame": target_frame,
                    "radius_um": radius_um,
                    "number_of_candidates": int(
                        len(local_candidates)
                    ),
                    "query_point_um_z,y,x": format_point(
                        query_point_um
                    ),
                    "predicted_point_um_z,y,x": format_point(
                        local_predicted_point_um
                    ),
                    "runtime_seconds": float(
                        local_runtime_seconds
                    ),
                    "same_match_as_full_search": (
                        same_match_as_full
                    ),
                }
            )

        # Add an empty row after this track, except after the last one.
        if (
            ADD_BLANK_ROW_BETWEEN_TRACKS
            and pair_number < len(label_pairs) - 1
        ):
            results.append(
                empty_result_row()
            )

    # -------------------------------------------------------------
    # Save result table
    # -------------------------------------------------------------

    results_df = pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS,
    )

    print("\n" + "=" * 70)
    print("Summary table")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        OUTPUT_RESULTS_PATH,
        index=False,
    )

    print("\nResults saved to:")
    print(OUTPUT_RESULTS_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
