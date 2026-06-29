import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from GUI.utils import (
    DINOv3Encoder,
    compute_embedding,
    crop_around_point,
    get_points_stack2,
)

# This file is:
# Neurofind Project/experiments/compute_full_embeddings_hpc.py
#
# Therefore parents[1] is:
# Neurofind Project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_STACK_NAME = "A5"

TARGET_STACK_PATH = (
    PROJECT_ROOT
    / "data"
    / "time_data_labeled"
    / "33648_A5_TS_dftcorr.tif"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "spine_embedder_ssl_dinov3_128_7_5pth.sec"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# None means process all candidate points from all A5 volumes.
#
# For a small test, use an integer such as 1000.
# This limit is applied to the total number of candidates across the
# entire TIFF, not separately to every volume.
MAX_CANDIDATES = None

CROP_SIZE_Z = 7
CROP_SIZE_Y = 128
CROP_SIZE_X = 128

EMB_OUTPUT_PATH = (
    OUTPUT_DIR
    / f"embeddings_dinov3_{TARGET_STACK_NAME}_128_7.npy"
)

POINTS_OUTPUT_PATH = (
    OUTPUT_DIR
    / f"candidate_points_{TARGET_STACK_NAME}.npy"
)

VOLUME_INDICES_OUTPUT_PATH = (
    OUTPUT_DIR
    / f"candidate_volume_indices_{TARGET_STACK_NAME}.npy"
)

def main():
    if not TARGET_STACK_PATH.is_file():
        raise FileNotFoundError(
            f"A5 TIFF stack was not found:\n{TARGET_STACK_PATH}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found:\n{MODEL_PATH}"
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Target stack: {TARGET_STACK_PATH}")
    print(f"Model checkpoint: {MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    target_stack_raw = tifffile.imread(TARGET_STACK_PATH)

    print(f"Raw A5 TIFF shape: {target_stack_raw.shape}")
    print(f"Raw A5 TIFF dtype: {target_stack_raw.dtype}")

    # Expected cases:
    #
    # 3D: (z, y, x)
    #     One 3D volume.
    #
    # 4D: (time_or_volume, z, y, x)
    #     Multiple 3D volumes.
    #
    # Convert a single 3D stack to a one-volume 4D representation so that
    # the remaining code can use the same processing loop.
    if target_stack_raw.ndim == 3:
        target_volumes = target_stack_raw[np.newaxis, ...]
    elif target_stack_raw.ndim == 4:
        target_volumes = target_stack_raw
    else:
        raise ValueError(
            "Expected the A5 TIFF to be either 3D or 4D, but received "
            f"shape {target_stack_raw.shape}."
        )

    num_volumes = target_volumes.shape[0]

    print(f"Number of A5 volumes to process: {num_volumes}")
    print(f"Individual volume shape: {target_volumes.shape[1:]}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cpu":
        print(
            "WARNING: CUDA is unavailable. Processing the complete A5 stack "
            "on the login node or CPU may take many hours."
        )

    model = DINOv3Encoder(
        embedding_dim=128,
        freeze_backbone=False,
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    # Support both a plain state dictionary and a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("DINOv3 model loaded successfully.")

    all_embeddings = []
    all_candidate_points = []
    all_volume_indices = []

    total_candidates_processed = 0

    with torch.inference_mode():
        for volume_index in range(num_volumes):
            # if volume_index != 1:
            #     continue # to get the candidates of a particular frame only
            target_volume = target_volumes[volume_index]

            print()
            print(
                f"Processing A5 volume "
                f"{volume_index + 1}/{num_volumes}"
            )

            candidate_points = np.asarray(
                get_points_stack2(target_volume)
            )

            if candidate_points.ndim != 2:
                raise ValueError(
                    f"Unexpected candidate-point shape for volume "
                    f"{volume_index}: {candidate_points.shape}"
                )

            print(
                f"Candidate points in volume {volume_index}: "
                f"{len(candidate_points)}"
            )

            # Apply an optional global test limit.
            if MAX_CANDIDATES is not None:
                remaining = MAX_CANDIDATES - total_candidates_processed

                if remaining <= 0:
                    break

                candidate_points = candidate_points[:remaining]

            volume_embeddings = []

            progress_description = (
                f"A5 volume {volume_index + 1}/{num_volumes}"
            )

            for point in tqdm(
                candidate_points,
                desc=progress_description,
            ):
                crop = crop_around_point(
                    target_volume,
                    point,
                    size_z=CROP_SIZE_Z,
                    size_y=CROP_SIZE_Y,
                    size_x=CROP_SIZE_X,
                )

                embedding = compute_embedding(
                    crop,
                    model,
                    device,
                )

                # Convert one embedding to a one-dimensional NumPy array.
                embedding_array = (
                    embedding
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                volume_embeddings.append(embedding_array)

            if volume_embeddings:
                volume_embeddings = np.stack(
                    volume_embeddings,
                    axis=0,
                )

                all_embeddings.append(volume_embeddings)
                all_candidate_points.append(candidate_points)

                all_volume_indices.append(
                    np.full(
                        shape=len(candidate_points),
                        fill_value=volume_index,
                        dtype=np.int32,
                    )
                )

                total_candidates_processed += len(candidate_points)

            print(
                f"Total candidates processed so far: "
                f"{total_candidates_processed}"
            )

    if not all_embeddings:
        raise RuntimeError(
            "No embeddings were generated. Check candidate-point generation "
            "and the input stack."
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    candidate_points = np.concatenate(
        all_candidate_points,
        axis=0,
    )

    candidate_volume_indices = np.concatenate(
        all_volume_indices,
        axis=0,
    )

    if not (
        len(embeddings)
        == len(candidate_points)
        == len(candidate_volume_indices)
    ):
        raise RuntimeError(
            "The number of embeddings, candidate points, and volume indices "
            "does not match."
        )

    print()
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Final candidate-points shape: {candidate_points.shape}")
    print(
        "Final volume-index array shape: "
        f"{candidate_volume_indices.shape}"
    )

    np.save(
        EMB_OUTPUT_PATH,
        embeddings,
    )

    np.save(
        POINTS_OUTPUT_PATH,
        candidate_points,
    )

    np.save(
        VOLUME_INDICES_OUTPUT_PATH,
        candidate_volume_indices,
    )

    print()
    print("Processing completed successfully.")
    print(f"Embeddings saved to:\n{EMB_OUTPUT_PATH}")
    print(f"Candidate points saved to:\n{POINTS_OUTPUT_PATH}")
    print(
        "Candidate volume indices saved to:\n"
        f"{VOLUME_INDICES_OUTPUT_PATH}"
    )

if __name__ == "__main__":
    main()