"""
compute_full_embeddings_hpc.py

This module computes embedding for every candidate point in every frame of a stack.
This is the heavy step, it is meant to be run once, on the HPC, with a GPU.

It writes four files into data/embeddings/:
    candidate_point_<STACK>.npy -- where evey candidate is
    embeddings_dinov3_<STACK>.npy -- each candidate's
    candidate_frame_numbers_<STACK>.npy -- which frame each candidate came from
    embedding_rate_<STACK>.json -- how long one candidate took to embedd

To change stack, modify only the relevant stacks below.

How to run it (from the project root):
    python -m src.base.compute_full_embeddings_hpc
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # windows workaround so numpy/torch libraries dont'clash

import json # to save the run-time of each candidate
import time # to measure how long the embedding work takes
from pathlib import Path # to build file paths that work on any machine
import sys # to read the stack name from the command line, if given

import numpy as np # for stacking the fingerprints and saving the .npy files
import tifffile # for reading the TIFF image stack
import torch # for loading the model and running it on GPU or CPU
from tqdm import tqdm # the progress bar, since this runs for a long time

from src.base.utils import(
    DINOv3Encoder,
    compute_embedding,
    crop_around_point,
    get_points_stack2,
)

STACK_NAME = "A5" # short name for the stack
STACK_FILENAME = "33648_A5_TS_dftcorr.tif" # place this raw TIFF in: data/time_data_labelled/

# If a stack name and filename are passed on the command line, use those instead.
# This lets several stacks run at once without editing this file.

if len(sys.argv) == 3:
	STACK_NAME = sys.argv[1]
	STACK_FILENAME = sys.argv[2]

# the crop around each point before embedding it
CROP_SIZE_Z = 7 
CROP_SIZE_Y = 128
CROP_SIZE_X = 128

# run setting, set a small number (e.g. 1000) for a quick test
MAX_CANDIDATES = None 

BASE_DIR = Path(__file__).resolve().parents[2]

TARGET_STACK_PATH = BASE_DIR / "data" / "time_data_labeled" / STACK_FILENAME
MODEL_PATH = BASE_DIR / "models" / "spine_embedder_ssl_dinov3_128_7_5pth.sec"

OUTPUT_DIR = BASE_DIR / "data" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINTS_OUTPUT_PATH = OUTPUT_DIR / f"candidate_points_{STACK_NAME}.npy"
EMB_OUTPUT_PATH = OUTPUT_DIR / f"embeddings_dinov3_{STACK_NAME}.npy"
FRAME_NUMBERS_OUTPUT_PATH = OUTPUT_DIR / f"candidate_frame_numbers_{STACK_NAME}.npy"
RATE_OUTPUT_PATH = OUTPUT_DIR / f"embedding_rate_{STACK_NAME}.json"

def main():
    if not TARGET_STACK_PATH.is_file():
        raise FileNotFoundError(f"TIFF stack was not found: \n{TARGET_STACK_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: \n{MODEL_PATH}")
    
    raw_stack = tifffile.imread(TARGET_STACK_PATH) # load the whole TIFF into the memory

    print(f"Raw TIFF shape: {raw_stack.shape}")
    print(f"Raw TIff dtype: {raw_stack.dtype}")

    # a single 3D volume becomes a one-frame 4D array, so the loop below can always assume 4D.
    if raw_stack.ndim == 3:
        all_frames = raw_stack[np.newaxis, ...]
    elif raw_stack.ndim == 4:
        all_frames = raw_stack
    else:
        raise ValueError(f"Expected the TIFF to be either 3D or 4D, but received shape {raw_stack.shape}.")
    
    num_frames = all_frames.shape[0] # number of frames to process

    print(f"Number of volumes to process: {num_frames}")
    print(f"Individual volume shape: {all_frames.shape[1:]}")

    # use the graphis card if there is one, otherwise fall back to the processor.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cpu":
        print("CUDA is unavailable. processing the complete stack on a CPU may take hours.")

    model = DINOv3Encoder(embedding_dim=128, freeze_backbone=False).to(device=device) # build the empty model and put it on the device

    checkpoint = torch.load(MODEL_PATH, map_location=device) # load the trained weights from disk

    # some training scripts save the raw weights, other wrap them in a dicitonary. This accepts either.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict=state_dict) # pour the tranied weights into the empty model
    model.eval()

    print("Model loaded successfully.")

    # collect the results from every frame here, then join them at the end.
    all_embeddings = []
    all_candidate_points = []
    all_frame_numbers = []

    total_candidate_processed = 0 # running count across all frames.

    # time only the embedding work (cropping + running the model)
    total_embedding_seconds = 0.0

    with torch.inference_mode():
        for frame_index in range(num_frames): # go through the frames one at a time
            one_frame = all_frames[frame_index] # pull out this frame's 3D image
            print()
            print(f"Processing volume {frame_index + 1}/{num_frames}")

            candidate_points = np.asarray(get_points_stack2(one_frame)) # find every bright pixel in this frame

            if candidate_points.ndim != 2: # it should be a list of (z, y, x) rows
                raise ValueError(
                    f"Unexpected candidate-point shape for volume {frame_index}:"
                    f"{candidate_points.shape}"
                ) # stop early if the shape is wrong.
            
            print(f"Candidate points in volume {frame_index}: {len(candidate_points)}")
            
            frame_embeddings = []
            progress_description = f"volume {frame_index + 1} / {num_frames}"

            # the timed section
            frame_start = time.perf_counter() #start the clock for this frame

            for point in tqdm(candidate_points, desc=progress_description): # every candidate in this frame
                crop = crop_around_point(
                    one_frame, point, # cut a box out of the image around this point
                    size_z=CROP_SIZE_Z, size_y=CROP_SIZE_Y, size_x=CROP_SIZE_X # the box size
                )
                embedding = compute_embedding(crop, model, device) # run the box through the model
                embedding_array = embedding.detach().cpu().numpy().reshape(-1) # turn it into a flat list of 128 numbers
                frame_embeddings.append(embedding_array)

            total_embedding_seconds += time.perf_counter() - frame_start

            if frame_embeddings: 
                frame_embeddings = np.stack(frame_embeddings, axis=0) # turn the list into one array

                all_embeddings.append(frame_embeddings)
                all_candidate_points.append(candidate_points)

                all_frame_numbers.append(
                    np.full(
                        shape=len(candidate_points),
                        fill_value=frame_index, 
                        dtype=np.int32,
                    )
                ) # record which frame these candidates came from

                total_candidate_processed += len(candidate_points)
            print(f"Total candidates processed so far: {total_candidate_processed}")
    if not all_embeddings: 
        raise RuntimeError("No embeddings were generated.")

    # join every frame's result inot three single arrays.
    embeddings = np.concatenate(all_embeddings, axis=0)
    candidate_points = np.concatenate(all_candidate_points, axis=0)
    candidate_frame_numbers = np.concatenate(all_frame_numbers, axis=0)

    if not (len(embeddings) == len(candidate_points) == len(candidate_frame_numbers)):
        raise RuntimeError("The number of embeddings, candidate points and frame numbers do not match")
    
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Final candidate-points shape: {candidate_points.shape}")
    print(f"Final frame-number array shape: {candidate_frame_numbers.shape}")

    np.save(EMB_OUTPUT_PATH, embeddings)
    np.save(POINTS_OUTPUT_PATH, candidate_points)
    np.save(FRAME_NUMBERS_OUTPUT_PATH, candidate_frame_numbers)

    seconds_per_candidate = total_embedding_seconds / total_candidate_processed # average time for one candidate

    rate_info = {
        "stack_name": STACK_NAME,
        "seconds_per_candidate": seconds_per_candidate,
        "total_candidates": int(total_candidate_processed),
        "total_embeddings_seconds": total_embedding_seconds,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "crop_size_z_y_x": [CROP_SIZE_Z, CROP_SIZE_Y, CROP_SIZE_X]
    }
    
    with open(RATE_OUTPUT_PATH, "w") as f:
        json.dump(rate_info, f, indent=2)

    print(f"Processing completed successfully")

if __name__ == "__main__":
    main()


    





