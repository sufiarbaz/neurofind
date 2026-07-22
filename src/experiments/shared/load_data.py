"""
load_data.py

All the loading of data is done here. 
"""

import numpy as np # for loading and handling the candidate files
import pandas as pd # for reading the pairs CSV as a table
import tifffile # for reading the TIFF image stack
import torch # for loading the trained model
import json # to read the embedding rate file

from src.base.utils import DINOv3Encoder # the model class, imported cleanly from src/base/
from src.base import organise_labeled_data

# load the query/target pairs file
def load_label_pairs(config):

    if not config.LABEL_PAIRS_PATH.is_file():
        print("Pairs file not found. Creating it from the raw labels...")
        created_path = organise_labeled_data.main()
        print(f"Created: {created_path.name}")
    label_pairs = pd.read_csv(config.LABEL_PAIRS_PATH)

    # check the columns we must have
    required_columns = {
        "track_id",
        "query_frame",
        "target_frame",
        "query_x_um",
        "query_y_um",
        "query_z_slice",
        "target_x_um",
        "target_y_um",
        "target_z_slice",
    }
    missing_columns = required_columns - set(label_pairs.columns)
    if missing_columns:
        raise ValueError(f"The pairs file is missing columns: {sorted(missing_columns)}")
    
    # if the test limit is set, keep only the first few rows
    if config.NUMBER_OF_PAIRS_TO_RUN is not None:
        label_pairs = label_pairs.iloc[: config.NUMBER_OF_PAIRS_TO_RUN].copy()

    return label_pairs.reset_index(drop=True)

# load the precomputed files by compute_full_embeddings_hpc.py (candidate_positions, their fingerprints, volume indices)
def load_candidate_arrays(config):
    all_candidate_points = np.load(config.TARGET_CANDIDATES_PATH)
    all_target_embeddings = np.load(config.TARGET_EMBEDDINGS_PATH)
    all_volume_indices = np.load(config.TARGET_VOLUME_INDICES_PATH)

    if not (len(all_candidate_points) == len(all_target_embeddings) == len(all_volume_indices)):
        raise ValueError("Candidate points, embeddings, and volume indices must have the same length but they don't.")
    
    return all_candidate_points, all_target_embeddings, all_volume_indices

# load how long the model took to make one fingerprint, measured during the HPC run
def load_embedding_rate(config):
    if not config.EMBEDDING_RATE_PATH.is_file():
        raise FileNotFoundError("The embedding rate file was not found.")
    
    with open(config.EMBEDDING_RATE_PATH) as f:
        rate_info = json.load(f)
        
    return rate_info["seconds_per_candidate"]

# keep only the candidates that belong to one time-frame
def candidates_for_frame(all_candidate_points, all_target_embeddings, all_volume_indices, frame):
    frame_mask = all_volume_indices == frame # true wherever a candidate belongs to the frame we want

    candidate_points = all_candidate_points[frame_mask]
    target_embeddings = all_target_embeddings[frame_mask]

    if len(candidate_points) == 0:
        raise ValueError(f"No candidates found for frame {frame}")
    
    return candidate_points, target_embeddings

# load the full image stack
def load_time_stack(config):
    stack = tifffile.imread(config.STACK_PATH)

    if stack.ndim != 4:
        raise ValueError(f"Expected the stack to be 4D (time, z, y, x) but got {stack.shape}")
    
    return stack

# load the trained model and get it ready for use
def load_model(device, config):
    model = DINOv3Encoder() # build the empty model structure

    checkpoint = torch.load(config.MODEL_PATH, map_location=device) # load the trained weights
    model.load_state_dict(checkpoint) # put the trained weights into the model

    model.to(device) # move the model to the chosen device
    model.eval() # switch to evaluation mode

    return model

# pick where the computation runs, the GPU if there is one, otherwise the CPU
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




    
