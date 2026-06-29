"""
data_loading.py

All the loading of data is done here.
"""

import sys # to modify the import path
import numpy as np
import pandas as pd
import tifffile
import torch

from experiments.local_search import config

sys.path.append(str(config.GUI_DIR))
from utils import DINOv3Encoder # import must come after sys.path change

# Load the query/target pairs file (produced by organise_labeled_data.py)
def load_label_pairs():
    label_pairs = pd.read_csv(config.LABEL_PAIRS_PATH)
    required_columns = {
        "track_id",
        "query_frame",
        "target_frame",
        "query_x_um",
        "query_y_um",
        "query_z_slice",
        "target_x_um",
        "target_y_um",
        "target_z_slice"
    }

    missing_columns = required_columns - set(label_pairs.columns)
    if missing_columns:
        raise ValueError(
            f"The pairs file is missing columns: {sorted(missing_columns)}"
        )
    
    if config.NUMBER_OF_PAIRS_TO_RUN is not None: # the test limit switch from config
        label_pairs = label_pairs.iloc[: config.NUMBER_OF_PAIRS_TO_RUN].copy()
    
    return label_pairs.reset_index(drop=True)

# load the precomputed arrays for the stack
def load_candidate_arrays():
    all_candidate_points = np.load(config.TARGET_CANDIDATES_PATH)
    all_target_embeddings = np.load(config.TARGET_EMBEDDINGS_PATH)
    all_volume_indices = np.load(config.TARGET_VOLUME_INDICES_PATH)

    if not (
        len(all_candidate_points) == len(all_target_embeddings) == len(all_volume_indices)
    ):
        raise ValueError("Candidate points, embeddings, and volume indices must have the same length but they don't")
    
    return all_candidate_points, all_target_embeddings, all_volume_indices

# Keep only the candidates belonging to one time-frame
def candidates_for_frame(all_candidate_points, all_target_embeddings, all_volume_indices, frame):
    frame_mask = all_volume_indices == frame # produces a true/false mask wherever a candidate's frame label equal the required frame
    candidate_points = all_candidate_points[frame_mask]
    target_embeddings_numpy = all_target_embeddings[frame_mask]

    if len(candidate_points) == 0:
        raise ValueError(f"No candidates found for frame {frame}")
    
    return candidate_points, target_embeddings_numpy

# Load the full TIFF stack, keeping the time dimension
def load_time_stack():
    stack = tifffile.imread(config.A5_STACK_PATH)

    if stack.ndim != 4:
        raise ValueError(f"Expected the A5 stack to be 4D but got {stack.shape}")
    
    return stack

# Load the trained DINOv3 spine-embedding model
def load_model(device):
    model = DINOv3Encoder() # create an instance of the model architecture

    checkpoint = torch.load(config.MODEL_PATH, map_location=device) # load the trained weights from the model file, checkpoint is the learned knowledge
    model.load_state_dict(checkpoint) # pour the trained weights into the empty model structure

    model.to(device) # move the model onto the chosen device
    model.eval() # switch the model to evaluation mode

    return model # return the ready-to-use model

# Use the GPU if one is available, otherwise CPU
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") # pick where computation run








