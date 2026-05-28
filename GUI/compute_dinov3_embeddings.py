import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from utils import (
    get_points_stack2,
    crop_around_point,
    compute_embedding,
    DINOv3Encoder,
)

BASE_DIR = Path("C:/Users/z0051rra/Downloads/Neurofind Project")

TARGET_STACK_NAME = "A5"
TARGET_STACK_PATH = BASE_DIR / "data" / "time_data_labeled" / "33648_A5_TS_dftcorr.tif"

MODEL_PATH = BASE_DIR / "models" / "spine_embedder_ssl_dinov3_128_7_5pth.sec"

OUTPUT_DIR = BASE_DIR / "GUI" / "embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set to an integer for testing, e.g. 1000.
# Set to None to compute embeddings for all candidates.
MAX_CANDIDATES = 1000

CROP_SIZE_Z = 7
CROP_SIZE_Y = 128
CROP_SIZE_X = 128

EMB_OUTPUT_PATH = OUTPUT_DIR / f"embeddings_dinov3_{TARGET_STACK_NAME}_128_7.npy"
POINTS_OUTPUT_PATH = OUTPUT_DIR / f"candidate_points_{TARGET_STACK_NAME}.npy"

# Load target stack
target_stack_raw = tifffile.imread(TARGET_STACK_PATH)
print(f"Raw target stack shape: {target_stack_raw.shape}")

# The TIFF contains multiple 3D volumes.
# For now, use the first volume.
target_stack = target_stack_raw[0]
print(f"Using target stack shape: {target_stack.shape}")

# Generate candidate points
candidate_points = np.asarray(get_points_stack2(target_stack))
print(f"Total candidate points: {len(candidate_points)}")

if MAX_CANDIDATES is not None:
    candidate_points = candidate_points[:MAX_CANDIDATES]
    print(f"Using first {MAX_CANDIDATES} candidate points for test run.")
else:
    print("Using all candidate points for embedding computation.")

# Load DINOv3 model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("WARNING: Running on CPU. Full embedding computation may take many hours.")

model = DINOv3Encoder(
    embedding_dim=128,
    freeze_backbone=False,
).to(device=device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Compute embeddings
embeddings = []

for point in tqdm(candidate_points, desc="Computing DINOv3 embeddings"):
    crop = crop_around_point(
        target_stack,
        point,
        size_z=CROP_SIZE_Z,
        size_y=CROP_SIZE_Y,
        size_x=CROP_SIZE_X,
    )

    embedding = compute_embedding(crop, model, device)
    embeddings.append(embedding.detach().cpu().numpy())

embeddings = np.stack(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

np.save(EMB_OUTPUT_PATH, embeddings)
np.save(POINTS_OUTPUT_PATH, candidate_points)

print("Embeddings saved!")
print("Candidate points saved!")