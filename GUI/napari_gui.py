import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows workaround so numpy/torch math libraries don't clash

from pathlib import Path
import numpy as np
import tifffile
import napari
import torch

from utils import (
    crop_around_point,  
    compute_embedding,    
    compare_embeddings,   
    DINOv3Encoder,        
)

# Project root, found from this file's own location.
BASE_DIR = Path(__file__).resolve().parents[1]

# One time-series file, frame 0 is the query, frame 1 is the target.
STACK_PATH = BASE_DIR / "data" / "time_data_labeled" / "33648_A5_TS_dftcorr.tif"

EMB_PATH = BASE_DIR / "data" / "embeddings" / "embeddings_dinov3_A5_128_7.npy"
POINTS_PATH = BASE_DIR / "data" / "embeddings" / "candidate_points_A5.npy"
VOLUME_INDICES_PATH = BASE_DIR / "data" / "embeddings" / "candidate_volume_indices_A5.npy"
MODEL_PATH = BASE_DIR / "models" / "spine_embedder_ssl_dinov3_128_7_5pth.sec"

QUERY_FRAME = 0    # the frame you click in
TARGET_FRAME = 1   # the frame the model searches in

# Use the GPU if one is available, otherwise the CPU.
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Load the TIFF and return the query frame and target frame volumes.
def load_frames():
    time_series = tifffile.imread(STACK_PATH)   # read the 4D TIFF: (time, z, y, x)

    if time_series.ndim != 4:
        raise ValueError(
            f"Expected a 4D time series (time, z, y, x), but got {time_series.shape}."
        )

    if QUERY_FRAME >= time_series.shape[0] or TARGET_FRAME >= time_series.shape[0]:
        raise IndexError("QUERY_FRAME or TARGET_FRAME is outside the available time points.")

    query_stack = time_series[QUERY_FRAME]      # 3D volume for frame 0 (what you click in)
    target_stack = time_series[TARGET_FRAME]    # 3D volume for frame 1 (what's searched)

    print(f"Query frame {QUERY_FRAME} shape: {query_stack.shape}")
    print(f"Target frame {TARGET_FRAME} shape: {target_stack.shape}")

    return query_stack, target_stack


# Load the candidate points and embeddings, keeping only the target frame.
def load_candidates():
    
    all_candidate_points = np.load(POINTS_PATH, mmap_mode="r")        # memory-mapped, not fully in RAM
    all_embeddings = np.load(EMB_PATH, mmap_mode="r")
    all_volume_indices = np.load(VOLUME_INDICES_PATH, mmap_mode="r")

    if not (len(all_candidate_points) == len(all_embeddings) == len(all_volume_indices)):
        raise ValueError("Candidate points, embeddings, and volume indices have different lengths.")

    target_mask = np.asarray(all_volume_indices == TARGET_FRAME)     # which candidates are in frame 1
    candidate_points_target = np.asarray(all_candidate_points[target_mask])  # frame-1 points (into RAM)
    embeddings_target = np.asarray(all_embeddings[target_mask])      # frame-1 embeddings (into RAM)

    print(f"Target-frame candidates: {len(candidate_points_target)}")

    return candidate_points_target, embeddings_target

# Load the trained DINOv3 model, ready to compute embeddings.
def load_model(device):
    model = DINOv3Encoder(embedding_dim=128, freeze_backbone=False).to(device=device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


# Given a clicked (z, y, x) point, compute its embedding and return the best-matching candidate point and that match's similarity score.
def predict_match(clicked_point, query_stack, model, device,
                  embeddings_tensor, candidate_points_target):
    z, y, x = clicked_point

    # Crop a 3D box around the clicked point and compute its embedding.
    crop = crop_around_point(query_stack, (z, y, x), size_z=7, size_y=128, size_x=128)

    with torch.no_grad():
        query_embedding = compute_embedding(crop, model, device).float().to(device)
        _, similarity = compare_embeddings(query_embedding, embeddings_tensor)

    # Pick the candidate with the highest similarity = the predicted match.
    similarity_np = similarity.detach().cpu().numpy().reshape(-1)
    best_index = int(np.argmax(similarity_np))
    predicted_point = np.asarray(candidate_points_target[best_index], dtype=float)
    best_similarity = float(similarity_np[best_index])

    return predicted_point, best_similarity


# Build the napari viewer and wire up the click-to-predict interaction.
def run_gui(query_stack, target_stack, model, device,
            embeddings_tensor, candidate_points_target):
    viewer = napari.Viewer()

    query_layer = viewer.add_image(query_stack, name=f"A5 frame {QUERY_FRAME} - Query")
    viewer.add_image(target_stack, name=f"A5 frame {TARGET_FRAME} - Target")

    # Layer handles for the dots, so each click can clear the previous ones.
    query_point_layer = None
    predicted_point_layer = None

    def remove_layer_if_present(layer):
        if layer is not None and layer in viewer.layers:
            viewer.layers.remove(layer)

    def on_click_query_stack(layer, event):
        nonlocal query_point_layer, predicted_point_layer

        # Only react to a normal left mouse click.
        if event.type != "mouse_press" or event.button != 1:
            return

        # Convert the screen click position into image (z, y, x) coordinates.
        coords = layer.world_to_data(event.position)
        clicked_point = np.rint(coords).astype(int)
        z, y, x = clicked_point.tolist()

        # Ignore clicks that land outside the image.
        if not (
            0 <= z < query_stack.shape[0]
            and 0 <= y < query_stack.shape[1]
            and 0 <= x < query_stack.shape[2]
        ):
            print("Click is outside the query stack.")
            return

        print(f"\nQuery point: z={z}, y={y}, x={x}")

        # Clear the dots drawn by the previous click.
        remove_layer_if_present(query_point_layer)
        remove_layer_if_present(predicted_point_layer)

        # Red dot: the point the user clicked (the query).
        query_point_layer = viewer.add_points(
            np.array([[z, y, x]], dtype=float),
            name="Query point",
            size=10,
            face_color="red",
            border_color="white",
        )

        # Run the model to find the best-matching target candidate.
        predicted_point, best_similarity = predict_match(
            (z, y, x), query_stack, model, device,
            embeddings_tensor, candidate_points_target,
        )

        print(
            f"Predicted point: z={predicted_point[0]:.0f}, "
            f"y={predicted_point[1]:.0f}, x={predicted_point[2]:.0f}, "
            f"similarity={best_similarity:.6f}"
        )

        # Green dot: the model's predicted match (drawn at its true coordinate).
        predicted_point_layer = viewer.add_points(
            np.array([predicted_point]),
            name="Predicted point",
            size=10,
            face_color="green",
            border_color="white",
        )

    # Attach the click handler to the query image layer.
    query_layer.mouse_drag_callbacks.append(on_click_query_stack)

    # Start the napari event loop (opens the window).
    napari.run()


def main():
    device = get_device()
    print(f"Using device: {device}")

    query_stack, target_stack = load_frames()
    candidate_points_target, embeddings_target = load_candidates()
    model = load_model(device)

    # Build the candidate embedding tensor once and reuse it for every click.
    embeddings_tensor = torch.from_numpy(embeddings_target).float().to(device)

    run_gui(
        query_stack,
        target_stack,
        model,
        device,
        embeddings_tensor,
        candidate_points_target,
    )


if __name__ == "__main__":
    main()
