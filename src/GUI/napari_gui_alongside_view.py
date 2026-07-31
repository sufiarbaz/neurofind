"""
napari_gui_alongside_view.py

An interactive viewer, click a point in the query frame, and the model finds the best-matching point in the target frame and draws it.

This is the SIDE-BY-SIDE variant: the query and target frames are shown in two
separate napari windows, positioned next to each other, instead of overlaid on
top of each other in one window (see napari_gui.py for that original version).

When a click happens, it crop a box around the click, turn it into a fingerprint.
Compare that fingerprint against EVERY candidate in the target frame, and show the closest match. 

How to run it (from the project root):
    python -m src.GUI.napari_gui_alongside_view
"""#

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows workaround so numpy/torch math libraries don't clash
 
from pathlib import Path
import numpy as np
import tifffile
import napari
import torch
 
from src.base.utils import (crop_around_point, compute_embedding, compare_embeddings, DINOv3Encoder)
from src.base import stack_settings
 
# choose stack and frames
STACK_NAME = stack_settings.STACK_NAME # short name for the stack, e.g., "A5" or "A1"
STACK_FILENAME = stack_settings.STACK_FILENAME # place this raw TIFF in: data/time_data_labeled/
 
QUERY_FRAME = stack_settings.QUERY_FRAME # the frame where the query comes from
TARGET_FRAME = stack_settings.TARGET_FRAME # the frame we search for the match
 
# crop size taken around the clicked point before making its fingerprint (the embedding)
CROP_SIZE_Z = stack_settings.CROP_SIZE_Z
CROP_SIZE_Y = stack_settings.CROP_SIZE_Y
CROP_SIZE_X = stack_settings.CROP_SIZE_X
 
BASE_DIR = Path(__file__).resolve().parents[2] # path to the project root
 
STACK_PATH = BASE_DIR / "data" / "time_data_labeled" / STACK_FILENAME
 
# The precomputed files. They live in data/embeddings/
# They are produced by running src/base/compute_full_embeddings_hpc.py
TARGET_CANDIDATES_PATH = BASE_DIR / "data" / "embeddings" / f"candidate_points_{STACK_NAME}.npy"
TARGET_EMBEDDINGS_PATH = BASE_DIR / "data" / "embeddings" / f"embeddings_dinov3_{STACK_NAME}.npy"
TARGET_VOLUME_INDICES_PATH = BASE_DIR / "data" / "embeddings" / f"candidate_frame_numbers_{STACK_NAME}.npy"
 
MODEL_PATH = BASE_DIR / "models" / "spine_embedder_ssl_dinov3_128_7_5pth.sec"  # place the trained model in: models/
 
 
# Use the GPU if one is available, otherwise the CPU.
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
 
 
# load the TIFF and return the query frame and target frame volumes.
def load_frames():
    time_series = tifffile.imread(STACK_PATH)  # read the 4D TIFF: (time, z, y, x)
 
    if time_series.ndim != 4:
        raise ValueError(
            f"Expected a 4D time series (time, z, y, x), but got {time_series.shape}."
        )
 
    if QUERY_FRAME >= time_series.shape[0] or TARGET_FRAME >= time_series.shape[0]:
        raise IndexError("QUERY_FRAME or TARGET_FRAME is outside the available time points.")
 
    query_stack = time_series[QUERY_FRAME]    # the 3D volume you click in
    target_stack = time_series[TARGET_FRAME]  # the 3D volume that gets searched
 
    print(f"Query frame {QUERY_FRAME} shape: {query_stack.shape}")
    print(f"Target frame {TARGET_FRAME} shape: {target_stack.shape}")
 
    return query_stack, target_stack
 
 
# load the candidate points and their fingerprints, keeping only the target frame's.
def load_candidates():
    # mmap_mode="r" means the files are not fully loaded into memory straight away.
    # we only need the target frame's slice of them, so there is no point loading it all.
    all_candidate_points = np.load(TARGET_CANDIDATES_PATH, mmap_mode="r")
    all_embeddings = np.load(TARGET_EMBEDDINGS_PATH, mmap_mode="r")
    all_volume_indices = np.load(TARGET_VOLUME_INDICES_PATH, mmap_mode="r")
 
    if not (len(all_candidate_points) == len(all_embeddings) == len(all_volume_indices)):
        raise ValueError("Candidate points, embeddings, and volume indices have different lengths.")
 
    # keep only the candidates that belong to the target frame.
    target_mask = np.asarray(all_volume_indices == TARGET_FRAME)
    candidate_points_target = np.asarray(all_candidate_points[target_mask])
    embeddings_target = np.asarray(all_embeddings[target_mask])
 
    print(f"Target-frame candidates: {len(candidate_points_target)}")
 
    return candidate_points_target, embeddings_target
 
 
# load the trained model, ready to make fingerprints.
def load_model(device):
    model = DINOv3Encoder(embedding_dim=128, freeze_backbone=False).to(device=device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # switch to evaluation mode (for using, not training)
    return model
 
 
# given a clicked point, make its fingerprint and find the best-matching candidate.
def predict_match(clicked_point, query_stack, model, device,
                  embeddings_tensor, candidate_points_target):
    z, y, x = clicked_point
 
    # cut a box around the clicked point and turn it into a fingerprint.
    crop = crop_around_point(
        query_stack, (z, y, x),
        size_z=CROP_SIZE_Z, size_y=CROP_SIZE_Y, size_x=CROP_SIZE_X,
    )
 
    with torch.no_grad():  # we are only using the model, not training it
        query_embedding = compute_embedding(crop, model, device).float().to(device)
        _, similarity = compare_embeddings(query_embedding, embeddings_tensor)
 
    # the candidate with the highest similarity score is the match.
    similarity_np = similarity.detach().cpu().numpy().reshape(-1)
    best_index = int(np.argmax(similarity_np))
    predicted_point = np.asarray(candidate_points_target[best_index], dtype=float)
    best_similarity = float(similarity_np[best_index])
 
    return predicted_point, best_similarity
 
 
# build the napari viewers and set up the click-to-predict behaviour.
# Two separate windows are used (query and target), positioned side by side,
# instead of one viewer with both images overlaid on top of each other.
def run_gui(query_stack, target_stack, model, device,
            embeddings_tensor, candidate_points_target):
    query_viewer = napari.Viewer(title=f"{STACK_NAME} frame {QUERY_FRAME} - Query")
    target_viewer = napari.Viewer(title=f"{STACK_NAME} frame {TARGET_FRAME} - Target")
 
    query_layer = query_viewer.add_image(query_stack, name=f"{STACK_NAME} frame {QUERY_FRAME} - Query")
    target_viewer.add_image(target_stack, name=f"{STACK_NAME} frame {TARGET_FRAME} - Target")
 
    # Try to auto-arrange the two windows side by side on screen.
    # This uses a private napari/Qt attribute, so it is wrapped in a try/except:
    # if it fails on some napari version, the two windows still open normally,
    # just not auto-positioned (drag them side by side manually in that case).
    try:
        from qtpy.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        half_width = screen.width() // 2
        query_viewer.window._qt_window.setGeometry(0, 0, half_width, screen.height())
        target_viewer.window._qt_window.setGeometry(half_width, 0, half_width, screen.height())
    except Exception as e:
        print(f"Could not auto-arrange windows side by side: {e}")
 
    # handles for the dots, so each new click can clear the previous ones.
    query_point_layer = None
    predicted_point_layer = None
 
    def remove_layer_if_present(viewer, layer):
        if layer is not None and layer in viewer.layers:
            viewer.layers.remove(layer)
 
    def on_click_query_stack(layer, event):
        # nonlocal is needed because this inner function reassigns the two variables above. Without it, Python would make new local ones instead.
        nonlocal query_point_layer, predicted_point_layer
 
        # only react to a normal left mouse click.
        if event.type != "mouse_press" or event.button != 1:
            return
 
        # turn the screen click position into image (z, y, x) coordinates.
        coords = layer.world_to_data(event.position)
        clicked_point = np.rint(coords).astype(int)
        z, y, x = clicked_point.tolist()
 
        # ignore clicks that land outside the image.
        if not (
            0 <= z < query_stack.shape[0]
            and 0 <= y < query_stack.shape[1]
            and 0 <= x < query_stack.shape[2]
        ):
            print("Click is outside the query stack.")
            return
 
        print(f"\nQuery point: z={z}, y={y}, x={x}")
 
        # clear the dots from the previous click, each in its own window.
        remove_layer_if_present(query_viewer, query_point_layer)
        remove_layer_if_present(target_viewer, predicted_point_layer)
 
        # red dot: where you clicked, shown in the query window.
        query_point_layer = query_viewer.add_points(
            np.array([[z, y, x]], dtype=float),
            name="Query point",
            size=10,
            face_color="red",
            border_color="white",
        )
 
        # run the model to find the best-matching candidate in the target frame.
        predicted_point, best_similarity = predict_match(
            (z, y, x), query_stack, model, device,
            embeddings_tensor, candidate_points_target,
        )
 
        print(
            f"Predicted point: z={predicted_point[0]:.0f}, "
            f"y={predicted_point[1]:.0f}, x={predicted_point[2]:.0f}, "
            f"similarity={best_similarity:.6f}"
        )
 
        # green dot: the model's prediction, shown in the target window.
        predicted_point_layer = target_viewer.add_points(
            np.array([predicted_point]),
            name="Predicted point",
            size=10,
            face_color="green",
            border_color="white",
        )
 
    # attach the click handler to the query image, in the query window.
    query_layer.mouse_drag_callbacks.append(on_click_query_stack)
 
    # start the Qt event loop -- this runs both windows at once.
    napari.run()
 
def main():
    device = get_device()
    print(f"Using device: {device}")
 
    query_stack, target_stack = load_frames()
    candidate_points_target, embeddings_target = load_candidates()
    model = load_model(device)
 
    # build the candidate fingerprint tensor once, so every click reuses it instead of rebuilding it each time.
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