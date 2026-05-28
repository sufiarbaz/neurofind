import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import tifffile
import napari
import torch

from utils import (
    crop_around_point,
    compute_embedding,
    compare_embeddings,
    DINOv3Encoder
)

BASE_DIR = Path("C:/Users/z0051rra/Downloads/Neurofind Project")

query_stack_path = BASE_DIR / "data" / "time_data_labeled" / "33648_A1_TS_dftcorr.tif"
target_stack_path = BASE_DIR / "data" / "time_data_labeled" / "33648_A5_TS_dftcorr.tif"

emb_path = BASE_DIR / "GUI" / "embeddings" / "embeddings_dinov3_A5_128_7_test.npy"
points_path = BASE_DIR / "GUI" / "embeddings" / "candidate_points_A5_test.npy"
model_path = BASE_DIR / "models" / "spine_embedder_ssl_dinov3_128_7_5pth.sec"

query_stack = tifffile.imread(query_stack_path)[0]
target_stack = tifffile.imread(target_stack_path)[0]

max_depth = max(query_stack.shape[0], target_stack.shape[0])

def pad_stack_to_depth(stack, target_depth):
    current_depth = stack.shape[0]
    if current_depth >= target_depth:
        return stack
    pad_after = target_depth - current_depth
    padded_stack = np.pad(
        stack,
        pad_width=((0, pad_after), (0, 0), (0, 0)),
        mode="edge"
    )
    return padded_stack

query_stack = pad_stack_to_depth(query_stack, max_depth)
target_stack = pad_stack_to_depth(target_stack, max_depth)


print(f"Query stack shape: {query_stack.shape}")
print(f"Target stack shape: {target_stack.shape}")


candidate_points_target = np.load(points_path)
embeddings_target = np.load(emb_path)

if len(candidate_points_target) != len(embeddings_target):
    raise ValueError("Number of candidate points and embeddings are not the same.")

print(f"Loaded {len(candidate_points_target)} candidate points")
print(f"Loaded embeddings shape: {embeddings_target.shape}")

selected_coordinates_layer = None
match_layer = None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = DINOv3Encoder(embedding_dim=128, freeze_backbone=False).to(device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def on_click_query_stack(layer, event):
    global selected_coordinates_layer, match_layer

    coords = layer.world_to_data(event.position)
    z, y, x = map(int, coords)

    if z < 0 or z >= query_stack.shape[0]:
        print("Invalid z-click for query stack.")
        return

    print(f"Clicked point: z={z}, y={y}, x={x}")

    if selected_coordinates_layer is not None:
        viewer.layers.remove(selected_coordinates_layer)

    selected_coordinates_layer = viewer.add_points(
        np.array([[z, y, x]]),
        name="Selected point",
        size=8,
        face_color="red",
    )

    crop = crop_around_point(
        query_stack,
        (z, y, x),
        size_z=7,
        size_y=128,
        size_x=128
    )

    query_embedding = compute_embedding(crop, model, device)
    query_tensor = query_embedding.float()

    embeddings_tensor = torch.from_numpy(embeddings_target).float()
    _, similarity = compare_embeddings(query_tensor, embeddings_tensor)

    similarity_np = similarity.detach().cpu().numpy()
    best_index = int(np.argmax(similarity_np))
    best_point = candidate_points_target[best_index]

    print(f"Best match: z={best_point[0]}, y={best_point[1]}, x={best_point[2]}")

    if match_layer is not None:
        viewer.layers.remove(match_layer)

    match_layer = viewer.add_points(
        np.array([best_point]),
        name="Match",
        size=8,
        face_color="green"
    )


viewer = napari.Viewer()

query_layer = viewer.add_image(query_stack, name="Query Stack")
query_layer.mouse_drag_callbacks.append(on_click_query_stack)

viewer.add_image(target_stack, name="Target Stack")

napari.run()