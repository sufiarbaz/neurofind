from pathlib import Path #import path so we can build file paths that work across operating systems
import time # import time so we can measure how long full search and local search take
import numpy as np # import numpy for loading .npy files and doing array operations
import torch # import PyTorch because embeddings are compared as Torch tensors.
import torch.nn.functional as F # import cosine similarity function from PyTorch
import sys # to manually add another folder to Python's import search path
import tifffile # to read microscopy image stacks saved as .tif files
import pandas as pd # to show results as a clean table

# Get the project root folder.
# __file__ is the path of this script.
# parents[1] goes two levles up from experiments/local_search_minimal.py to the project root.
BASE_DIR = Path(__file__).resolve().parents[1]
GUI_DIR = Path(BASE_DIR / "GUI") # path to GUI folder where model and helper functions are defined.
sys.path.append(str(GUI_DIR)) # add the GUI folder to Python's module search path, this allows Python to find and import utils.py from the GUI folder.
from utils import DINOv3Encoder, crop_around_point, compute_embedding

DATA_DIR = BASE_DIR / "data" / "time_data_labeled"
QUERY_STACK_PATH = DATA_DIR / "33648_A1_TS_dftcorr.tif"
MODEL_PATH = BASE_DIR / "models" / "spine_embedder_ssl_dinov3_128_7_5pth.sec"

TARGET_CANDIDATES_PATH = BASE_DIR / "outputs" / "embeddings" / "candidate_points_A5_test.npy" # path to saved candidate points coordinates
TARGET_EMBEDDINGS_PATH = BASE_DIR / "outputs" / "embeddings" / "embeddings_dinov3_A5_128_7_test.npy" # path to saved DINOv3 embeddings for candidate points (test subset of 1000 candidates)

LABEL_PAIRS_PATH = BASE_DIR / "outputs" / "label_pairs_A1_A5_frame0.csv" # loads the 15 human-labeled A1 and A5 pairs.
OUTPUT_RESULTS_PATH = BASE_DIR / "outputs" / "Ex1_local_search_results_A1_A5.csv" # saves the experiment result table

# Now the query point should come from "label_pairs_A1_A5_frame0.csv"
# QUERY_POINT = np.array([3, 256, 256]) # example query point coordinates (z, y, x), later come from real click
SEARCH_RADII = [50, 100, 150, 200, 250] # example search radii in pixels for local search

def load_stack(path): # define a function that laods a TIFF image stack
    stack = tifffile.imread(path) # read the TIFF file using tifffile, the result is a Numpy array containing the image data.
    if stack.ndim == 4: # check if the loaded stack has 4 dimensions, can have shape like (time, z, y, x)
        stack = stack[0] # select the first time volume, this changes the stack from (time, z, y, x) to (z, y, x)
    return stack # return the 3D stack

def filter_candidates_by_radius(candidate_points, center_point, radius): # define a function that keeps only target candidates close to the query point.
    # candidate_points has shape (N, 3) where N is number of candidates and each candidates has three coordinate values (z, y, x).
    # center_point is the selected/query point, it also has the form (z, y, x).
    # radius is the maximum allowed distance from the center point for candidates to be included in the local search.

    dy = candidate_points[:, 1] - center_point[1] # compute difference in y coordinate between every candidate and the center point.
    dx = candidate_points[:, 2] - center_point[2] # compute difference in x coordinate between every candidate and the center point.

    distances = np.sqrt(dy**2 + dx**2) # compute 2D distance in thee y-x image plane, we ignore z here because query stack and target stack can have different z-depths.
    mask = distances <= radius # create  a bookean mask that is True for candidates inside the radius and false means the candidate is outside the radius.
    local_candidates = candidate_points[mask] # use the mask to select only the nearby candidate points.
    return local_candidates, mask # return both

def find_best_match(query_embedding, target_embeddings, candidate_points): # define a function that compares one query embedding with many target embeddings.
    # query_embedding is one of feature vector, shape is usually (128,).
    # target_embeddings contains many feature vectors, shape is usually (N, 128).
    # candidate_points contains the coordinates  belonging to target_embeddings, candidate_points[i] belongs to target_embeddings[i].
    query_embedding_batch = query_embedding.unsqueeze(0) # add one batch dimension to query_embdding, this changes shape from (128,) to (1, 128) so we can compare it with all target embeddings at once.
    similarities = F.cosine_similarity( # compute cosine similarity between the query embedding and every target embedding, output shape is (N,), higher value means more similar according to the embedding model.
        query_embedding_batch, 
        target_embeddings, 
        dim=1
    )
    best_idx = torch.argmax(similarities).item() # find the index of the highest similarity score, this is the best matching target candidate.
    best_point = candidate_points[best_idx] # use the best index to get the coordinatte of the best matching coordinate.
    highest_similarity_score = similarities[best_idx].item() # use the best index to get thee similarity score of the best match.
    return best_point, highest_similarity_score # return predicted best coordinate and its similarity score.

# this helper method measure how far the predicted point is from the human-labeled target point, later this becoms the accuracy-related metric.
def distance_3d(point_a, point_b):
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)

def main(): # define the main function where the experiment runs.
    print("Starting minimal local-search test...")
    results = [] # create an empty list where each experiment result will be stored

    # loads the 15 matched A1 and A5 human label pairs
    label_pairs = pd.read_csv(LABEL_PAIRS_PATH)
    print(f"Number of label pairs: {len(label_pairs)}")

    candidate_points = np.load(TARGET_CANDIDATES_PATH) # load candidate point coordinates from the .npy file.
    target_embeddings = np.load(TARGET_EMBEDDINGS_PATH) # load precomputed DINOv3 embeddings from the .npy file.
    
    print(f"Candidate points shape: {candidate_points.shape}")
    print(f"Target embeddings shape: {target_embeddings.shape}")

    target_embeddings = torch.from_numpy(target_embeddings) # convert target embeddings from NumPy array to Torch tensor, PyTorch cosine_similarity expects Torch tensors.
    target_embeddings = target_embeddings.float() # convert target embeddings to float32, this avoid dtype problems during similarity computation.

    # query_embedding = target_embeddings[0]  # for this minimal test, we use the first target embedding as a fake query embedding, this lets us test the search code before adding DINOv3 crop computation.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # select the device for model computation, if a GPU is available, use CUDA; otherwise use CPU.
    print(f"Device: {device}")

    query_stack = load_stack(QUERY_STACK_PATH) # load the A1 query stack from the TIFF file. 
    print(f"Query stack shape: {query_stack.shape}") # print the query stack shape to veify that it is a 3D volume.

    model = DINOv3Encoder() # create the DINOv3 model object.
    checkpoint = torch.load(MODEL_PATH, map_location=device) # load the saved trained weights from the checkpoint file.
    model.load_state_dict(checkpoint) # put the checkpoint weights into the model architecuter.
    model.to(device) # move the model to the selected device
    model.eval() # put the model in evaluation mode means we use the model for inference, not training. 
    
    # query_point_tuple = tuple(QUERY_POINT) # convert QUERY_POINT from NumPy array to a normal tuple, crop_around_point expects the point as (z, y, x)
    # row = label_pairs.iloc[0] # first we test only the first human-labeled pair
    for _, row in label_pairs.iterrows():
        query_point_tuple = ( # creates the A1 query point in the correct image-array order
            int(row["query_px_z"]),
            int(row["query_px_y"]),
            int(row["query_px_x"])
        )
        target_point_tuple = ( # human-labeled correct A5 target point
            int(row["target_px_z"]),
            int(row["target_px_y"]),
            int(row["target_px_x"]),
        )
        print("\nUsing label pair:")
        print(f"track_id: {row["track_id"]}")
        print("query_point A1 z, y, x:", query_point_tuple)
        print("target point A5 z, y, x:", target_point_tuple)

        z, y, x = query_point_tuple # extract z, y and x from the query point.
        if z < 0 or z >= query_stack.shape[0]: # check the z-coordinate is valid for the query stack, this prevents trying to crop from a z-slice that does not exist.
            raise ValueError(
                f"Invalid z={z}. Query stack valid z range is 0 to {query_stack.shape[0] - 1}."
            )

        crop = crop_around_point( # extract a 3D crop around the selected query point, the crop is 7x128x128 that is what DINOv3 expects.
            query_stack,
            query_point_tuple,
            size_z=7,
            size_y=128,
            size_x=128
        )

        query_embedding = compute_embedding(crop, model, device) # pas the crop through and get a query embedding
        query_embedding = query_embedding.float() # convert to float32
        query_embedding = query_embedding.cpu() # move the query embedding to cpu so it is on the same device as target_embeddings

        print(f"Query embedding shape: {query_embedding.shape}")

        print("\nFull search baseline")
        start_time = time.time() # start measuring runtime.
        best_point, highest_similarity_score = find_best_match(
            query_embedding=query_embedding, 
            target_embeddings=target_embeddings,
            candidate_points=candidate_points
        )
        runtime_seconds = time.time() - start_time # compute how many seconds the full search took.
        print(f"Number of candidates: {len(candidate_points)}")
        print(f"Best point: {best_point}, Best score: {highest_similarity_score:.4f}")
        print(f"Runtime: {runtime_seconds:.6f} seconds")
        results.append({
            "strategy": "full_search",
            "track_id": int(row["track_id"]),
            "radius": None,
            "number_of_candidates": len(candidate_points),
            "predicted_z,y,x": best_point,
            "target_z,y,x": target_point_tuple,
            "distance_to_target": float(distance_3d(best_point, target_point_tuple)),
            "highest_similarity_score": float(highest_similarity_score),
            "runtime_seconds": float(runtime_seconds)
        })

        print("\nLocal Search")
        for radius in SEARCH_RADII: # loop over every search radius we want to test.
            local_candidates, mask = filter_candidates_by_radius( # filter candidates using the current radius.
                candidate_points=candidate_points,
                center_point=query_point_tuple,
                radius=radius
            )
        
            local_embeddings = target_embeddings[mask] # use the same mask to select the corresponding embeddings
            print(f"\nRadius: {radius}") # print the current radius.
            print(f"Number of local candidates: {len(local_candidates)}")
            if len(local_candidates) == 0: # if no candidates are inside this radius, matching is impossible
                print("No candidates found inside this radius.") # print a message explaining that no search can be done.
                results.append({
                    "strategy": "local_search",
                    "track_id": int(row["track_id"]),
                    "radius": radius,
                    "number_of_candidates": 0,
                    "predicted_z,y,x": None,
                    "target_z,y,x": target_point_tuple,
                    "distance_to_target": None,
                    "highest_similarity_score": None,
                    "runtime_seconds": None
                })
                continue
            start_time = time.time() # start measuring runtime for local search.
            best_point, highest_similarity_score = find_best_match( # compare query embedding only with local candidate embeddings
                query_embedding=query_embedding,
                target_embeddings=local_embeddings,
                candidate_points=local_candidates
            )
            runtime_seconds = time.time() - start_time # stop measuing runtime for local search
            print(f"Best point: {best_point}")
            print(f"Best score: {highest_similarity_score:.4f}")
            print(f"Runtime: {runtime_seconds:.6f} seconds")
            results.append({
                "strategy": "local_search",
                "track_id": int(row["track_id"]),
                "radius": radius,
                "number_of_candidates": len(local_candidates),
                "predicted_z,y,x": best_point,
                "target_z,y,x": target_point_tuple,
                "distance_to_target": float(distance_3d(best_point, target_point_tuple)),
                "highest_similarity_score": float(highest_similarity_score),
                "runtime_seconds": float(runtime_seconds)
            })

    results_df = pd.DataFrame(results)
    print("\nSummary Table")
    print(results_df)
    results_df.to_csv(OUTPUT_RESULTS_PATH, index=False)
    print("Results saved")
    print("\nDone.")

if __name__ == "__main__":
    main()

