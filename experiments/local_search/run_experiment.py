"""
run_experiment.py

Runs the Hypothesis 1 experiment.
For each track (query/target pair):
    - Compute the query embeddig from the raw image
    - Run full search (compare against all frame1 candidates)
    - Run euclidean, manhattan, and knn at each area percentage
    - Score every prediction against the human-labeled target point.

Results are saved to one CSV.
"""

import time
import pandas as pd
import torch
from experiments.local_search import (
    config,
    coordinates,
    data_loading,
    matching,
    scoring,
    search_strategies
)
from utils import crop_around_point, compute_embedding

# A blank row, used to visually  separate tracks in the CSV
def empty_result_row():
    return {column: None for column in config.OUTPUT_COLUMNS}

def run_full_search(query_embedding, target_embedding, candidate_points):
    start_time = time.perf_counter() # retrun a precise timestamp

    best_point_px = matching.find_best_match(
        query_embedding=query_embedding,
        target_embeddings=target_embedding,
        candidate_points=candidate_points
    )

    runtime_seconds = time.perf_counter() - start_time
    return best_point_px, runtime_seconds # return predicted point and how long it took

# Given a strategy's selected candidates, run the embedding match on jsut that subset and time it
def run_local_strategy(selection_result, query_embedding, target_embeddings):
    start_time = time.perf_counter()
    local_points = selection_result["local_points"] # the candidate points a particular strategy selected
    local_mask = selection_result["local_mask"] # True/False per candidate

    if len(local_points) == 0:
        runtime_seconds = time.perf_counter() - start_time
        return None, runtime_seconds
    
    local_embeddings = target_embeddings[local_mask] # pull the embeddings of only the selected candidates

    best_point_px = matching.find_best_match(
        query_embedding=query_embedding,
        target_embeddings=local_embeddings,
        candidate_points=local_points,
    )

    runtime_seconds = time.perf_counter() - start_time
    return best_point_px, runtime_seconds

# Assemble one complete results row
def build_result_row(strategy_name, track_id, query_frame, target_frame,
                     local_area_requested, number_of_candidates,
                     area_clipped_fraction, query_point_um,
                     predicted_point_px, target_point_um, runtime_seconds):

    # Convert the predicted point from pixels to micrometers
    predicted_point_um = (
        coordinates.point_pixel_to_um(predicted_point_px)
        if predicted_point_px is not None
        else None
    )

    # Score: distance from the prediction to the human-labeled target.
    distance = scoring.xy_distance_um(
        predicted_point_um=predicted_point_um,
        target_point_um=target_point_um,
    )

    return {
        "track_id": track_id,
        "strategy": strategy_name,
        "local_area_requested": local_area_requested,
        "number_of_candidates": number_of_candidates,
        "area_clipped_fraction": area_clipped_fraction,
        "query_point_z,y,x": coordinates.format_point(query_point_um),
        "predicted_point_z,y,x": coordinates.format_point(predicted_point_um),
        "target_point_z,y,x": coordinates.format_point(target_point_um),
        "runtime_seconds": runtime_seconds,
        "distance_to_human_label": distance,
        "query_frame": query_frame,
        "target_frame": target_frame,
    }

def main():
    print("Starting Hypothesis 1 local search experiment...")

    results = []   # every result row will be collected here

    # --- Load everything once, up front ---

    label_pairs = data_loading.load_label_pairs()
    print(f"Loaded {len(label_pairs)} query/target pairs.")

    all_candidate_points, all_target_embeddings, all_volume_indices = (
        data_loading.load_candidate_arrays()
    )

    a5_stack = data_loading.load_time_stack()
    print(f"Loaded A5 stack with shape {a5_stack.shape}.")

    device = data_loading.get_device()
    print(f"Using device: {device}")

    model = data_loading.load_model(device)
    print("Model loaded.")

    # --- Process each track ---

    for pair_number, row in label_pairs.iterrows():
        track_id = int(row["track_id"])
        query_frame = int(row["query_frame"])
        target_frame = int(row["target_frame"])

        # Query point (frame 0) in micrometers.
        query_x_um = float(row["query_x_um"])
        query_y_um = float(row["query_y_um"])
        query_z_slice = int(round(row["query_z_slice"]))
        query_point_um = (query_z_slice, query_y_um, query_x_um)

        # Human-labeled target point (frame 1) -- the ground truth.
        target_point_um = (
            int(round(row["target_z_slice"])),
            float(row["target_y_um"]),
            float(row["target_x_um"]),
        )

        # Convert the query point to pixels (to crop the image around it).
        query_point_px = coordinates.point_um_to_pixel(
            x_um=query_x_um, y_um=query_y_um, z_slice=query_z_slice
        )

        print(f"\nTrack {track_id} ({pair_number + 1}/{len(label_pairs)})")

        # Get the 3D volume for the query's frame, then crop around the
        # query point and compute its embedding.
        query_volume = a5_stack[query_frame]

        crop = crop_around_point(
            query_volume, query_point_px,
            size_z=config.CROP_SIZE_Z,
            size_y=config.CROP_SIZE_Y,
            size_x=config.CROP_SIZE_X,
        )

        with torch.no_grad():
            query_embedding = compute_embedding(crop, model, device)

        query_embedding = query_embedding.float().cpu().squeeze()

        # Get the candidates for the target frame (frame 1), and convert
        # their embeddings to a torch tensor for matching.
        candidate_points, target_embeddings_numpy = data_loading.candidates_for_frame(
            all_candidate_points, all_target_embeddings, all_volume_indices, target_frame
        )

        target_embeddings = torch.from_numpy(target_embeddings_numpy).float().cpu()

        # --- Full search: the baseline, uses every candidate ---

        full_best_point_px, full_runtime_seconds = run_full_search(
            query_embedding, target_embeddings, candidate_points
        )

        full_search_row = build_result_row(
            strategy_name="full_search",
            track_id=track_id,
            query_frame=query_frame,
            target_frame=target_frame,
            local_area_requested=None,
            number_of_candidates=len(candidate_points),
            area_clipped_fraction=None,
            query_point_um=query_point_um,
            predicted_point_px=full_best_point_px,
            target_point_um=target_point_um,
            runtime_seconds=full_runtime_seconds,
        )

        results.append(full_search_row)

        print(f"  full_search: {len(candidate_points)} candidates, "
              f"error = {full_search_row['distance_to_human_label']}")
        
        # --- Local strategies, at each area percentage ---

        for area_percentage in config.AREA_PERCENTAGES_TO_TEST:

            # Run the two area-based strategies (euclidean, manhattan).
            euclidean_selection = search_strategies.select_by_euclidean_area(
                candidate_points, query_point_px, area_percentage
            )
            manhattan_selection = search_strategies.select_by_manhattan_area(
                candidate_points, query_point_px, area_percentage
            )

            # Match knn to euclidean's candidate count, then run knn.
            matched_count = int(euclidean_selection["local_mask"].sum())
            knn_selection = search_strategies.select_by_knn_match_count(
                candidate_points, query_point_px, matched_count
            )

            selections = {
                "euclidean": euclidean_selection,
                "manhattan": manhattan_selection,
                "knn": knn_selection,
            }

            # For each strategy, run the match and record a result row.
            for strategy_name in config.LOCAL_STRATEGIES_TO_RUN:
                selection = selections[strategy_name]

                predicted_point_px, runtime_seconds = run_local_strategy(
                    selection, query_embedding, target_embeddings
                )

                result_row = build_result_row(
                    strategy_name=strategy_name,
                    track_id=track_id,
                    query_frame=query_frame,
                    target_frame=target_frame,
                    local_area_requested=area_percentage,
                    number_of_candidates=len(selection["local_points"]),
                    area_clipped_fraction=selection["area_clipped_fraction"],
                    query_point_um=query_point_um,
                    predicted_point_px=predicted_point_px,
                    target_point_um=target_point_um,
                    runtime_seconds=runtime_seconds,
                )

                results.append(result_row)

                print(f"  {strategy_name} @ {area_percentage}: "
                      f"{len(selection['local_points'])} candidates, "
                      f"error = {result_row['distance_to_human_label']}")

        # Optional blank row to separate tracks in the CSV.
        if config.ADD_BLANK_ROW_BETWEEN_TRACKS:
            results.append(empty_result_row())#

        # --- Save all results to one CSV ---

    results_df = pd.DataFrame(results, columns=config.OUTPUT_COLUMNS)

    config.H1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.OUTPUT_RESULTS_PATH, index=False)

    print(f"\nDone. Results saved to:\n{config.OUTPUT_RESULTS_PATH}")


if __name__ == "__main__":
    main()