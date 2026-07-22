"""
run_experiment.py

Runs the Hypothesis 1 experiment.
For each track (query/target pair):
    - Compute the query fingerprint from the raw image
    - Run full search (compare against all target frame candidates)
    - Run euclidean and manhattan at each area percentage
    - Score every prediction against the human-labeled target point.

Results are saved to one CSV.
"""

import time
import pandas as pd
import torch

from src.experiments.local_search import (config, search_strategies)
from src.experiments.shared import (convert_units, load_data, measure_accuracy)
from src.experiments.shared.find_best_match import find_best_match
from src.base.utils import crop_around_point, compute_embedding

# a blank row, used to visually  separate tracks in the CSV
def empty_result_row():
    return {column: None for column in config.OUTPUT_COLUMNS}

# the baseline: search every candidate in the target frame, and time it
def run_full_search(query_embedding, target_embedding, candidate_points):
    start_time = time.perf_counter() # a precise timestamp

    best_point_px = find_best_match(
        query_embedding=query_embedding,
        target_embeddings=target_embedding,
        candidate_points=candidate_points
    )

    runtime_seconds = time.perf_counter() - start_time
    return best_point_px, runtime_seconds # return predicted point and how long it took

# given a strategy's selected candidates, run the match on just that subset and time it
def run_local_strategy(strategy_selection, query_embedding, target_embeddings):
    start_time = time.perf_counter()
    local_points = strategy_selection["local_points"] # the candidate points a particular strategy selected
    local_mask = strategy_selection["local_mask"] # True/False per candidate

    if len(local_points) == 0:
        runtime_seconds = time.perf_counter() - start_time
        return None, runtime_seconds
    
    local_embeddings = target_embeddings[local_mask] # pull the embeddings of only the selected candidates

    best_point_px = find_best_match(
        query_embedding=query_embedding,
        target_embeddings=local_embeddings,
        candidate_points=local_points,
    )

    runtime_seconds = time.perf_counter() - start_time
    return best_point_px, runtime_seconds

# assemble one complete results row
def build_result_row(strategy_name, track_id, query_frame, target_frame,
                     local_area_requested, number_of_candidates,
                     area_clipped_fraction, query_point_um,
                     predicted_point_px, target_point_um, runtime_seconds, seconds_per_candidate):

    # convert the predicted point from pixels to micrometers
    predicted_point_um = (
        convert_units.point_pixel_to_um(predicted_point_px, config)
        if predicted_point_px is not None
        else None
    )

    # distance from the prediction to the human-labeled target, in micrometers.
    distance = measure_accuracy.xy_distance_um(
        predicted_point_um=predicted_point_um,
        target_point_um=target_point_um,
    )

    embedding_time_seconds = number_of_candidates * seconds_per_candidate
    total_time_seconds = runtime_seconds + embedding_time_seconds

    return {
        "track_id": track_id,
        "strategy": strategy_name,
        "local_area_requested": local_area_requested,
        "number_of_candidates": number_of_candidates,
        "query_point_z,y,x": convert_units.point_to_string(query_point_um),
        "predicted_point_z,y,x": convert_units.point_to_string(predicted_point_um),
        "target_point_z,y,x": convert_units.point_to_string(target_point_um),
        "runtime_seconds": runtime_seconds,
        "embedding_time_seconds": embedding_time_seconds,
        "total_time_seconds": total_time_seconds,
        "distance_to_human_label": round(distance, 2) if distance is not None else None,
        "area_clipped_fraction": area_clipped_fraction,
        "query_frame": query_frame,
        "target_frame": target_frame,
    }

def main():
    print("Starting Hypothesis 1 local search experiment...")
    results = []   # every result row will be collected here

    label_pairs = load_data.load_label_pairs(config)
    print(f"Loaded {len(label_pairs)} query/target pairs.")

    all_candidate_points, all_target_embeddings, all_volume_indices = (
        load_data.load_candidate_arrays(config)
    )

    image_stack = load_data.load_time_stack(config)
    print(f"Loaded image stack with shape {image_stack.shape}.")

    device = load_data.get_device()
    print(f"Using device: {device}")

    model = load_data.load_model(device, config)
    print("Model loaded.")

    seconds_per_candidate = load_data.load_embedding_rate(config)
    print(f"Embedding rate: {seconds_per_candidate:.6f} seconds per candidate")

    # process each track
    for pair_number, row in label_pairs.iterrows():
        track_id = int(row["track_id"])
        query_frame = int(row["query_frame"])
        target_frame = int(row["target_frame"])
        query_x_um = float(row["query_x_um"])
        query_y_um = float(row["query_y_um"])
        query_z_slice = int(round(row["query_z_slice"]))
        query_point_um = (query_z_slice, query_y_um, query_x_um)
        target_x_um = float(row["target_x_um"])
        target_y_um = float(row["target_y_um"])
        target_z_slice = int(round(row["target_z_slice"]))
        target_point_um = (target_z_slice, target_y_um, target_x_um)

        # Convert the query point to pixels (to crop the image around it).
        query_point_px = convert_units.point_um_to_pixel(
            x_um=query_x_um, y_um=query_y_um, z_slice=query_z_slice, config=config
        )

        print(f"\nTrack {track_id} ({pair_number + 1}/{len(label_pairs)})")

        # cut a crop around the click and run it through the model to get its fingerprint
        query_volume = image_stack[query_frame]

        crop = crop_around_point(
            query_volume, query_point_px,
            size_z=config.CROP_SIZE_Z,
            size_y=config.CROP_SIZE_Y,
            size_x=config.CROP_SIZE_X,
        )

        with torch.no_grad():
            query_embedding = compute_embedding(crop, model, device)

        query_embedding = query_embedding.float().cpu().squeeze()

        # narrow the candidates down to just the target frame's ones, and turn their fingerprints into a torch sensor for matching
        candidate_points, target_embeddings_numpy = load_data.candidates_for_frame(
            all_candidate_points, all_target_embeddings, all_volume_indices, target_frame
        )
        target_embeddings = torch.from_numpy(target_embeddings_numpy).float().cpu()

        # full search: the baseline, uses every candidate
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
            seconds_per_candidate=seconds_per_candidate,
        )

        results.append(full_search_row)

        print(f"full_search: {len(candidate_points)} candidates")
        
        # local strategies, at each area percentage
        for area_percentage in config.AREA_PERCENTAGES_TO_TEST:

            # run the two area-based strategies (euclidean, manhattan).
            euclidean_selection = search_strategies.select_by_euclidean_area(
                candidate_points, query_point_px, area_percentage
            )
            manhattan_selection = search_strategies.select_by_manhattan_area(
                candidate_points, query_point_px, area_percentage
            )

            selections = {
                "euclidean": euclidean_selection,
                "manhattan": manhattan_selection
            }

            # for each strategy, run the match and record a result row.
            for strategy_name in config.LOCAL_STRATEGIES_TO_RUN:
                strategy_selection = selections[strategy_name]

                predicted_point_px, runtime_seconds = run_local_strategy(
                    strategy_selection, query_embedding, target_embeddings
                )

                result_row = build_result_row(
                    strategy_name=strategy_name,
                    track_id=track_id,
                    query_frame=query_frame,
                    target_frame=target_frame,
                    local_area_requested=area_percentage,
                    number_of_candidates=len(strategy_selection["local_points"]),
                    area_clipped_fraction=strategy_selection["area_clipped_fraction"],
                    query_point_um=query_point_um,
                    predicted_point_px=predicted_point_px,
                    target_point_um=target_point_um,
                    runtime_seconds=runtime_seconds,
                    seconds_per_candidate=seconds_per_candidate,
                )

                results.append(result_row)

                print(f"  {strategy_name} @ {area_percentage}: "
                      f"{len(strategy_selection['local_points'])} candidates, ")

        # blank row to separate tracks in the CSV.
        if config.ADD_BLANK_ROW_BETWEEN_TRACKS:
            results.append(empty_result_row())

    # save all results to a CSV file
    results_df = pd.DataFrame(results, columns=config.OUTPUT_COLUMNS)
    config.H1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.OUTPUT_RESULTS_PATH, index=False)
    print(f"\nDone. Results saved to:\n{config.OUTPUT_RESULTS_PATH}")

if __name__ == "__main__":
    main()