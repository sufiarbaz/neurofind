"""
run_experiment.py

For each track (query/target pair):
    - Compute the query fingerprint from the raw image
    - Run full search (compare against every candidate in the target frame), the baseline
    - Run the grid search at each grid size (2, 4, 8, 16, 32)
    - Check whether the human's label even landed in the searched square
    - Score every prediction against the human-labeled target point

Results are saved to one CSV.
"""

import time
import pandas as pd
import torch

from src.experiments.grid_search import (config, grid)
from src.experiments.shared import (convert_units, load_data, measure_accuracy,)
from src.experiments.shared.find_best_match import find_best_match
from src.base.utils import crop_around_point, compute_embedding


# a blank row, used to visually separate tracks in the CSV
def empty_result_row():
    return {column: None for column in config.OUTPUT_COLUMNS}

# the baseline: search every candidate in the target frame, and time it
def run_full_search(query_embedding, target_embeddings, candidate_points):
    start_time = time.perf_counter()  # a precise timestamp

    best_point_px = find_best_match(
        query_embedding=query_embedding,
        target_embeddings=target_embeddings,
        candidate_points=candidate_points,
    )

    runtime_seconds = time.perf_counter() - start_time
    return best_point_px, runtime_seconds  # the predicted point and how long it took


# the grid search: match only against the candidates in the click's square, and time it
def run_grid_search(cell_selection, query_embedding, target_embeddings):
    start_time = time.perf_counter()

    cell_points = cell_selection["cell_points"]  # the candidate points inside the query square
    cell_mask = cell_selection["cell_mask"]  # marks which of all candidates selected points are

    # if the cell doesn't contain any candidate point
    if len(cell_points) == 0:
        runtime_seconds = time.perf_counter() - start_time
        return None, runtime_seconds

    # use the same mask to pull the fingerprints of exactly selected candidate points, so the positions and the fingerprints stay lined up.
    cell_embeddings = target_embeddings[cell_mask]

    best_point_px = find_best_match(
        query_embedding=query_embedding,
        target_embeddings=cell_embeddings,
        candidate_points=cell_points,
    )

    runtime_seconds = time.perf_counter() - start_time
    return best_point_px, runtime_seconds


# assemble one complete results row
def build_result_row(strategy_name, track_id, query_frame, target_frame,
                     grid_size, number_of_candidates, target_in_query_cell,
                     query_point_um, predicted_point_px, target_point_um,
                     runtime_seconds, seconds_per_candidate):

    # Convert the predicted point from pixels back to micrometers
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
        "grid_size": grid_size,
        "number_of_candidates": number_of_candidates,
        "query_point_z,y,x": convert_units.point_to_string(query_point_um),
        "predicted_point_z,y,x": convert_units.point_to_string(predicted_point_um),
        "target_point_z,y,x": convert_units.point_to_string(target_point_um),
        "runtime_seconds": runtime_seconds,
        "embedding_time_seconds": embedding_time_seconds,
        "total_time_seconds": total_time_seconds,
        "target_in_click_cell": target_in_query_cell,
        "distance_to_human_label": round(distance, 2) if distance is not None else None,
        "query_frame": query_frame,
        "target_frame": target_frame,
    }


def main():
    print("Starting Hypothesis 2 grid search experiment...")
    results = []  # every result row will be collected here

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

        # convert query and target points to pixels.
        query_point_px = convert_units.point_um_to_pixel(
            x_um=query_x_um, y_um=query_y_um, z_slice=query_z_slice, config=config
        )
        target_point_px = convert_units.point_um_to_pixel(
            x_um=target_x_um, y_um=target_y_um, z_slice=target_z_slice, config=config
        )

        print(f"\nTrack {track_id} ({pair_number + 1}/{len(label_pairs)})")

        # cut a crop around the click and run it through the model to get its fingerprint.
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

        # narrow the candidates down to just the target frame's ones, and turn their fingerprints into a torch tensor for matching.
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
            grid_size=None,  # full search has no grid
            number_of_candidates=len(candidate_points),
            target_in_query_cell=None,  # full search has no square, so this does not apply
            query_point_um=query_point_um,
            predicted_point_px=full_best_point_px,
            target_point_um=target_point_um,
            runtime_seconds=full_runtime_seconds,
            seconds_per_candidate=seconds_per_candidate
        )

        results.append(full_search_row)
        print(f"full_search: {len(candidate_points)} candidates")

        # grid search, at each grid size
        for grid_size in config.GRID_SIZES_TO_TEST:

            # pick out only the candidates in the same square as the click.
            cell_selection = grid.select_candidates_in_query_cell(
                candidate_points, query_point_px, grid_size
            )

            # check whether query and target points are in the same cell
            target_in_query_cell = grid.is_target_in_query_cell(
                target_point_px, query_point_px, grid_size
            )

            predicted_point_px, runtime_seconds = run_grid_search(
                cell_selection, query_embedding, target_embeddings
            )

            result_row = build_result_row(
                strategy_name="grid",
                track_id=track_id,
                query_frame=query_frame,
                target_frame=target_frame,
                grid_size=f"{grid_size} x {grid_size}",
                number_of_candidates=len(cell_selection["cell_points"]),
                target_in_query_cell=target_in_query_cell,
                query_point_um=query_point_um,
                predicted_point_px=predicted_point_px,
                target_point_um=target_point_um,
                runtime_seconds=runtime_seconds,
                seconds_per_candidate=seconds_per_candidate
            )

            results.append(result_row)
            print(f"  grid {grid_size}x{grid_size}: "
                  f"{len(cell_selection['cell_points'])} candidates")

        # blank row to separate tracks in the CSV.
        if config.ADD_BLANK_ROW_BETWEEN_TRACKS:
            results.append(empty_result_row())

    # save all results to a CSV file
    results_df = pd.DataFrame(results, columns=config.OUTPUT_COLUMNS)
    config.H2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.OUTPUT_RESULTS_PATH, index=False)
    print(f"\nDone. Results saved to:\n{config.OUTPUT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
