"""
search_strategies.py

Different ways of doing a local search around a query point.

The strategies:
    - euclidean: candidates inside a circle around the query point.
    - manhattan: candidates inside a diamond around the query point.
    - knn: the N nearest candidates, no fixed shape.

For euclidean and manhattan, the search region is sized to a fixed percentage of the total image area (25% and 40%).
The two shapes use the same area, so they end up with slightly different numbers of candidates.
knn uses the same candidate count as euclidean.

Points near the image edge can't fit the full area inside the image, so each strategy also reports how much of its fell outside.
"""

import numpy as np
from experiments.local_search import config

# For every candidate, how far it is from the query point in X and Y, in micrometers
def _xy_offset_um(candidate_points, center_point_px):
    dy_px = candidate_points[:, 1] - center_point_px[1] # Y pixel offset
    dx_px = candidate_points[:, 2] - center_point_px[2] # X pixel offset

    dy_um = dy_px / config.A5_PIXELS_PER_UM_Y # Y offset in um
    dx_um = dx_px / config.A5_PIXELS_PER_UM_X # X pixel offset

    return dy_um, dx_um

# Total image of the area in square micrometers
def total_imaged_area_um2():
    width_um = config.A5_STACK_WIDTH_PX / config.A5_PIXELS_PER_UM_X # image width in um
    height_um = config.A5_STACK_HEIGHT_PX / config.A5_PIXELS_PER_UM_Y # image height in um

    return width_um * height_um # area = width * height

# Need the radius of the circle that has exactly that area.
def _euclidean_radius_for_area(target_area_um2):
    return float(np.sqrt(target_area_um2 / np.pi)) 

# Need the radius of the diamond (square rotated 45 degrees) that has exactly that area
def _manhattan_radius_for_area(target_area_um2):
    return float(np.sqrt(target_area_um2 / 2.0))

# Roughly how much of the circular search region falls outside the image, for a query point near the edge
def _clipped_area_fraction(center_point_px, radius_um):
    image_width_um = config.A5_STACK_WIDTH_PX / config.A5_PIXELS_PER_UM_X
    image_height_um = config.A5_STACK_HEIGHT_PX / config.A5_PIXELS_PER_UM_Y

    _, y_px, x_px = center_point_px
    x_um = x_px / config.A5_PIXELS_PER_UM_X
    y_um = y_px / config.A5_PIXELS_PER_UM_Y 

    # Distance from the query point to the nearest image edge
    distance_to_nearest_edge_um = min(
        x_um, # distance to the left edge
        image_width_um - x_um, # distance to the right edge
        y_um, # distance to the top edge
        image_height_um - y_um # distance to the bottom edge
    )
    
    if distance_to_nearest_edge_um >= radius_um:
        return 0.0 # whole region fits, nothing clipped
    
    if distance_to_nearest_edge_um <= 0:
        return 1.0
    
    # Otherwise, part of the region overshoots the nearest edge.
    overshoot = radius_um - distance_to_nearest_edge_um
    return round(float(min(overshoot / radius_um, 1.0)), 1)

# Euclidean strategy
def select_by_euclidean_area(candidate_points, center_point_px, area_percentage):
    # Turn the requested percentage into a target area, then into a radius
    target_area_um2 = area_percentage * total_imaged_area_um2()
    radius_um = _euclidean_radius_for_area(target_area_um2)

    # Straight line distance from the query to every candidate (um)
    dy_um, dx_um = _xy_offset_um(candidate_points, center_point_px)
    distances_um = np.sqrt(dy_um**2 + dx_um**2)

    # A candidate is "in" if it lies within the circle radius
    mask = distances_um <= radius_um
    local_points = candidate_points[mask]

    clipped_fraction = _clipped_area_fraction(center_point_px, radius_um)

    return {
        "local_points": local_points,
        "local_mask": mask,
        "search_radius_um": radius_um,
        "area_clipped_fraction": clipped_fraction
    }

# Manhattan strategy
def select_by_manhattan_area(candidate_points, center_point_px, area_percentage):
    # Turn the requested percentage into a target area, then into a radius
    target_area_um2 = area_percentage * total_imaged_area_um2()
    radius_um = _manhattan_radius_for_area(target_area_um2)

    # Manhattan distance from every query to every candidate (um)
    dy_um, dx_um = _xy_offset_um(candidate_points, center_point_px)
    distances_um = np.abs(dy_um) + np.abs(dx_um)

    # A candidate is "in" if it lies within the diamond
    mask = distances_um <= radius_um
    local_points = candidate_points[mask]

    clipped_fraction = _clipped_area_fraction(center_point_px, radius_um)

    return {
        "local_points": local_points,
        "local_mask": mask,
        "search_radius_um": radius_um,
        "area_clipped_fraction": clipped_fraction
    }

# kNN strategy
def select_by_knn_match_count(candidate_points, center_point_px, matched_count):
    # Straight line distance from the query to every candidate (um)
    dy_um, dx_um = _xy_offset_um(candidate_points, center_point_px)
    distances_um = np.sqrt(dy_um**2 + dx_um**2)

    # Don't ask for more candidates than exist
    budget = min(matched_count, len(candidate_points))

    # Edge case: nothing to select, return an empty selection (an all False mask)
    if budget <= 0:
        mask = np.zeros(len(candidate_points), dtype=bool)
        return {
            "local_points": candidate_points[mask],
            "local_mask": mask,
            "search_radius_um": None,
            "area_clipped_fraction": None,
        }
    
    # Indices of the "budget" closest candidates
    nearest_indices = np.argpartition(distances_um, budget - 1)[:budget]

    # Build a full-length True/False mask marking those nearest points
    mask = np.zeros(len(candidate_points), dtype=bool)
    mask[nearest_indices] = True

    local_points = candidate_points[mask]

    return {
        "local_points": local_points,
        "local_mask": mask,
        "search_radius_um": None, # knn has no single radius
        "area_clipped_fraction": None, # knn is count-based, not area-based
    }

# Loop up a strategy by its name
AREA_BASED_STRATEGY_REGISTRY = {
    "euclidean": select_by_euclidean_area,
    "manhattan": select_by_manhattan_area,
}

COUNT_BASED_STRATEGY_REGISTRY = {
    "knn": select_by_knn_match_count,
}