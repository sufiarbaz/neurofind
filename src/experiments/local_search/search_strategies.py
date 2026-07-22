"""
search_strategies.py

Different ways of doing a local search around a query point.

The strategies:
    - euclidean: candidates inside a circle around the query point.
    - manhattan: candidates inside a diamond around the query point.

For euclidean and manhattan, the search region is sized to a fixed percentage of the total image area (25% and 40%).
The two shapes use the same area, so they end up with slightly different numbers of candidates.

Points near the image edge can't fit the full area inside the image, so each strategy also reports how much of its fell outside.
"""

import numpy as np  # for the distance maths across all candidates at once
from src.experiments.local_search import config  # the image size and the pixels-per-micrometer numbers come from here

# for every candidate, how far it is from the query point
def _distance_from_query_um(candidate_points, query_point_px):
    dy_px = candidate_points[:, 1] - query_point_px[1] # how far down
    dx_px = candidate_points[:, 2] - query_point_px[2] # how far across

    dy_um = dy_px / config.PIXELS_PER_UM_Y # how far down, in micrometers
    dx_um = dx_px / config.PIXELS_PER_UM_X # how far across, in micrometers

    return dy_um, dx_um

# total image of the area in square micrometers
def total_imaged_area_um2():
    width_um = config.STACK_WIDTH_PX / config.PIXELS_PER_UM_X # image width in um
    height_um = config.STACK_HEIGHT_PX / config.PIXELS_PER_UM_Y # image height in um

    return width_um * height_um # area = width * height

# need the radius of the circle that has exactly that area.
def _euclidean_radius_for_area(target_area_um2):
    return float(np.sqrt(target_area_um2 / np.pi)) 

# need the radius of the diamond (square rotated 45 degrees) that has exactly that area
def _manhattan_radius_for_area(target_area_um2):
    return float(np.sqrt(target_area_um2 / 2.0))

# roughly how much of the circular search region falls outside the image, for a query point near the edge
def _clipped_area_fraction(query_point_px, radius_um):
    image_width_um = config.STACK_WIDTH_PX / config.PIXELS_PER_UM_X
    image_height_um = config.STACK_HEIGHT_PX / config.PIXELS_PER_UM_Y

    _, y_px, x_px = query_point_px
    x_um = x_px / config.PIXELS_PER_UM_X
    y_um = y_px / config.PIXELS_PER_UM_Y 

    # distance from the query point to the nearest image edge
    distance_to_nearest_edge_um = min(
        x_um, # distance to the left edge
        image_width_um - x_um, # distance to the right edge
        y_um, # distance to the top edge
        image_height_um - y_um # distance to the bottom edge
    )
    
    if distance_to_nearest_edge_um >= radius_um:
        return 0.0 # whole region fits, nothing clipped
    
    if distance_to_nearest_edge_um <= 0:
        return 1.0 # fully clipped, the query is on or past the edge
    
    # otherwise, part of the region overshoots the nearest edge.
    overshoot = radius_um - distance_to_nearest_edge_um
    return round(float(min(overshoot / radius_um, 1.0)), 1)

# euclidean strategy
def select_by_euclidean_area(candidate_points, query_point_px, area_percentage):
    # turn the requested percentage into a target area, then into a radius
    target_area_um2 = area_percentage * total_imaged_area_um2()
    radius_um = _euclidean_radius_for_area(target_area_um2)

    # straight line distance from the query to every candidate (um)
    dy_um, dx_um = _distance_from_query_um(candidate_points, query_point_px)
    distances_um = np.sqrt(dy_um**2 + dx_um**2)

    # a candidate is "in" if it lies within the circle radius
    mask = distances_um <= radius_um
    local_points = candidate_points[mask]

    clipped_fraction = _clipped_area_fraction(query_point_px, radius_um)

    return {
        "local_points": local_points,
        "local_mask": mask,
        "search_radius_um": radius_um,
        "area_clipped_fraction": clipped_fraction
    }

# Manhattan strategy
def select_by_manhattan_area(candidate_points, query_point_px, area_percentage):
    # turn the requested percentage into a target area, then into a radius
    target_area_um2 = area_percentage * total_imaged_area_um2()
    radius_um = _manhattan_radius_for_area(target_area_um2)

    # manhattan distance from every query to every candidate (um)
    dy_um, dx_um = _distance_from_query_um(candidate_points, query_point_px)
    distances_um = np.abs(dy_um) + np.abs(dx_um)

    # a candidate is "in" if it lies within the diamond
    mask = distances_um <= radius_um
    local_points = candidate_points[mask]

    clipped_fraction = _clipped_area_fraction(query_point_px, radius_um)

    return {
        "local_points": local_points,
        "local_mask": mask,
        "search_radius_um": radius_um,
        "area_clipped_fraction": clipped_fraction
    }