"""
grid.py

Lays a grid over the image, finds which square the click falls into and searches only that square instead of the whole image.
"""

import numpy as np  
from src.experiments.grid_search import config  # the image size come from here

# to find out on which square of the grid the query point lands in.
def get_cell_for_point(point_px, grid_size):
    # Pull out the y (down) and x (across) position.
    _, y_px, x_px = point_px

    # how many pixels wide and tall one square is.
    cell_height_px = config.STACK_HEIGHT_PX / grid_size
    cell_width_px = config.STACK_WIDTH_PX / grid_size

    # Figure out which square the point is in.
    row = int(y_px // cell_height_px) # y (down), row
    col = int(x_px // cell_width_px) # x (across), column

    # Tiny edge case: a point sitting exactly on the far right or very bottom edge
    # would count as "one square past the end", which doesn't exist. If that happens,
    # just put it in the last real square.
    if col == grid_size:
        col = grid_size - 1
    if row == grid_size:
        row = grid_size - 1

    return row, col

# keep only the candidates that are in the same square as the query point.
def select_candidates_in_query_cell(candidate_points, query_point_px, grid_size):
    # firstly, which square is the query in?
    query_row, query_col = get_cell_for_point(query_point_px, grid_size)

    # get the cell height and width
    cell_height_px = config.STACK_HEIGHT_PX / grid_size
    cell_width_px = config.STACK_WIDTH_PX / grid_size

    # pull out the x (across) and y (down) position of every candidate dot at once.
    candidate_y = candidate_points[:, 1]
    candidate_x = candidate_points[:, 2]

    # find out which square each candidate point is in.
    candidate_rows = (candidate_y // cell_height_px).astype(int)
    candidate_cols = (candidate_x // cell_width_px).astype(int)

    # any candidate point exactly on the far edge gets pulled back into the last real square.
    candidate_cols[candidate_cols == grid_size] = grid_size - 1
    candidate_rows[candidate_rows == grid_size] = grid_size - 1

    # mark every candidate point that is in the same square as the query point:
    mask = (candidate_rows == query_row) & (candidate_cols == query_col)

    # Keep only the marked candidate points.
    cell_points = candidate_points[mask]

    # return the kept candidate points plus the true/false list and a couple of useful notes.
    return {
        "cell_points": cell_points,
        "cell_mask": mask,
        "grid_size": grid_size,
        "query_cell": (query_row, query_col),
    }

# check whether the query and target points both in the same cell
def is_target_in_query_cell(target_point_px, query_point_px, grid_size):
    # find which square each one is in, then just check if they're the same square.
    target_cell = get_cell_for_point(target_point_px, grid_size)
    query_cell = get_cell_for_point(query_point_px, grid_size)
    return target_cell == query_cell

