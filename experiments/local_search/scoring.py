"""
scoring.py

Scores a predicted against the human-labeled target point.

This is the source of "correctness" or "accuracy" in the experiment.
Every strategy is calculated by how close its predicted point is to the human label.
Distance is measured in the XY plane and in micrometers.
"""

import numpy as np # for the square-root / distance math

def xy_distance_um(predicted_point_um, target_point_um):
    if predicted_point_um is None or target_point_um is None:
        return None
    dy_um = predicted_point_um[1] - target_point_um[1] # difference in Y
    dx_um = predicted_point_um[2] - target_point_um[2] # difference in X

    return float(np.sqrt(dy_um ** 2 + dx_um ** 2)) # straight-line XY distance using Pythagorean theorem




