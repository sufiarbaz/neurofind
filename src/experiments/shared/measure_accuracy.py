"""
measure_accuracy.py

This file measures accuracy, how far the predicted point is from the human-label.
The distance is measured on the flat image (X and Y only) and given in micrometers.
"""

import numpy as np # for the square-root used in the distance

# measure the straight line distance between the predicted point and the target point.
def xy_distance_um(predicted_point_um, target_point_um):
    if predicted_point_um is None or target_point_um is None:
        return None
    
    dy = predicted_point_um[1] - target_point_um[1]
    dx = predicted_point_um[2] - target_point_um[2]

    return float(np.sqrt(dy ** 2 + dx ** 2))