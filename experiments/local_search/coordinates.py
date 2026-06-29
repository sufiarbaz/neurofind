"""
coordinates.py

Converts between the two coordinate systems used in the project:
    - micrometers (um): used by the human-label CSV (query/target points)
    - pixels: used by the TIFF image and candidate-point arrays

Everything the experiment reports is in micrometers; pixels are only used internally to index into the image and candidate arrays.
This file is the single place where the conversion happens, no no other file has to track which unit a value is in.
"""

from experiments.local_search import config

def point_um_to_pixel(x_um, y_um, z_slice):
    x_px = int(round(x_um * config.A5_PIXELS_PER_UM_X)) # um -> pixels on X
    y_px = int(round(y_um * config.A5_PIXELS_PER_UM_Y)) # um -> pixels on Y
    z_px = int(round(z_slice)) # z is already a slice index (not micrometers), so it is only rounded to a whole number

    return z_px, y_px, x_px

def point_pixel_to_um(point_px):
    z_slice = int(point_px[0]) # z stays the slice index
    y_um = round(float(point_px[1] / config.A5_PIXELS_PER_UM_Y), 8) # pixels -> um on Y
    x_um = round(float(point_px[2] / config.A5_PIXELS_PER_UM_X), 8) # pixels -> um on X

    return z_slice, y_um, x_um

def format_point(point):
    if point is None:
        return None
   
    return f"({point[0]}, {point[1]}, {point[2]})" # z, y, x in one string



