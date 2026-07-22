"""
convert_units.py

This file changes a point between two ways of measuring position:
    - micrometers (um): the real-world distance used in the human-label CSV
    - pixels: the whole-number position used by the image
"""

# change a point from micrometers into pixels
def point_um_to_pixel(x_um, y_um, z_slice, config):
    # multiply by pixels-per-micrometer to get pixels, then round to a whole number
    x_px = int(round(x_um * config.PIXELS_PER_UM_X))
    y_px = int(round(y_um * config.PIXELS_PER_UM_Y))
    z_px = int(round(z_slice)) 
    return z_px, y_px, x_px

# change a point from pixels back to micrometers
def point_pixel_to_um(point_px, config):
    z_slice= int(point_px[0]) # z stays exactly as it is
    # divide pixels-per-micrometers to get micrometers.
    y_um = round(float(point_px[1] / config.PIXELS_PER_UM_Y), 8)
    x_um = round(float(point_px[2] / config.PIXELS_PER_UM_X), 8)
    return z_slice, y_um, x_um

# turn three numbers into one readable string for a cell of the result file
def point_to_string(point):
    if point is None:
        return None
    return f"({point[0]}, {round(point[1], 2)}, {round(point[2], 2)})" # z, y, x in one string


