"""
read_stack_info.py

A small helper you run once when you switch to a new stack.

It opens the TIFF named in stack_settings.py and reads the numbers that file needs:
  - the calibration (how many pixels make one micrometer), from the file's metadata
  - the image width and height in pixels, from the image's shape
"""

from pathlib import Path  # to build file paths that work on any machine
import tifffile  # for reading the TIFF

from src.base import stack_settings  # the stack we are looking at comes from here


# path to root directory
BASE_DIR = Path(__file__).resolve().parents[2]

STACK_PATH = BASE_DIR / "data" / "time_data_labeled" / stack_settings.STACK_FILENAME

# read the pixels-per-micrometer calibration from a TIFF file's metadata
def read_pixels_per_um(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]  # the first image page holds the resolution tags

        x_top, x_bottom = page.tags["XResolution"].value  # the fraction, across
        y_top, y_bottom = page.tags["YResolution"].value  # the fraction, down

        pixels_per_um_x = x_top / x_bottom
        pixels_per_um_y = y_top / y_bottom

    return pixels_per_um_x, pixels_per_um_y

# read the image width and height in pixels, from the images's shape
def read_width_and_height(tiff_path):
    stack = tifffile.imread(tiff_path)

    width_px = stack.shape[-1]   # last axis is x (width)
    height_px = stack.shape[-2]  # second-to-last axis is y (height)

    return width_px, height_px


def main():
    if not STACK_PATH.is_file():
        raise FileNotFoundError(
            f"\n\nThe TIFF was not found:\n"
            f"  {STACK_PATH}\n\n"
            f"Check STACK_FILENAME in src/base/stack_settings.py, and that the file "
            f"is in data/time_data_labeled/\n"
        )

    pixels_per_um_x, pixels_per_um_y = read_pixels_per_um(STACK_PATH)
    width_px, height_px = read_width_and_height(STACK_PATH)

    # Print everything, ready to paste into stack_settings.py
    print(f"Stack: {stack_settings.STACK_NAME}")
    print(f"File : {STACK_PATH.name}")
    print()
    print("Paste these into src/base/stack_settings.py:")
    print()
    print(f"  PIXELS_PER_UM_X = {pixels_per_um_x}")
    print(f"  PIXELS_PER_UM_Y = {pixels_per_um_y}")
    print(f"  STACK_WIDTH_PX = {width_px}")
    print(f"  STACK_HEIGHT_PX = {height_px}")


if __name__ == "__main__":
    main()
