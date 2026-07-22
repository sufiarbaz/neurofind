"""
stack_settings.py

This file says which stack and which frame pairs the experiment works on.
So to run on a different stack, or a different pair of frames, change the required variables.
"""

STACK_NAME = "A1" # short name for the stack, e.g., "A5" or "A1"
STACK_FILENAME = "33648_A1_TS_dftcorr.tif" # place this raw TIFF in: data/time_data_labeled/
RAW_LABELS_FILENAME = "33648_A1_TS_dftcorr_01.csv" # the raw human labels, in: data/time_data_labeled/

QUERY_FRAME = 4 # the frame where the query comes from
TARGET_FRAME = 5 # the frame we search for the match

# pixels per micrometer and stack width & height. If you don't know these values, run src/base/read_stack_info.py on the TIFF.
PIXELS_PER_UM_X = 7.23589         
PIXELS_PER_UM_Y = 7.23589
STACK_WIDTH_PX = 512 
STACK_HEIGHT_PX = 512

# crop size taken around each point before making its fingerprint (the embedding)
CROP_SIZE_Z = 7
CROP_SIZE_Y = 128
CROP_SIZE_X = 128