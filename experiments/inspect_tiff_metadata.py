from pathlib import Path # helps build file paths
import tifffile # read TIFF image data and TIFF metadata

BASE_DIR = Path(__file__).resolve().parents[1] # main project folder
DATA_DIR = BASE_DIR / "data" / "time_data_labeled" # path to .tif files

A1_TIFF_PATH = DATA_DIR / "33648_A1_TS_dftcorr.tif" # points to A1 TIFF file
A5_TIFF_PATH = DATA_DIR / "33648_A5_TS_dftcorr.tif" # points to A5 TIFF file

def inspect_tiff(path): # starts a resuable method
    print("\n" + "=" * 80) # prints a separator
    print("File:")
    print(path) # prints file path

    with tifffile.TiffFile(path) as tif: # opens the TIFF file safely, with makes sure the files closes automatically afterward
        print("\nNumber of pages:") # TIFF files can contain many pages, a page is often one 2D image slice
        print(len(tif.pages))

        series = tif.series[0]
        print("\nSeries shape:") # tells the image dimensions
        print(series.shape)
        print("\nSeries axes:") # tells what each dimension means
        print(series.axes)

        print("\nImageJ metadata:")
        print(tif.imagej_metadata) # some microscope TIFFs store calibration here, we are looking for values like spacing, unit, finterval, pixel size information

        print("\nOME metadata exists?")
        print(tif.ome_metadata is not None) # some TIFFs store metadata in OME-XML format. This line only checks whether such metadata exists

        if tif.ome_metadata is not None:
            print("\nFirst 1000 characters of OME metadata:")
            print(tif.ome_metadata[:1000]) # if OME metadata exists, print only the first 1000 characters

        first_page = tif.pages[0] # inspect the first 2D page of the TIFF
        print("\nImportant TIFF tags from first page:") # this is just a label for the output

        useful_tags = [
            "ImageWidth",
            "ImageLength",
            "XResolution",
            "YResolution",
            "ResolutionUnit",
            "ImageDescription",
        ] # these are the metadata tags that may contain image size or pixel calibration

        for tag_name in useful_tags:
            if tag_name in first_page.tags:
                print(f"\n{tag_name}")
                print(first_page.tags[tag_name].value) # loop over useful tag names, if the tag exists, print its value

inspect_tiff(A1_TIFF_PATH)
inspect_tiff(A5_TIFF_PATH)







