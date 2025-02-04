import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal_array import BandReadAsArray, CopyDatasetInfo, BandWriteArray
import sys
import glob
from datetime import datetime
import os.path


# This function improves the resolution of a 1-channel or 3-channel image by a factor of 4
# using the pre-trained EDSR4 deep learning model through openCV.
# Author: Benoit St-Onge.
# Created 14 June 2021.
# Updated 3 February 2022.


def improve_resol(image_file_name, output_image_file_name, mult_factor):

    # Read image.
    if os.path.isfile(image_file_name):
        IMAGE = gdal.Open(image_file_name, gdal.GA_ReadOnly)  # Open input image (must be 1 band TIFF file).
    else:
        print('Image file', image_file_name, 'not found')
        print('Process halted')
        sys.exit(1)

    image_gt = IMAGE.GetGeoTransform()
    n_bands = int(IMAGE.RasterCount)
    if n_bands != 1 and n_bands != 3:
        print('The input image must contain either 1 or 3 channels. This images contains', n_bands, 'channels.')
        print('Processing stopped.')
        sys.exit(0)
    image_array = []  # Original array.
    im_arr_cmp = []  # Array compressed to 0-255 (Byte)
    image_min = []  # Minimum value of image.
    image_max = []  # Maximum value of image.
    for b in range(n_bands):
        image_band = IMAGE.GetRasterBand(b + 1)
        image_array.append(BandReadAsArray(image_band))
        image_min.append(np.min(image_array))
        image_max.append(np.max(image_array))
        # Compress to byte pixel depth.
        im_arr_cmp.append(((image_array[b] - image_min[b]) / (image_max[b] - image_min[b]) * mult_factor).astype(int))

    if n_bands == 3:
        input_image = np.stack((im_arr_cmp[0], im_arr_cmp[1], im_arr_cmp[2]), axis=2)  # Stack 3 different bands.
    else:  # Single band case.
        input_image = np.stack((im_arr_cmp[0], im_arr_cmp[0], im_arr_cmp[0]), axis=2)  # Stack the same band thrice.

    # Load super-resolution pre-trained model.
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = "EDSR_x4.pb"  # Should reside in the same folder as the script.
    sr.readModel(model_path)

    # Apply super-resolution model.
    sr.setModel("edsr", 4)  # Sets model to increases resolution by 4.
    image_super_res = sr.upsample(input_image)

    # Write results to georeferenced output image.
    driver = gdal.GetDriverByName("Gtiff")
    OUT_IMAGE = driver.Create(output_image_file_name, IMAGE.RasterXSize * 4, IMAGE.RasterYSize * 4, n_bands,
                              gdal.GDT_Float32)
    if OUT_IMAGE is None:
        print('Cannot create output file ', output_image_file_name)
        sys.exit(1)
    CopyDatasetInfo(IMAGE, OUT_IMAGE)
    OUT_IMAGE.SetGeoTransform([image_gt[0], image_gt[1] / 4.0, 0, image_gt[3], 0, image_gt[5] / 4.0])

    for b in range(n_bands):
        output_array = image_super_res[:, :, b]
        final_output_array = output_array / mult_factor * (image_max[b] - image_min[b]) + image_min[
            b]  # Decompres results.
        arrayOut = OUT_IMAGE.GetRasterBand(b + 1)
        arrayOut.SetNoDataValue(-99.0)
        BandWriteArray(arrayOut, final_output_array)

    OUT_IMAGE = None


# MAIN PROGRAM

startTime = datetime.now()

# Input/output parameters.

input_folder = '/media/benoit/geophoton5/Data/Total_Energie/Sentinel_2/superres/'
output_folder = '/media/benoit/geophoton5/Data/Total_Energie/Sentinel_2/superres/output/'
output_suffix = '_25cm'
extension = '.tif'

mult_factor = 255.0  # Multiplicative factor for compressing and decompressing the image values
                     # (the DnnSuperResImpl_create() function only accepts byte pixel depth).

i = 0
for f in glob.glob(input_folder + '*.tif'):
    image_root_name = os.path.basename(f)
    image_root_name = image_root_name.strip('.tif)')
    output_image_file_name = output_folder + image_root_name + output_suffix + extension
    i += 1
    print('Processing image:', i, f)
    improve_resol(f, output_image_file_name, mult_factor)

print('Processing completed. Total processing time: ', datetime.now() - startTime)
