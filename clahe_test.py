import cv2
import numpy as np
from osgeo import gdal

def read_geotiff(image_path):
    """Read a GeoTIFF image and return the image data and metadata."""
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise ValueError("Could not open the image file.")

    # Read the image data
    band = dataset.GetRasterBand(1)
    image = band.ReadAsArray()

    # Get metadata
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    metadata = {
        'geotransform': geotransform,
        'projection': projection,
        'driver': dataset.GetDriver().ShortName,
    }

    dataset = None  # Close the dataset
    return image, metadata

def apply_clahe(image, clip_limit, tile_grid_size):
    """Apply CLAHE to enhance local contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(image)
    return enhanced_image


def write_geotiff(output_path, image, metadata):
    """Write an image to a GeoTIFF file with metadata."""
    driver = gdal.GetDriverByName(metadata['driver'])
    rows, cols = image.shape

    # Create the output dataset
    out_dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)
    if out_dataset is None:
        raise ValueError("Could not create the output image file.")

    # Set metadata
    out_dataset.SetGeoTransform(metadata['geotransform'])
    out_dataset.SetProjection(metadata['projection'])

    # Write the image data
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(image)
    out_band.FlushCache()

    out_dataset = None  # Close the dataset

def main():
    """Main function to read, process, and write the image."""

    # User-defined variables
    clahe_grid_size = 64
    clip_limit = 2.0
    input_image = "/media/benoit/geophoton5/Data/Total_Energie/Sentinel_2/superres/output/clahe/clip_2020_ori_3chan_25cm_b1.tif"  # Path to the input GeoTIFF image
    output_image = f"/media/benoit/geophoton5/Data/Total_Energie/Sentinel_2/superres/output/clahe/output/clip_2020_ori_3chan_25cm_b1_clahe_{clahe_grid_size}.tif"  # Path to save the output enhanced GeoTIFF image

    # Step 1: Read the input GeoTIFF image
    image, metadata = read_geotiff(input_image)

    # Step 2: Apply CLAHE for local contrast enhancement
    tile_grid_size = (clahe_grid_size, clahe_grid_size)
    enhanced_image = apply_clahe(image, clip_limit, tile_grid_size)

    # Step 3: Write the enhanced image to a new GeoTIFF file
    write_geotiff(output_image, enhanced_image, metadata)
    print(f"Enhanced image saved to {output_image}")


# Run the main function
if __name__ == "__main__":
    main()