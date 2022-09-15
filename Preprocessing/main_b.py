import arcpy
import glob
import os


# 3 - Crop image according to shp
def cropRasters():
    """
    Crop image according to shp
    """
    arcpy.CheckOutExtension('Spatial')

    # Get all shp file paths
    masks_path = r"F:\Preprocessing\shp"
    # Paths to files where .shp is stored
    masks = list()
    # Get UK folder names by region
    masks_zones = os.listdir(masks_path)
    # Alphabetical order
    masks_zones.sort()

    for index, zone_name in enumerate(masks_zones):
        zone_path = masks_path + "\\" + zone_name
        # Read all .shp files in the folder
        zone_masks = glob.glob(os.path.join(zone_path, "*.shp"))
        zone_masks.sort()
        masks.extend(zone_masks)
    print(masks)

    # The path to the folder where the data is entered
    input_data_path = r"F:\Preprocessing\data"
    # The path to the folder where the tifs are exported
    output_tifs_path = r"F:\Preprocessing\crop_copy1\5Bands_crop"

    years = os.listdir(input_data_path)
    for year in years:
        bands_path = input_data_path + "\\" + year + "\\5Bands"
        tifs_path = glob.glob(os.path.join(bands_path, "*.tif"))
        tifs_path.sort()
        print(tifs_path)
        for tif_path in tifs_path:
            # Cropping by .shp files
            for index, mask in enumerate(masks):
                out_extract = arcpy.sa.ExtractByMask(tif_path, mask)
                crop_tif_path = output_tifs_path + "\\" + tif_path.split(".")[0].rsplit("\\")[-1] + "_" + str(
                    index + 1) + ".tif"
                out_extract.save(crop_tif_path)


# 4 - Implement the function of "copy raster" in arcgis, and convert 32-bit depth to 8-bit.
def copyRaster():
    """
    Convert 32-bit depth to 8-bit
    """
    tifs_path = glob.glob(os.path.join("F:\\Preprocessing\\crop_copy1\\1Bands_crop", "*.tif"))
    for tif_path in tifs_path:
        tif_out_path = "F:\\Preprocessing\\crop_copy1\\1Bands_crop_copy\\"
        arcpy.CopyRaster_management(tif_path,
                                    tif_out_path + tif_path.split(".")[0].rsplit("\\")[-1] + ".tif", "DEFAULTS",
                                    "", "", "", "", "8_BIT_UNSIGNED", True)


if __name__ == '__main__':
    # crop
    # cropRasters()
    # 32-bit->8-bit
    copyRaster()
