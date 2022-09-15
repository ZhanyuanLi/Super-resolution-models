# Preprocessing is executed first for .m files,
# and then sequentially according to steps 1, 2, 3, etc.
# It is divided into two .py files because of the library incompatibility problem.
# Processed standard input data, the first band being rainfall data.
"""
Important process file structure during preprocessing:
data/
    └── 2002
    └── 2003
    └── 2004
    ...
    └── 2021
          └── 1Bands
          ├── 4Bands
          ├── 5Bands
          ├── hurs
          ├── pv
          ├── rainfall
          └── tas
"""
import os
import torch
from osgeo import gdal
from osgeo.gdalconst import GDT_Float64
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pandas.core.frame import DataFrame


class _const:
    """
    Define a constant class to implement the function of constants.
    Block constants from being modified or deleted. See reference 1.
    """

    def __init__(self):
        pass

    class ConstError(TypeError):
        pass

    # The '__setattr__' object method of '_const' determines whether the attribute 'name' exists for that object
    def __setattr__(self, name, value):
        if name in self.__dict__:
            # If present, the custom exception ConstError is thrown, otherwise the attribute is created.
            raise self.ConstError("Can't rebind const (%s)" % name)
        self.__dict__[name] = value

    # Avoid constant deletion: del const.BUFFER_SIZE
    def __delattr__(self, name):
        if self.__dict__:
            raise self.ConstError("Can't unbind const (%s)" % name)
        raise AttributeError("const instance has no attribute '%s'" % name)


# Class instantiation
const = _const()
const.DEM_PATH = os.path.join(os.getcwd(), 'dem', 'dem.tif')
const.DATA_PATH = os.path.join(os.getcwd(), 'data')
# shp
const.SHP_PATH = os.path.join(os.getcwd(), 'shp')
# Path to the folder where the cropped tifs are located
const.CROPPED_TIFS_PATH = "F:\\Preprocessing\\tifs"
const.COPY_TIFS_PATH = "F:\\Preprocessing\\rainfall_crop_copy"
# const.COPY_TIFS_PATH = "F:\\"
const.JPG_PATH = "F:\\Preprocessing\\rainfall_crop_copy_jpgs"


# 1 - Get the address of the image to be composited
def getImageList(year_path, dem_path):
    """
    Obtain the image address to be synthesized. Rainfall, temperature and humidity.
    """

    # Store all paths
    dirs_paths = list()
    # Get the year folder name
    years = os.listdir(year_path)
    years.sort()

    # Obtain the file name of rainfall, temperature and humidity in each year
    for year in years:
        year_dirs = list()  # year
        rainfall_dirs = os.listdir(year_path + "\\" + year + "\\" + "rainfall")
        # File name sorting
        rainfall_dirs.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        tas_dirs = os.listdir(year_path + "\\" + year + "\\" + "tas")
        tas_dirs.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        hurs_dirs = os.listdir(year_path + "\\" + year + "\\" + "hurs")
        hurs_dirs.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        pv_dirs = os.listdir(year_path + "\\" + year + "\\" + "pv")
        pv_dirs.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))

        for rainfall, tas, hurs, pv in zip(rainfall_dirs, tas_dirs, hurs_dirs, pv_dirs):
            # for rainfall, hurs in zip(rainfall_dirs, hurs_dirs):
            rainfall_path = year_path + "\\" + year + "\\" + "rainfall" + "\\" + rainfall
            tas_path = year_path + "\\" + year + "\\" + "tas" + "\\" + tas
            hurs_path = year_path + "\\" + year + "\\" + "hurs" + "\\" + hurs
            pv_path = year_path + "\\" + year + "\\" + "pv" + "\\" + pv

            rdth_paths = [rainfall_path, dem_path, tas_path, hurs_path, pv_path]  # Absolute path information of 5 bands
            # rdth_paths = [rainfall_path, dem_path, tas_path, hurs_path]  # Absolute path information of 4 bands
            # rdth_paths = [rainfall_path, dem_path, hurs_path]  # Absolute path information of 3 bands
            year_dirs.append(rdth_paths)

        dirs_paths.append(year_dirs)

    print(dirs_paths)

    return dirs_paths


'''
def getImageList(year_path, dem_path):
    """
    Get the address of the image to be composited. Just the rainfall data.
    """
    # All data
    dirs_paths = list()
    # Get year folder name
    years = os.listdir(year_path)
    years.sort()

    # Get the file name of rainfall under each year
    for year in years:
        year_dirs = list()  # Year
        rainfall_dirs = os.listdir(year_path + "\\" + year + "\\" + "rainfall")
        rainfall_dirs.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))  # File name sorting

        for rainfall in rainfall_dirs:
            # for rainfall, hurs in zip(rainfall_dirs, hurs_dirs):
            rainfall_path = year_path + "\\" + year + "\\" + "rainfall" + "\\" + rainfall

            rdth_paths = [rainfall_path]
            year_dirs.append(rdth_paths)

        dirs_paths.append(year_dirs)

    print(dirs_paths)

    return dirs_paths
'''


# 2 - 4 bands fusion, output tifs to "Bands_4"
def bandsSynthesis(dirs_paths):
    """
    Band synthesis: 5-bands or 4-bands or 3-bands
    """
    # All tif data paths
    tif_paths = list()
    # global band_temp
    for year in dirs_paths:
        # Create Output Folder
        bandsfolder_path = year[0][0].rsplit("\\", 2)[0] + "\\5Bands"
        os.makedirs(bandsfolder_path, exist_ok=True)

        # Path of tif files for all months of a year
        year_dirs = list()
        for month in year:
            # Obtain information such as projection band
            dataset_init = gdal.Open(month[0])
            # Create a graph to be output
            gtiff_driver = gdal.GetDriverByName('GTiff')
            png_driver = gdal.GetDriverByName('PNG')
            file_path = bandsfolder_path + "\\" + month[0].split("_", 1)[-1]  # 2020_1.tif
            year_dirs.append(file_path)
            out_rdth = gtiff_driver.Create(file_path, dataset_init.RasterXSize, dataset_init.RasterYSize, 5,
                                           GDT_Float64)  # 5 bands
            # out_rdth = gtiff_driver.Create(file_path, dataset_init.RasterXSize, dataset_init.RasterYSize, 3,
            #                              GDT_Float64)  # 3 bands
            out_rdth.SetProjection(dataset_init.GetProjection())
            # Obtain the geographic information of the original band
            out_rdth.SetGeoTransform(dataset_init.GetGeoTransform())

            # Fill in each band in the tiff
            for index, rdth_file in enumerate(month):
                # 4 bands: rainfall, dem, tas, hurs
                # 5 bands: rainfall, dem, tas, hurs, pv
                dataset = gdal.Open(rdth_file)
                band_temp = dataset.GetRasterBand(1)
                out_rdth.GetRasterBand(1 + index).WriteArray(band_temp.ReadAsArray())

            # Save as png
            # png_driver.CreateCopy(bandsfolder_path + "\\" + month[0].split("_", 1)[-1].split(".")[0] + '.png', out_rdth)
            del out_rdth

        tif_paths.append(year_dirs)

    return tif_paths


'''
def bandsSynthesis(dirs_paths):
    """
    Save rainfall data as tiff
    """
    # All tif data paths
    tif_paths = list()

    for year in dirs_paths:
        # Create Output Folder
        bandsfolder_path = year[0][0].rsplit("\\", 2)[0] + "\\1Bands"
        os.makedirs(bandsfolder_path, exist_ok=True)

        # Path of tif files for all months of a year
        year_dirs = list()
        for month in year:
            # Obtain information such as projection band
            dataset_init = gdal.Open(month[0])
            # Create a tiff to be output
            gtiff_driver = gdal.GetDriverByName('GTiff')
            png_driver = gdal.GetDriverByName('PNG')
            file_path = bandsfolder_path + "\\" + month[0].split("_", 1)[-1]  # 2020_1.tif
            year_dirs.append(file_path)
            out_rdth = gtiff_driver.Create(file_path, dataset_init.RasterXSize, dataset_init.RasterYSize, 1,
                                           GDT_Float64)  # 1 band

            out_rdth.SetProjection(dataset_init.GetProjection())
            # Obtain the geographic information of the original band
            out_rdth.SetGeoTransform(dataset_init.GetGeoTransform())

            # Fill in each band in the figure
            for index, rdth_file in enumerate(month):
                # 1 band: rainfall
                dataset = gdal.Open(rdth_file)
                band_temp = dataset.GetRasterBand(1)
                out_rdth.GetRasterBand(1 + index).WriteArray(band_temp.ReadAsArray())

            # Save as png
            # png_driver.CreateCopy(bandsfolder_path + "\\" + month[0].split("_", 1)[-1].split(".")[0] + '.png', out_rdth)
            del out_rdth

        tif_paths.append(year_dirs)

    return tif_paths
'''


# 5 - Calculate weights
def calculateWeights(tifs_path):
    """
    Random Forest calculates the weights.
    """
    # Paths to files where .tif is stored
    tifs_paths = list()
    # Read all .tif files in the folder
    zone_masks = glob.glob(os.path.join(tifs_path, "*.tif"))
    zone_masks.sort()
    tifs_paths.extend(zone_masks)

    rainfall_list = list()
    dem_list = list()
    tas_list = list()
    hurs_list = list()
    for tif_path in tifs_paths:
        tif = cv2.imread(tif_path, -1)

        # The rainfall, dem, tas, hurs passages of the divisional picture
        (rainfall, dem, tas, hurs) = cv2.split(tif)

        # Convert an array to a list, add it to the list
        rainfall_list.extend(np.array(rainfall).flatten().tolist())
        dem_list.extend(np.array(dem).flatten().tolist())
        tas_list.extend(np.array(tas).flatten().tolist())
        hurs_list.extend(np.array(hurs).flatten().tolist())

    dem_tas_hurs = list()
    for dem_value, tas_value, hurs_value in zip(dem_list, tas_list, hurs_list):
        dem_tas_hurs.append([dem_value, tas_value, hurs_value])
    # List → DataFrame
    data = DataFrame(dem_tas_hurs)
    # Define the name of each column of data, in the order of the columns in your own file
    data.columns = ["dem", "tas", "hurs"]

    # Random forest assignment weights
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:], rainfall_list, test_size=0.3,
                                                        random_state=0)
    print((x_train.shape[1]))

    forest = RandomForestClassifier(n_estimators=1200, random_state=0, n_jobs=-1)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    # [::-1] denotes the output of each indicator ranked by weight size
    indices = np.argsort(importances)[::-1]
    feat_labels = ["dem", "tas", "hurs"]
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


# 6 - Convolutional dimensionality reduction
def convDimensionalityReduction(tifs_path, out_path):
    tifs_paths = glob.glob(os.path.join(tifs_path, "*.tif"))
    tifs_paths.sort()

    for tif_path in tifs_paths:
        tif = gdal.Open(tif_path)  # 4bands
        # Turn a multichannel image into a multidimensional array in numpy: number of channels, length, width.
        tif_numpy = tif.ReadAsArray()  # (4, 64, 64) uint8

        # Convert to tensor [4, 64, 64] torch.uint8
        tif_tensor = torch.tensor(tif_numpy)
        # print("tif_tensor[[:]].min(): {}".format(tif_tensor[[3].min()))
        # print("tif_tensor[::3].max(): {}".format(tif_tensor[::3].max()))

        # [1, 4, 64, 64] torch.float32
        x = tif_tensor.unsqueeze(0).to(torch.float32)
        # print(x[0][0][:].max())

        conv = torch.nn.Conv2d(4, 3, kernel_size=1, bias=False)

        kernel_value = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0.4, 0.6, 0]]
        kernel_c = torch.Tensor(kernel_value).view(3, 4, 1, 1)
        conv.weight.data = kernel_c.data
        tif_conv = conv(x)  # torch.Size([1, 3, 64, 64])

        # (64, 64, 3) float32
        tif_conv_numpy = tif_conv.squeeze(0).permute(1, 2, 0).detach().numpy()

        tif_conv_uint8 = np.rint(tif_conv_numpy).astype(np.uint8)

        jpg_out_path = out_path + "\\" + tif_path.split(".")[0].rsplit("\\")[-1] + ".tif"

        # RGB->BGR
        tif_conv_uint8_bgr = cv2.cvtColor(tif_conv_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(jpg_out_path, tif_conv_uint8_bgr)


# 7 - Change the file suffix. tif->jpg
def changeFileSuffix(files_path):
    files_paths = glob.glob(os.path.join(files_path, "*.tif"))
    files_paths.sort()
    for file_path in files_paths:
        new_name = file_path.split(".")[0] + ".jpg"
        # Implementing renaming operations
        os.rename(
            os.path.join(files_path, file_path),
            os.path.join(files_path, new_name))


'''
# Image geometry transformation
def rotateImg(input_img, out_img):
    img = cv2.imread(input_img)
    # src = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # Image height, width and number of channels
    rows, cols, channel = img.shape

    # Rotate around the centre of the image
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))

    # Image Flip
    # 0: Flip with X-axis as symmetry axis
    # >0: Flip with Y-axis as symmetry axis
    # <0: X-axis and Y-axis are both flipped
    # img0 = cv2.flip(src, 0) # 0 1 -1

    # Display images
    cv2.imshow("img", img)
    cv2.imshow("rotated", rotated)
    # Waiting to be shown
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(out_img, rotated)
'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read raw data
    # dirs_paths = getImageList(const.DATA_PATH, const.DEM_PATH)
    # Band fusion to form new data
    # tif_paths = bandsSynthesis(dirs_paths)

    # tifToJPG(const.COPY_TIFS_PATH, const.JPG_PATH)

    # Random forest calculation weights
    # calculateWeights("F:\\")

    # Dimensionality reduction
    # convDimensionalityReduction("F:\\Preprocessing\\crop_copy1\\4Bands_crop_copy",
    # "F:\\Preprocessing\\dimensionality_reduction2\\4Bands_crop_copy_tifs")

    # tif -> jpg
    changeFileSuffix("F:\\Preprocessing\\dimensionality_reduction2\\4Band_crop_copy_tifs_jpgs")

    # Image geometry transformation
    # rotateImg("F:\\", "G:\\")
