#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from osgeo import gdal
import sys
import matplotlib.ticker as ticker
import scipy.ndimage.filters
from skimage.morphology import disk
from skimage.filters import median
from datetime import datetime, timedelta
from skimage import exposure

def gtiff_enhance(tif_path, out_path):
    # Open mosaic
    ds_mos = gdal.Open(tif_path)

    # Clip data and rescale
    band = ds_mos.GetRasterBand(1)
    A0 = band.ReadAsArray()
    A0[A0 == -999.] = np.nan
    [rows, cols] = A0.shape

    ###################################
    # Contrast enhance
    ###################################
    # Map values to the (0, 255) range:

    A0 = (A0 - np.nanmin(A0)) * 254.0 / (np.nanmax(A0) - np.nanmin(A0))

    # Kernel for negative Laplacian:
    kernel = np.ones((3, 3)) * (-1)
    kernel[1, 1] = 6

    # Convolution of the image with the kernel:
    Lap = scipy.ndimage.filters.convolve(A0, kernel)

    # Map Laplacian to some new range:
    ShF = 100  # Sharpening factor!
    Laps = Lap * ShF / np.nanmax(Lap)

    # Add negative Laplacian to the original image:
    A = A0 + Laps
    # Set negative values to 0, values over 254 to 254:
    A = np.clip(A, 0, 254)
    ###################################
    # END contrast enhance
    ###################################

    #print('\nWriting geotiff...')
    #A = median(A, disk(3))

    # Contrast stretching
    p2, p98 = np.nanpercentile(A, (2, 98))
    A = exposure.rescale_intensity(A, in_range=(p2, p98), out_range=(0, 254))

    A[np.isnan(A)] = 255

    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(out_path, cols, rows, 1, gdal.GDT_Byte)
    outdata.SetGeoTransform(ds_mos.GetGeoTransform())
    outdata.SetProjection(ds_mos.GetProjection())
    outdata.GetRasterBand(1).WriteArray(A)
    outdata.GetRasterBand(1).SetNoDataValue(255)
    outdata.FlushCache()
    #print('Done.\n')

    ds_mos = None
    band = None
    outdata = None

    return A

in_path = sys.argv[1]
mask = sys.argv[2]
out_path = sys.argv[3]

try:
    os.makedirs(out_path)
except:
    pass

td = datetime.today()
days = 1
dts = []
#dt_td_str = '%s%02d%02dT' % (td.year, td.month, td.day)
#dts.append(dt_td_str)

for i in range(days+1):
    td_yest = datetime.today() - timedelta(days=i)
    dt_str_yest = '%s%02d%02dT' % (td_yest.year, td_yest.month, td_yest.day)
    dts.append(dt_str_yest)

print(dts)

# Find files for 2 days
files_to_mosaic = []
for root, dirs, files in os.walk(in_path):
    for fname in files:
        for idt in dts:
            if fname.find(mask) >= 0 and fname.find(idt) >= 0:
                files_to_mosaic.append('%s/%s' % (root, fname))

print(files_to_mosaic)

#files_to_mosaic = glob.glob('%s/UPS*%s*.tif*' % (in_path, mask))
files_to_mosaic.sort(key=lambda x: os.path.basename(x).split('_')[6])

#files_string = (" ".join(files_to_mosaic))
#print(files_string)

# Out filename
out_title = '%s_%s' % (os.path.basename(files_to_mosaic[0]).split('_')[6],
                       os.path.basename(files_to_mosaic[-1]).split('_')[6])

# Path to 'full' tiff mosaic
tif_path = '%s/S1_mos_%s.tif' % (out_path, out_title)

# Path to Byte tiff mosaic
out_path = '%s/ps_S1_mos_%s.tif' % (out_path, out_title)

#command = "gdal_merge.py -ps 200. 200. -o %s/S1_mos_%s.tif -of gtiff %s" % (out_path, out_title, files_string)

print('\nStart making mosaic from %s files...' % len(files_to_mosaic))
g = gdal.Warp(tif_path, files_to_mosaic, format="GTiff", options=["COMPRESS=LZW", "TILED=YES"],
              srcNodata=np.nan,
              dstNodata=-999.)
print('Done.\n')

# Enhance contrast and save to tiff
print('\nConverting to Byte...')
gtiff_enhance(tif_path, out_path)
print('Done.\n')

# Remove 'full' mosaic
os.remove(tif_path)

g = None

'''

def raster2array(geotif_file):
    metadata = {}
    dataset = gdal.Open(geotif_file)
    metadata['array_rows'] = dataset.RasterYSize
    metadata['array_cols'] = dataset.RasterXSize
    metadata['bands'] = dataset.RasterCount
    metadata['driver'] = dataset.GetDriver().LongName
    metadata['projection'] = dataset.GetProjection()
    metadata['geotransform'] = dataset.GetGeoTransform()
    
    mapinfo = dataset.GetGeoTransform()
    metadata['pixelWidth'] = mapinfo[1]
    metadata['pixelHeight'] = mapinfo[5]

    xMin = mapinfo[0]
    xMax = mapinfo[0] + dataset.RasterXSize/mapinfo[1]
    yMin = mapinfo[3] + dataset.RasterYSize/mapinfo[5]
    yMax = mapinfo[3]
    
    metadata['extent'] = (xMin,xMax,yMin,yMax)
    
    raster = dataset.GetRasterBand(1)
    array_shape = raster.ReadAsArray(0, 0, metadata['array_cols'],metadata['array_rows']).astype(np.float).shape
    metadata['noDataValue'] = np.nan # raster.GetNoDataValue()
    metadata['scaleFactor'] = raster.GetScale()
    
    array = np.zeros((array_shape[0], array_shape[1], dataset.RasterCount),'uint8') #pre-allocate stackedArray matrix
    
    if metadata['bands'] == 1:
        raster = dataset.GetRasterBand(1)
        metadata['noDataValue'] = raster.GetNoDataValue()
        metadata['scaleFactor'] = raster.GetScale()
              
        array = dataset.GetRasterBand(1).ReadAsArray(0,0,metadata['array_cols'],metadata['array_rows']).astype(np.float)
        #array[np.where(array==metadata['noDataValue'])]=np.nan
        array = array/metadata['scaleFactor']
    
    elif metadata['bands'] > 1:    
        for i in range(1, dataset.RasterCount+1):
            band = dataset.GetRasterBand(i).ReadAsArray(0,0,metadata['array_cols'],metadata['array_rows']).astype(np.float)
            #band[np.where(band==metadata['noDataValue'])]=np.nan
            band = band/metadata['scaleFactor']
            array[..., i-1] = band

    return array, metadata
    
data, metadata = raster2array('%s/S1_mos_%s.tif' % (out_path, out_title))


#print metadata in alphabetical order
for item in sorted(metadata):
    print(item + ':', metadata[item])

def plot_array(array, spatial_extent, colorlimit,ax=plt.gca(), title='', cmap_title='', colormap=''):
    plt.clf()
    plot = plt.imshow(array, extent=spatial_extent)
    cbar = plt.colorbar(plot, aspect=40)
    plt.set_cmap(colormap)
    cbar.set_label(cmap_title, rotation=90, labelpad=20)
    plt.title(title)
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    rotatexlabels = plt.setp(ax.get_xticklabels(), rotation=90)
    plt.savefig('%s/ql_S1_mos_%s.png' % (out_path, out_title), bbox_inches='tight', dpi=300)


plot_array(data, metadata['extent'],
           (-35, -5),
           title='S1 mosaic %s' % out_title,
           cmap_title=r'$\sigma_{0}$, dB',
           colormap='gray')

'''