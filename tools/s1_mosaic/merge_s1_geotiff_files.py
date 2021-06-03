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

in_path = sys.argv[1]
mask = sys.argv[2]
out_path = sys.argv[3]

try:
    os.makedirs(out_path)
except:
    pass

files_to_mosaic = glob.glob('%s/UPS*%s*.tif*' % (in_path, mask))
files_to_mosaic.sort(key=lambda x: os.path.basename(x).split('_')[6])
files_string = (" ".join(files_to_mosaic))

print(files_string)

out_title = '%s_%s' % (os.path.basename(files_to_mosaic[0]).split('_')[6], os.path.basename(files_to_mosaic[-1]).split('_')[6])

#command = "gdal_merge.py -ps 200. 200. -o %s/S1_mos_%s.tif -of gtiff %s" % (out_path, out_title, files_string)

print('\nStart making mosaic...')
g = gdal.Warp('%s/S1_mos_%s.tif' % (out_path, out_title), files_to_mosaic, format="GTiff", options=["COMPRESS=LZW", "TILED=YES"],
              srcNodata=np.nan,
              dstNodata=-999.)
g = None
print('Done.\n')

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