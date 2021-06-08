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
import re

def apply_anisd(img, gamma=0.25, step=(1., 1.), kappa=50, ploton=False):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal

    Sep 2017 modified by Denis Demchev
    """

    # init args

    # Conduction coefficient
    kappa = kappa

    # Number of iterations
    niter = 10

    # Number of equation (1,2)
    option = 1

    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    p2, p98 = np.nanpercentile(imgout, (2, 98))
    im = exposure.rescale_intensity(imgout, in_range=(p2, p98), out_range=(0, 254))
    return im

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
    '''
    A0 = (A0 - np.nanmin(A0)) * 254.0 / (np.nanmax(A0) - np.nanmin(A0))

    # Kernel for negative Laplacian:
    kernel = np.ones((3, 3)) * (-1)
    kernel[1, 1] = 5

    # Convolution of the image with the kernel:
    Lap = scipy.ndimage.filters.convolve(A0, kernel)

    # Map Laplacian to some new range:
    ShF = 50  # Sharpening factor!
    Laps = Lap * ShF / np.nanmax(Lap)

    # Add negative Laplacian to the original image:
    A = A0 + Laps
    # Set negative values to 0, values over 254 to 254:
    A = np.clip(A, 0, 254)
    '''
    ###################################
    # END contrast enhance
    ###################################

    #print('\nWriting geotiff...')
    #A = median(A, disk(3))

    # Contrast stretching
    #from skimage.filters.rank import mean_bilatera
    from skimage.restoration import denoise_tv_chambolle
    print('\nStart filtering...\n')
    #A = denoise_tv_chambolle(A0, weight=10)#median(A0, disk(1))
    #A = apply_anisd(A0)
    #№from skimage.restoration import denoise_bilateral
    #A = denoise_bilateral(A0)
    #print('\nEnd filtering\n')

    p2, p98 = np.nanpercentile(A0, (2, 98))
    A = exposure.rescale_intensity(A0, in_range=(p2, p98), out_range=(0, 254))

    # Gamma correction
    A = exposure.adjust_gamma(A, 1.6)

    print('\nMedian filtering...')
    A = median(A, disk(1))
    print('\nDone.')

    # Rescale again
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
days = 2

dt_2 = datetime.today()
dt_1 = datetime.today() - timedelta(days=days)

# Find files within i days time gap
files_to_mosaic = []
for root, dirs, files in os.walk(in_path):
    for fname in files:
        if fname.endswith('tiff'):
            m = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', fname)[0]
            dt_f = datetime.strptime(('%s/%s/%sT%s:%s:%s' %
                                     (m[0:4], m[4:6], m[6:8], m[9:11], m[11:13], m[13:15])),
                                    '%Y/%m/%dT%H:%M:%S')

            if fname.find(mask) >= 0 and dt_f >= dt_1 and dt_f <= dt_2:
                print('%s added!' % fname)
                files_to_mosaic.append('%s/%s' % (root, fname))

print(files_to_mosaic)

files_to_mosaic.sort(key=lambda x: os.path.basename(x).split('_')[6], reverse=True)

#files_string = (" ".join(files_to_mosaic))
#print(files_string)

# Out filename
'''
out_title = '%s_%s' % (os.path.basename(files_to_mosaic[0]).split('_')[6],
                       os.path.basename(files_to_mosaic[-1]).split('_')[6])
'''

out_title = '%s' % os.path.basename(files_to_mosaic[0]).split('_')[6][0:8]

# Path to 'full' tiff mosaic
tif_path = '%s/S1_mos_%s.tif' % (out_path, out_title)

# Path to Byte tiff mosaic
out_path = '%s/ps_S1_mos_%s.tif' % (out_path, out_title)

#command = "gdal_merge.py -ps 200. 200. -o %s/S1_mos_%s.tif -of gtiff %s" % (out_path, out_title, files_string)

print('\nStart making mosaic from %s files...' % len(files_to_mosaic))
# "COMPRESS=LZW"
try:
    os.remove(tif_path)
except:
    pass

g = gdal.Warp(tif_path, files_to_mosaic, format="GTiff", options=["BIGTIFF=YES", "TILED=YES"],
              xRes=300., yRes=300.,
              dstSRS='EPSG:3995',
              srcNodata=np.nan,
              dstNodata=-999.)
print('Done.\n')
g = None

# Enhance contrast and save to tiff
print('\nConverting to Byte...')
# Delete old files
try:
    os.remove(out_path)
except:
    pass

gtiff_enhance(tif_path, out_path)
print('Done.\n')

# Remove 'full' mosaic
os.remove(tif_path)

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