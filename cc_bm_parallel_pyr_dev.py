import matplotlib
matplotlib.use('Agg')
# coding: utf-8
#
# Ice drift retrieval algorithm based on [1] from a pair of SAR images
# [1] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light and Magic.
#
##################################################
# Last modification: 22 July, 2019
# TODO:
# 1) Pyramidal strategy (do we need this?)
# 2) add ocean cm maps ('Balance' for divergence)
##################################################

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

from skimage.feature import match_template
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io, img_as_ubyte
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.filters import laplace
from skimage import exposure
from skimage.filters.rank import gradient
from skimage import filters
from sklearn.neighbors import KDTree
import sys
import sklearn.neighbors
import re
import geojson
import shapefile as sf
import pyproj
from osgeo import gdal, osr
from datetime import datetime

from netCDF4 import Dataset

from osgeo import gdal, osr, gdal_array, ogr
import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
import time

def remove_files(ddir):
    ffiles = glob.glob('%s/*.*' % ddir)
    for ifile in ffiles:
        try:
            os.remove(ifile)
        except:
            pass
def length_between(v1, v2):
    v1_length = np.hypot(v1[0], v1[1])
    v2_length = np.hypot(v2[0], v2[1])
    return abs(v1_length - v2_length)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

        angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        angle_between((1, 0, 0), (1, 0, 0))
        0.0
        angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return np.degrees(0.0)
        else:
            return np.degrees(np.pi)
    return np.degrees(angle)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 6}

matplotlib.rc('font', **font)

def plot_peaks(immm1, immm2, uuu, vvv, iidx_line, iidx_row, resss, pref,
               lline_1, rrow_1, lline_2, rrow_2, u_direct, Li0, v_direct, Li1):
    plt.clf()
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(immm1, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(immm2, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    rect = plt.Rectangle((uuu - Conf.grid_step, vvv - Conf.grid_step), Conf.block_size, Conf.block_size,
                         edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(resss)
    ax3.set_axis_off()
    ax3.set_title('match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(uuu, vvv, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    # !Plot control imformation
    plt.title('ll1: %s rr1:%s ll2:%s rr2:%s\nu: %s v: %s Li0: %s Li1: %s' %
              (lline_1, rrow_1, lline_2, rrow_2,
               u_direct, v_direct, Li0, Li1))

    # plt.show()
    plt.savefig('peaks_plot/%s_%s_%s.png' % (pref, iidx_line, iidx_row), bbox_inches='tight', dpi=300)

# TODO: check
def check_borders(im):
    ''' n pixels along line means image has a black border '''
    flag = 0
    ch = 0
    j = 0
    for i in range(im.shape[0] - 1):
        while j < im.shape[1] - 1 and im[i,j] > 0:
            j += 1
        else:
            if j < im.shape[1] - 1 and (im[i,j] == 0 or im[i,j] == 255):
                while im[i,j] == 0 and j < im.shape[1] - 1:
                    j += 1
                    ch += 1
                if ch >= 15:
                    flag = 1
                    #print('Black stripe detected!')
                    return flag
        j = 0
        ch = 0
    return flag

# Matching
def matching(templ, im):
    ''' Matching '''
    # Direct macthing
    #pool = Pool(processes=3)
    #result = pool.apply(match_template, args=(im, templ, True, 'edge',))
    #pool.close()

    result = match_template(im, templ, True, 'edge',)

    # Drihle statement
    # No need if 'edge' in 'match_template'
    #n = Conf.block_size #/ 2  # 100
    n = int(im.shape[0]/10.)
    # First and last n lines
    result[0:n, :] = 0.
    result[-n:, :] = 0.
    # First and last n rows
    result[:, 0:n] = 0.
    result[:, -n:] = 0.

    ij = np.unravel_index(np.argmax(result), result.shape)
    u_peak, v_peak = ij[::-1]

    #print('u_peak, v_peak: (%s, %s)' % (u_peak, v_peak))

    return u_peak, v_peak, result

def filter_local_homogenity(arr_cc_max, y, x, u, v, filter_all=False):
    '''
    Local homogenity filtering (refine CC peak)
    y - axe (top -> bottom)
    x - axe (left -> right)
    u - along Y (top -> bottom)
    v - along X (left -> right)
    mask - indicate that a vector has been reprocessed
    '''

    # Mask array with refined tie points
    mask = np.zeros_like(arr_cc_max)

    # TODO: processing of border vectors
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            # Calculate median of u and v for 8 neighbors

            # Matrix with negbors
            nn = np.zeros(shape=(2, 3, 3))
            nn[:] = np.nan

            # U and V
            #if not np.isnan(u[i - 1, j - 1]):
            nn[0, 0, 0] = u[i - 1, j - 1]
            nn[0, 0, 1] = u[i - 1, j]
            nn[0, 0, 2] = u[i - 1, j + 1]

            nn[1, 0, 0] = v[i - 1, j - 1]
            nn[1, 0, 1] = v[i - 1, j]
            nn[1, 0, 2] = v[i - 1, j + 1]

            nn[0, 1, 0] = u[i, j-1]
            nn[0, 1, 2] = u[i, j+1]

            nn[1, 1, 0] = v[i, j - 1]
            nn[1, 1, 2] = v[i, j + 1]

            nn[0, 2, 0] = u[i + 1, j - 1]
            nn[0, 2, 1] = u[i + 1, j]
            nn[0, 2, 2] = u[i + 1, j + 1]

            nn[1, 2, 0] = v[i + 1, j - 1]
            nn[1, 2, 1] = v[i + 1, j]
            nn[1, 2, 2] = v[i + 1, j + 1]

            # Check number of nans and find median for U and V
            uu = nn[0, :, :]
            # If number of neighbors <= 3
            if len(uu[np.isnan(uu)]) > 5:
                u[i, j] = np.nan
                v[i, j] = np.nan
                arr_cc_max[i, j] = 0
                #print 'NANs > 3!'
            else:
                u_median = np.nanmedian(nn[0, :, :])
                v_median = np.nanmedian(nn[1, :, :])

                if not filter_all:
                    if np.isnan(u[i, j]) or abs(u[i, j] - u_median) > abs(u_median) or \
                            abs(v[i, j] - v_median) > abs(v_median):
                        u[i, j] = u_median
                        v[i, j] = v_median
                        mask[i, j] = 1
                        arr_cc_max[i, j] = 1
                        #print '%s %s %s %s' % (u[i, j], v[i, j], u_median, v_median)
                else:
                    u[i, j] = u_median
                    v[i, j] = v_median
                    mask[i, j] = 1
                    arr_cc_max[i, j] = 1

    return mask, y, x, u, v, arr_cc_max

def filter_Rmin(arr_cc_max):
    ''' Minimum correlation threshold filtering '''
    # Remove and plot vectors with R < Rmin, where Rmin = Rmean - Rstd
    R_mean = np.nanmean(arr_cc_max)
    R_std = np.nanstd(arr_cc_max)
    R_min = R_mean - R_std

    mask = np.zeros_like(arr_cc_max)
    mask[(arr_cc_max < R_min)] = 1

    return mask

def plot_scatter(fname, img, x, y, msize=0.1):
    ''' Plot scatter of initial points '''
    plt.clf()
    plt.imshow(Conf.img1, cmap='gray')
    plt.scatter(x, y, s=msize, color='red')
    plt.savefig(fname, bbox_inches='tight', dpi=600)

def plot_arrows(fname, img, x, y, u, v, cc, arrwidth=0.005, headwidth=3.5, flag_color=True):
    ''' Plot arrows on top of image '''
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.imshow(img, cmap='gray')
    if flag_color:
        plt.quiver(x, y, u, v, cc, angles='xy', scale_units='xy', width=arrwidth, headwidth=headwidth,
               scale=1, cmap='jet')

        plt.quiver(x, y, u, v, cc, angles='xy', scale_units='xy', width=arrwidth, headwidth=headwidth,
                   scale=1, cmap='jet')
        cbar = plt.colorbar()
        cbar.set_label('Correlation coeff.')
    else:
        plt.quiver(x, y, u, v, angles='xy', scale_units='xy', width=arrwidth, headwidth=headwidth,
                   scale=1, color='yellow')
    plt.savefig(fname, bbox_inches='tight', dpi=600)

    # Plot start points
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.imshow(img, cmap='gray')
    plt.scatter(x[~np.isnan(u)], y[~np.isnan(u)], s=Conf.grid_step/2., facecolors='yellow', edgecolors='black')
    plt.savefig('%s/pts_%s' % (os.path.dirname(fname), os.path.basename(fname)), bbox_inches='tight', dpi=600)



# TODO!: remove
def plot_arrows_one_color(fname, img, x, y, u, v, cc, arrwidth=0.005, headwidth=3.5, flag_color=False):
    ''' Plot arrows on top of image '''
    plt.clf()
    plt.imshow(img, cmap='gray')
    if flag_color:
        plt.quiver(x, y, u, v, cc, angles='xy', scale_units='xy', width=arrwidth, headwidth=headwidth,
               scale=1, cmap='jet')
        cbar = plt.colorbar()
        cbar.set_label('Correlation coeff.')
    else:
        plt.quiver(x, y, u, v, angles='xy', scale_units='xy', width=arrwidth, headwidth=headwidth,
                   scale=1, color='yellow')
    plt.savefig(fname, bbox_inches='tight', dpi=1200)

def crop_images(img1, img2, y0, x0):
    '''
    :param Conf.img1: image1
    :param Conf.img2: image2
    :param x0: center of patch on image2
    :param y0: center of patch on image2
    :return: image patches
    '''

    # TODO: x2, y2 for Conf.img2

    height, width = img1.shape

    # Crop Conf.img1
    iidx_line = int(x0)
    iidx_row = int(y0)

    LLt0 = np.max([0, iidx_line - Conf.grid_step])
    LLt1 = np.max([0, iidx_row - Conf.grid_step])
    RRt0 = np.min([iidx_line + Conf.grid_step, height])
    RRt1 = np.min([iidx_row + Conf.grid_step, width])

    # Crop patch from Conf.img1
    im1 = Conf.img1[LLt0:RRt0, LLt1:RRt1]

    LLi0 = np.max([0, iidx_line - Conf.block_size * Conf.search_area])
    LLi1 = np.max([0, iidx_row - Conf.block_size * Conf.search_area])
    RRi0 = np.min([iidx_line + Conf.block_size * Conf.search_area, height])
    RRi1 = np.min([iidx_row + Conf.block_size * Conf.search_area, width])

    # Crop search area from Conf.img2
    im2 = Conf.img2[LLi0:RRi0, LLi1:RRi1]

    # Offset for image1
    y_offset_Conf.img1 = iidx_line  # - Conf.block_size/2
    x_offset_Conf.img1 = iidx_row  # - Conf.block_size/2

    #####################
    # Filtering
    #####################
    # Median filtering
    if Conf.img_median_filtering:
        # print 'Median filtering'
        # im2 = median(im2, disk(3))
        # im1 = median(im1, disk(3))
        im2 = median(im2, disk(Conf.median_kernel))
        im1 = median(im1, disk(Conf.median_kernel))

    if Conf.img_laplace_filtering:
        im2 = laplace(im2)
        im1 = laplace(im1)

    if Conf.img_gradient_filtering:
        im2 = gradient(im2, disk(3))
        im1 = gradient(im1, disk(3))

    if Conf.img_scharr_filtering:
        # filters.scharr(camera)
        im2 = filters.scharr(im2)
        im1 = filters.scharr(im1)

    ########################
    # End filtering
    ########################

    # Check for black stripes
    flag1 = check_borders(im1)
    flag2 = check_borders(im2)

    return im1, im2

# TODO: EXPERIMENTAL
def cc_bm(arguments):
    # BM test flag
    f=0
    # Parse arguments
    iidx_line, iidx_row, LLi0, LLi1, im1_name, im2_name, pref, lll_line_start, lll_row_start = arguments

    if iidx_line is not None:
        # Open two images
        im1 = io.imread(im1_name, 0)
        im2 = io.imread(im2_name, 0)

        #####################
        # Filtering
        #####################
        # Median filtering
        if Conf.img_median_filtering:
            # print 'Median filtering'
            # im2 = median(im2, disk(3))
            # im1 = median(im1, disk(3))
            im1 = median(im1, disk(Conf.median_kernel))
            im2 = median(im2, disk(Conf.median_kernel))

        if Conf.img_laplace_filtering:
            im1 = laplace(im1)
            im2 = laplace(im2)

        if Conf.img_gradient_filtering:
            im1 = gradient(im1, disk(3))
            im2 = gradient(im2, disk(3))

        if Conf.img_scharr_filtering:
            # filters.scharr(camera)
            im1 = filters.scharr(im1)
            im2 = filters.scharr(im2)

        ########################
        # End filtering
        ########################

        # Check for black stripes
        flag1 = check_borders(im1)
        flag2 = check_borders(im2)

        # No black borders in the first image
        if flag1 == 0 and flag2 == 0:
            u_direct, v_direct, result = matching(im1, im2)
            # Peak maximum CC
            cc_max = np.max(result)

            # Get coordinates with offsets
            lline_2, rrow_2 = u_direct + LLi0, v_direct + LLi1

            lline_2_test, rrow_2_test = v_direct + LLi0, u_direct + LLi1

            lline_1, rrow_1 = iidx_line, iidx_row

            # If obtained end of bm vectors compared to start points of direct

            if abs(lline_2_test - lll_line_start) < Conf.bm_th and abs(rrow_2_test - lll_row_start) < Conf.bm_th:
                #print('\nlline_2_test, lll_line_start: (%s, %s)' % (lline_2_test, lll_line_start))
                #print('rrow_2_test, lll_row_start: (%s, %s)\n' % (rrow_2_test, lll_row_start))

                #print('\nCOORDS: %s %s' % (arr_lines_1[i, j], arr_rows_1[i, j]))
                #print('COORDS: %s %s\n' % (arr_lines_2[i, j], arr_rows_2[i, j]))

                # Peaks plot
                if Conf.plot_correlation_peaks:
                    plot_peaks(im1, im2, u_direct, v_direct, iidx_line, iidx_row, result, pref)
                    #plot_peaks(im1_bm, im2_bm, uu_bm, vv_bm, iidx_line, iidx_row,
                    #           result_bm, 'bm')

                return lline_1, rrow_1, lline_2-lline_1, rrow_2-rrow_1, cc_max
                #return lline_2, rrow_2, lline_1 - lline_2, rrow_1 - rrow_2, cc_max
            else:
                pass

        else:
            # if crop images have black stripes
            if flag1 == 1:
                print('IMG_1: %s_%s' % (iidx_line, iidx_row))
                io.imsave('ci_%s_1/black_%s_%s.png' % (Conf.out_fname, iidx_line, iidx_row), im1)
            if flag2 == 1:
                print('IMG_2: %s_%s' % (idx_line, idx_row))
                io.imsave('ci_%s_2/black_%s_%s.png' % (Conf.out_fname, iidx_line, iidx_row), im2)

def filter_BM(th = 10):
    ''' Back matching test '''
    Conf.bm_th = th # pixels

    u_back = arr_rows_2_bm - arr_rows_1_bm
    u_direct = arr_rows_2 - arr_rows_1

    v_back = arr_lines_2_bm - arr_lines_1_bm
    v_direct = arr_lines_2 - arr_lines_1

    u_dif = u_direct - u_back * (-1)
    v_dif = v_direct - v_back * (-1)


    #arr_rows_1, arr_lines_1, arr_rows_2, arr_lines_2, arr_cc_max
    #arr_rows_1_bm, arr_lines_1_bm, arr_rows_2_bm, arr_lines_2_bm, arr_cc_max_bm

    mask = np.zeros_like(arr_cc_max)
    mask[:,:] = 1
    mask[((abs(u_dif) < Conf.bm_th) & (abs(v_dif) < Conf.bm_th))] = 0

    #mask[((abs(arr_lines_1 - arr_lines_2_bm) > Conf.bm_th) | (abs(arr_rows_1 - arr_rows_2_bm) > Conf.bm_th))] = 1

    return mask

def plot_arrows_from_list(pref, fname, img, ll_data, arrwidth=0.005, headwidth=3.5, flag_color=True):
    ''' Plot arrows on top of image form a list of data '''
    plt.clf()
    plt.imshow(img, cmap='gray')

    # Get list without none and each elements
    ll_data = [x for x in ll_data if x is not None]

    yyy = [i[0] for i in ll_data]
    xxx = [i[1] for i in ll_data]
    uuu = [i[2] for i in ll_data]
    vvv = [i[3] for i in ll_data]
    ccc = [i[4] for i in ll_data]

    if flag_color:
        plt.quiver(xxx, yyy, uuu, vvv, ccc, angles='xy', scale_units='xy', width=arrwidth, headwidth=headwidth,
               scale=1, cmap='jet')

        cbar = plt.colorbar()
        cbar.set_label('Correlation coeff.')

        # Plot text with coordinates
        for i in range(len(xxx)):
            plt.text(xxx[i], yyy[i], r'(%s,%s)' % (yyy[i], xxx[i]), fontsize=0.07, color='yellow')
            plt.text(xxx[i] + uuu[i], yyy[i] + vvv[i], r'(%s,%s)' % (yyy[i] + vvv[i], xxx[i] + uuu[i]),
                     fontsize=0.07, color='yellow') # bbox={'facecolor': 'yellow', 'alpha': 0.5}
    else:
        plt.quiver(xxx, yyy, uuu, vvv, ccc, angles='xy', scale_units='xy', width=arrwidth, headwidth=headwidth,
                   scale=1, color='yellow')
    plt.savefig(fname, bbox_inches='tight', dpi=600)

    # Filter outliers here and plot
    plt.clf()
    plt.imshow(img, cmap='gray')



def outliers_filtering(x1, y1, uu, vv, cc, radius=256, angle_difference=5, length_difference=30,
                       total_neighbours=7, angle_neighbours=7, length_neighbours=7):
    # Get values of vector components
    #uu = x2 - x1
    #vv = y2 - y1

    idx_mask = []
    # Make 2D data of components
    #data = np.vstack((uu, vv)).T

    x1, y1, uu, vv, cc = np.array(x1), np.array(y1),\
                         np.array(uu, float), np.array(vv, float), np.array(cc, float)

    #  Radius based filtering
    vector_start_data = np.vstack((x1, y1)).T
    vector_start_tree = sklearn.neighbors.KDTree(vector_start_data)

    for i in range(0, len(x1), 1):
        # For list
        # req_data = np.array([x1[i], y1[i]]).reshape(1, -1)
        req_data = np.array((x1[i], y1[i])).reshape(1, -1)
        # Getting number of neighbours
        num_nn = vector_start_tree.query_radius(req_data, r=radius, count_only=True)

        if num_nn[0] < total_neighbours:
            idx_mask.append(i)

        # Keep small vectors
        if np.hypot(uu[i], vv[i]) < 10.:
            pass
        else:
            nn = vector_start_tree.query_radius(req_data, r=radius)
            data = np.vstack((uu[nn[0]], vv[nn[0]])).T

            num_of_homo_NN = 0
            num_of_length_homo_NN = 0

            ####################################################################
            # Loop through all found ice drift vectors to filter not homo
            ####################################################################
            for ii in range(num_nn[0]):

                # Angle between "this" vector and others
                angle_v1_v2 = angle_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])

                # Length between "this" vector and others
                diff_v1_v2 = length_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])

                if angle_v1_v2 <= angle_difference:
                    num_of_homo_NN = num_of_homo_NN + 1

                if diff_v1_v2 < length_difference:
                    num_of_length_homo_NN = num_of_length_homo_NN + 1

            if not (num_of_homo_NN >= angle_neighbours and num_of_length_homo_NN >= length_neighbours):
                idx_mask.append(i)

    tt = list(set(idx_mask))
    iidx_mask = np.array(tt)

    # Delete bad data
    '''
    x1_f = np.delete(x1, iidx_mask)
    y1_f = np.delete(y1, iidx_mask)
    uu_f = np.delete(uu, iidx_mask)
    vv_f = np.delete(vv, iidx_mask)
    cc_f = np.delete(cc, iidx_mask)
    '''

    # Mask (=NaN) bad values
    uu = np.array(uu, np.float)
    vv = np.array(vv, np.float)
    uu[iidx_mask] = np.nan
    vv[iidx_mask] = np.nan
    cc[iidx_mask] = 0.

    return x1, y1, uu, vv, cc

def export_to_vector(gtiff, x1, y1, u, v, output_path, gridded=False, data_format='geojson'):
    print('\nStart exporting to vector file...')
    if data_format not in ['geojson', 'shp']:
        print('Invalid format')
        return

    x2 = x1 + u
    y2 = y1 + v

    ds = gdal.Open(gtiff)

    geotransform = ds.GetGeoTransform()

    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjection())

    new_cs = osr.SpatialReference()
    new_cs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    transform = osr.CoordinateTransformation(old_cs, new_cs)

    if data_format == 'shp':
        w = sf.Writer(sf.POLYLINE)
        # w.field('id', 'C', '40')
        w.field('lat1', 'C', '40')
        w.field('lon1', 'C', '40')
        w.field('lat2', 'C', '40')
        w.field('lon2', 'C', '40')
        w.field('drift_m', 'C', '40')
        w.field('direction', 'C', '40')

    if data_format == 'geojson':
        features = []

    pixelWidth = geotransform[1]
    pixelHeight = geotransform[-1]

    print('Pixel size (%s, %s) m' % (pixelWidth, pixelHeight))

    for i in range(len(x1)):
        # print '%s  %s  %s  %s' % (y[ch], x[ch], u[ch], v[ch])

        if np.isnan(x2[i]) == False and np.isnan(y2[i]) == False:
            #!TODO: Need to be fixed!
            yy1 = geotransform[0] + float(x1[i]) * pixelWidth
            xx1 = geotransform[3] + float(y1[i]) * pixelHeight

            yy2 = geotransform[0] + float(x2[i]) * pixelWidth
            xx2 = geotransform[3] + float(y2[i]) * pixelHeight

            # print(xx1, yy1)

            latlon = transform.TransformPoint(float(xx1), float(yy1))
            lon1 = latlon[0]
            lat1 = latlon[1]

            latlon = transform.TransformPoint(float(xx2), float(yy2))
            lon2 = latlon[0]
            lat2 = latlon[1]

            # Big circle length
            try:
                mag, az = calc_distance(float(lon1), float(lat1), float(lon2), float(lat2))
                az = float(az)
                if az <= 180.0:
                    az = az + 180.0
                else:
                    az = az - 180.0
            except:
                mag, az = 999., 999.

            if data_format == 'shp':
                w.line(parts=[[[lon1, lat1], [lon2, lat2]]])
                w.record(str(i), str(lat1), str(lon1), str(lat2), str(lon2), str(mag), str(az))

            # coords_list.append((lon1, lat1))

            if data_format == 'geojson':
                new_line = geojson.Feature(geometry=geojson.LineString([(lon1, lat1), (lon2, lat2)]),
                                           properties={'id': str(i),
                                                       'lat1': lat1,
                                                       'lon1': lon1,
                                                       'lat2': lat2,
                                                       'lon2': lon2,
                                                       'drift_m': mag,
                                                       'azimuth': az})
                features.append(new_line)

    if data_format == 'shp':
        try:
            w.save(output_path)
            # create the PRJ file
            prj = open('%s.prj' % output_path.split('.')[0], "w")
            prj.write(
                '''GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]''')
            prj.close()
        except:
            print('Impossible to create shapefile, sorry.')

    if data_format == 'geojson':
        try:
            collection = geojson.FeatureCollection(features=features)
            output_geojson = open(output_path, 'w')
            output_geojson.write(geojson.dumps(collection))
            output_geojson.close()
        except Exception:
            print('Impossible to create geojson, sorry: %s' % str(Exception))

    print('Geojson creation success!\n')

def export_to_text(x1, y1, u, v, output_path):
    print('\nStart exporting to vector file...')

    with open(output_path, 'w') as f:
        for i in range(len(x1)):
            f.write('%.1f %.1f %.1f %.1f\n' % (x1[i], y1[i], u[i], v[i]))
    print('Text file creation success!\n')

def export_to_npz(u_2d, v_2d, output_path):
    print('\nStart exporting to NPZ file...')
    np.savez(output_path, u=u_2d, v=v_2d)
    print('NPZ file creation success!\n')

def calc_distance(lon1, lat1, lon2, lat2):
    import pyproj
    geod = pyproj.Geod(ellps="WGS84")
    angle1, angle2, distance = geod.inv(lon1, lat1, lon2, lat2)
    return '%0.2f' % distance, '%0.1f' % angle2

def median_filtering(x1, y1, uu, vv, cc, radius=512, total_neighbours=7):
    '''
        Median filtering of resultant ice vectors as a step before deformation calculation
    '''

    fast_ice_th = 5.
    # Get values of vector components
    #uu = x2 - x1
    #vv = y2 - y1

    idx_mask = []
    # Make 2D data of components
    #data = np.vstack((uu, vv)).T

    x1, y1, uu, vv, cc = np.array(x1), np.array(y1), np.array(uu), np.array(vv), np.array(cc)

    #  Radius based filtering
    vector_start_data = np.vstack((x1, y1)).T
    vector_start_tree = sklearn.neighbors.KDTree(vector_start_data)

    for i in range(0, len(x1), 1):
        # If index of element in mask list form 'outliers_filtering' then replace with median
        #if i in mask_proc:
        #    print('Replace with median!')
        req_data = np.array([x1[i], y1[i]]).reshape(1, -1)
        # Getting number of neighbours
        num_nn = vector_start_tree.query_radius(req_data, r=radius, count_only=True)

        # Check number of neighboors
        '''
        if num_nn[0] < total_neighbours:
            idx_mask.append(i)
            cc[i] = 0.
        else:
        '''
        # Apply median filtering
        nn = vector_start_tree.query_radius(req_data, r=radius)
        data = np.vstack((uu[nn[0]], vv[nn[0]])).T

        ####################################################################
        # Loop through all found ice drift vectors to filter not homo
        ####################################################################
        for ii in range(num_nn[0]):
            # Calculate median
            #data[:, 0][ii], data[:, 1][ii]

            # Replace raw with median
            # If not fast ice (> 5 pixels)
            if (np.hypot(uu[i], vv[i]) > fast_ice_th or np.isnan(uu[i]) or np.isnan(vv[i])):
                u_median = np.nanmedian(data[:, 0][ii])
                v_median = np.nanmedian(data[:, 1][ii])
                #u_median = np.nanmean(data[:, 0][ii])
                #v_median = np.nanmean(data[:, 1][ii])
                uu[i], vv[i] = u_median, v_median
            cc[i] = 0

    #tt = list(set(idx_mask))
    #iidx_mask = np.array(tt)

    x1_f = np.array(x1)
    y1_f = np.array(y1)
    uu_f = np.array(uu)
    vv_f = np.array(vv)
    cc_f = np.array(cc)

    return x1_f, y1_f, uu_f, vv_f, cc_f


def calc_deformations(dx, dy, normalization=False, normalization_time=None, cell_size=1.,
                      invert_meridional=True, out_png_name='test.png'):
    '''
    Calculate deformation invariants from X and Y ice drift components

    dx, dy - x and y component of motion (pixels)
    normalization - normalize to time (boolean)
    normalization_time - normalization time (in seconds)
    cell_size - ground meters in a pixel
    invert_meridional - invert y component (boolean)
    '''

    # Cell size factor (in cm)
    cell_size_cm = cell_size * 100.
    cell_size_factor = 1 / cell_size_cm

    m_div = np.empty((dx.shape[0], dx.shape[1],))
    m_div[:] = np.NAN
    m_curl = np.empty((dx.shape[0], dx.shape[1],))
    m_curl[:] = np.NAN
    m_shear = np.empty((dx.shape[0], dx.shape[1],))
    m_shear[:] = np.NAN
    m_tdef = np.empty((dx.shape[0], dx.shape[1],))
    m_tdef[:] = np.NAN

    # Invert meridional component
    if invert_meridional:
        dy = dy * (-1)

    # Normilize u and v to 1 hour
    if not normalization:
        pass
    else:
        # Convert to ground distance (pixels*cell size(m) * 100.)
        dx = dx * cell_size_cm # cm
        dy = dy * cell_size_cm # cm

        # Get U/V components of speed (cm/s)
        dx = dx / normalization_time
        dy = dy / normalization_time

    # Calculate magnitude (speed module) (cm/s)
    mag_speed = np.hypot(dx, dy)

    # Print mean speed in cm/s
    print('Mean speed: %s [cm/s]' % (np.nanmean(mag_speed)))

    #cell_size_factor = 1 / cell_size

    # Test
    #plt.clf()
    #plt.imshow(m_div)

    for i in range(1, dx.shape[0] - 1):
        for j in range(1, dx.shape[1] - 1):
            # div
            if (np.isnan(dx[i, j + 1]) == False and np.isnan(dx[i, j - 1]) == False
                and np.isnan(dy[i - 1, j]) == False and np.isnan(dy[i + 1, j]) == False
                and (np.isnan(dx[i, j]) == False or np.isnan(dy[i, j]) == False)):
                # m_div[i,j] = 0.5*((u_int[i,j + 1] - u_int[i,j - 1])  + (v_int[i + 1,j] - v_int[i - 1,j]))/m_cell_size

                # !Exclude cell size factor!
                m_div[i, j] = cell_size_factor * 0.5 * ((dx[i, j + 1] - dx[i, j - 1])
                                                        + (dy[i - 1, j] - dy[i + 1, j]))
                # print m_div[i,j]

            # Curl
            if (np.isnan(dy[i, j + 1]) == False and np.isnan(dy[i, j - 1]) == False and
                        np.isnan(dx[i - 1, j]) == False and np.isnan(dx[i + 1, j]) == False
                and (np.isnan(dx[i, j]) == False or np.isnan(dy[i, j]) == False)):

                # !Exclude cell size factor!
                m_curl[i, j] = cell_size_factor * 0.5 * (dy[i, j + 1] - dy[i, j - 1]
                                      - dx[i - 1, j] + dx[i + 1, j]) / cell_size

            # Shear
            if (np.isnan(dy[i + 1, j]) == False and np.isnan(dy[i - 1, j]) == False and
                        np.isnan(dx[i, j - 1]) == False and np.isnan(dx[i, j + 1]) == False and
                        np.isnan(dy[i, j - 1]) == False and np.isnan(dy[i, j + 1]) == False and
                        np.isnan(dx[i + 1, j]) == False and np.isnan(dx[i - 1, j]) == False and
                    (np.isnan(dx[i, j]) == False or np.isnan(dy[i, j]) == False)):
                dc_dc = cell_size_factor * 0.5 * (dy[i + 1, j] - dy[i - 1, j])
                dr_dr = cell_size_factor * 0.5 * (dx[i, j - 1] - dx[i, j + 1])
                dc_dr = cell_size_factor * 0.5 * (dy[i, j - 1] - dy[i, j + 1])
                dr_dc = cell_size_factor * 0.5 * (dx[i + 1, j] - dx[i - 1, j])

                # !Exclude cell size factor!
                m_shear[i, j] = np.sqrt(
                    (dc_dc - dr_dr) * (dc_dc - dr_dr) + (dc_dr - dr_dc) * (dc_dr - dr_dc)) / cell_size

                '''
                # Den
                dc_dc = 0.5*(v_int[i + 1,j] - v_int[i - 1,j])
                dr_dr = 0.5*(u_int[i,j + 1] - u_int[i,j - 1])
                dc_dr = 0.5*(v_int[i,j + 1] - v_int[i,j - 1])
                dr_dc = 0.5*(u_int[i + 1,j] - u_int[i - 1,j])

                m_shear[i,j] = np.sqrt((dc_dc -dr_dr) * (dc_dc -dr_dr) + (dc_dr - dr_dc) * (dc_dr - dr_dc))/m_cell_size
                '''

            # Total deformation
            if (np.isnan(m_shear[i, j]) == False and np.isnan(m_div[i, j]) == False):
                m_tdef[i, j] = np.hypot(m_shear[i, j], m_div[i, j])

    # Invert dy back
    if invert_meridional:
        dy = dy * (-1)

    # data = np.vstack((np.ravel(xx_int), np.ravel(yy_int), np.ravel(m_div), np.ravel(u_int), np.ravel(v_int))).T
    divergence = m_div

    # TODO: Plot Test Div
    plt.clf()
    plt.gca().invert_yaxis()

    plt.imshow(divergence, cmap='RdBu', vmin=-0.00008, vmax=0.00008,
               interpolation='nearest', zorder=2) # vmin=-0.06, vmax=0.06,


    # Plot u and v values inside cells (for testing porposes)
    '''
    font_size = .0000003
    for ii in range(dx.shape[1]):
        for jj in range(dx.shape[0]):
            try:
                if not np.isnan(divergence[ii,jj]):
                    if divergence[ii,jj] > 0:
                        plt.text(jj, ii,
                             'u:%.2f\nv:%.2f\n%s ij:(%s,%s)\n%.6f' %
                             (dx[ii,jj], dy[ii,jj], '+', ii, jj, divergence[ii,jj]),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=font_size, color='k')
                    if divergence[ii,jj] < 0:
                        plt.text(jj, ii,
                             'u:%.2f\nv:%.2f\n%s ij:(%s,%s)\n%.6f' %
                             (dx[ii,jj], dy[ii,jj], '-', ii, jj, divergence[ii,jj]),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=font_size, color='k')
                    if divergence[ii,jj] == 0:
                        plt.text(jj, ii,
                             'u:%.2f\nv:%.2f\n%s ij:(%s,%s)\n%.6f' %
                             (dx[ii,jj], dy[ii,jj], '0', ii, jj, divergence[ii,jj]),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=font_size, color='k')

                    if np.isnan(divergence[ii,jj]):
                        plt.text(jj, ii,
                             'u:%.2f\nv:%.2f\n%s ij:(%s,%s)' % 
                             (dx[ii,jj], dy[ii,jj], '-', ii, jj),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=font_size, color='k')

                # Plot arrows on top of the deformation
                xxx = range(dx.shape[1])
                yyy = range(dx.shape[0])
            except:
                pass

    '''
    # Plot drift arrows on the top
    #import matplotlib.cm as cm
    #from matplotlib.colors import Normalize
    # Invert meridional component for plotting
    ddy = dy * (-1)

    #norm = Normalize()
    colors = np.hypot(dx, ddy)
    #print(colors)
    #norm.autoscale(colors)
    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]
    #colormap = cm.inferno

    # Plot arrows on top of the deformation
    xxx = range(dx.shape[1])
    yyy = range(dx.shape[0])

    plt.quiver(xxx, yyy, dx, ddy, colors, cmap='Greys', zorder=3) #'YlOrBr')

    # Invert Y axis
    plt.savefig(out_png_name, bbox_inches='tight', dpi=800)

    curl = m_curl
    shear = m_shear
    total_deform = m_tdef

    # return mag in cm/s
    return mag_speed, divergence, curl, shear, total_deform

# !TODO:
def make_nc(nc_fname, lons, lats, data):
    """
    Make netcdf4 file for deformation (divergence, shear, total deformation), scaled 10^(-4)

    """

    print('\nStart making nc for defo...')

    ds = Dataset(nc_fname, 'w', format='NETCDF4_CLASSIC')
    print(ds.file_format)

    # Dimensions
    y_dim = ds.createDimension('y', lons.shape[0])
    x_dim = ds.createDimension('x', lons.shape[1])
    time_dim = ds.createDimension('time', None)
    #data_dim = ds.createDimension('data', len([k for k in data.keys()]))

    # Variables
    times = ds.createVariable('time', np.float64, ('time',))
    latitudes = ds.createVariable('lat', np.float32, ('y', 'x',))
    longitudes = ds.createVariable('lon', np.float32, ('y', 'x',))

    for var_name in data.keys():
        globals()[var_name] = ds.createVariable(var_name, np.float32, ('y', 'x',))
        globals()[var_name][:, :] = data[var_name]['data']
        globals()[var_name].units = data[var_name]['units']
        globals()[var_name].scale_factor = data[var_name]['scale_factor']

    # Global Attributes
    ds.description = 'Sea ice deformation product'
    ds.history = 'Created ' + time.ctime(time.time())
    ds.source = 'NIERSC/NERSC'

    # Variable Attributes
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    times.units = 'hours since 0001-01-01 00:00:00'
    times.calendar = 'gregorian'

    # Put variables
    latitudes[:, :] = lats
    longitudes[:, :] = lons

    ds.close()

def _create_geotiff(suffix, Array, NDV, xsize, ysize, GeoT, Projection, deformation):
    from osgeo import gdal_array
    DataType = gdal_array.NumericTypeCodeToGDALTypeCode(Array.dtype)

    if type(DataType) != np.int:
        if DataType.startswith('gdal.GDT_') == False:
            DataType = eval('gdal.GDT_' + DataType)

    NewFileName = suffix + '.tif'
    zsize = 1 #Array.shape[0]

    driver = gdal.GetDriverByName('GTiff')

    Array[np.isnan(Array)] = NDV

    DataSet = driver.Create(NewFileName, xsize, ysize, zsize, DataType)

    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projection)#.ExportToWkt())

    # for testing
    # DataSet.SetProjection('PROJCS["NSIDC Sea Ice Polar Stereographic North",GEOGCS["Unspecified datum based upon the Hughes 1980 ellipsoid",DATUM["Not_specified_based_on_Hughes_1980_ellipsoid",SPHEROID["Hughes 1980",6378273,298.279411123061,AUTHORITY["EPSG","7058"]],AUTHORITY["EPSG","6054"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4054"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",70],PARAMETER["central_meridian",-45],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3411"]]')

    #for i in xrange(0, zsize):
    DataSet.GetRasterBand(1).WriteArray(deformation)  # Array[i])
    DataSet.GetRasterBand(1).SetNoDataValue(NDV)

    DataSet.FlushCache()
    return NewFileName

def create_geotiff(suffix, data, NDV, GeoT, Projection):
    ''' Create geotiff file (1 band)'''
    # Get GDAL data type
    dataType = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)
    # NaNs to the no data value
    data[np.isnan(data)] = NDV
    if dataType != 7:
        print(f'####234 {type(dataType)}')
        if dataType.startswith('gdal.GDT_') == False:
            dataType = eval('gdal.GDT_' + dataType)
    newFileName = suffix + '_test.tif'
    cols = data.shape[1]
    rows = data.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newFileName, cols, rows, 1, dataType)
    #outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outRaster.SetGeoTransform(GeoT)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(data)
    outRaster.SetProjection(Projection)
    outband.SetNoDataValue(NDV)
    outband.FlushCache()
    return newFileName


def cc(arguments):
    # BM test flag
    f=0
    # Parse arguments
    #iidx_line, iidx_row, LLi0, LLi1, im1_name, im2_name, pref = arguments
    iidx_line, iidx_row, Lt0, Rt0, Lt1, Rt1, Li0, Ri0, Li1, Ri1, pref, Conf.img1, Conf.img2, itr, itrCnt = arguments

    #print("Processing block: {} from {} ({:.2f}%) at pid={}".format(itr, itrCnt, itr/itrCnt*100, multiprocessing.current_process()))

    if iidx_line is not None:
        # Open two images
        im1 = Conf.img1[Lt0:Rt0, Lt1:Rt1]
        im2 = Conf.img2[Li0:Ri0, Li1:Ri1]

        #####################
        # Filtering
        #####################
        # Median filtering
        if Conf.img_median_filtering:
            # print 'Median filtering'
            # im2 = median(im2, disk(3))
            # im1 = median(im1, disk(3))
            im1 = median(im1, disk(Conf.median_kernel))
            im2 = median(im2, disk(Conf.median_kernel))

        if Conf.img_laplace_filtering:
            im1 = laplace(im1)
            im2 = laplace(im2)

        if Conf.img_gradient_filtering:
            im1 = gradient(im1, disk(3))
            im2 = gradient(im2, disk(3))

        if Conf.img_scharr_filtering:
            # filters.scharr(camera)
            im1 = filters.scharr(im1)
            im2 = filters.scharr(im2)

        ########################
        # End filtering
        ########################

        # Check for black stripes
        flag1 = check_borders(im1)
        flag2 = check_borders(im2)

        # No black borders in the first image
        if flag1 == 0: # and flag2 == 0:
            u_direct, v_direct, result = matching(im1, im2)
            # Peak maximum CC
            cc_max = np.max(result)

            # Get coordinates with offsets
            lline_2, rrow_2 = v_direct + Li0, u_direct +  Li1
            lline_1, rrow_1 = iidx_line, iidx_row


            #ff_out_txt.write('%s, %s, %s, %s, %s, %s, %s, %s' %
            #                 (lline_1, rrow_1, lline_2, rrow_2, u_direct, Li0, v_direct, Li1))
            print(lline_1, rrow_1, lline_2, rrow_2, u_direct, Li0, v_direct, Li1)

            #print('\nCOORDS: %s %s' % (arr_lines_1[i, j], arr_rows_1[i, j]))
            #print('COORDS: %s %s\n' % (arr_lines_2[i, j], arr_rows_2[i, j]))

            # Peaks plot
            if Conf.plot_correlation_peaks:
                plot_peaks(im1, im2, u_direct, v_direct, iidx_line, iidx_row, result, pref,
                           lline_1, rrow_1, lline_2, rrow_2, u_direct, Li0, v_direct, Li1)
                #plot_peaks(im1_bm, im2_bm, uu_bm, vv_bm, iidx_line, iidx_row,
                #           result_bm, 'bm')

            # If all elements are equal
            if np.unique(result).size == 1:
                return np.nan, np.nan, np.nan, np.nan, np.nan

            # If second peak close to first
            flat = result.flatten()
            flat.sort()

            #print('#Flat: %s' % flat)

            #if abs(flat[-1]-flat[-2]) < 0.05:
            #    return np.nan, np.nan, np.nan, np.nan, np.nan

            ret = (lline_1, rrow_1, rrow_2-rrow_1, lline_2-lline_1, cc_max)
            #return lline_1, rrow_1, u_direct, v_direct, cc_max
        else:
            #pass
            # ! Testing (return result in any case)
            ret = (np.nan, np.nan, np.nan, np.nan, np.nan)
            '''
            # if crop images have black stripes
            if flag1 == 1:
                print('IMG_1: %s_%s' % (iidx_line, iidx_row))
                io.imsave('ci_%s_1/black_%s_%s.png' % (Conf.out_fname, iidx_line, iidx_row), im1)
            if flag2 == 1:
                print('IMG_2: %s_%s' % (idx_line, idx_row))
                io.imsave('ci_%s_2/black_%s_%s.png' % (Conf.out_fname, iidx_line, iidx_row), im2)
            '''

    #print("Processed block: {} from {}".format(itr, itrCnt))
    return ret

def apply_anisd(img, gamma=0.25, step=(1., 1.), ploton=False):
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
    kappa = Conf.speckle_filter_parameters[Conf.speckle_filter_name]['kappa']
    niter = Conf.speckle_filter_parameters[Conf.speckle_filter_name]['N']
    option = Conf.speckle_filter_parameters[Conf.speckle_filter_name]['equation']

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()
    # niter

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

    return cv2.convertScaleAbs(imgout)

#################################################################################
#################################################################################
#################################################################################
# MAIN PROGRAM
#################################################################################
#################################################################################
#################################################################################
# run cc_bm_parallel_dev.py ./data/test_kara_01.tif ./data/test_kara_02.tif 64 4 100

import cc_config
import cc_calc_drift
import cc_calc_drift_filter
import cc_calc_defo

#VAS
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # check command line args
    assert (len(sys.argv) == 6), "Expecting 5 arguments: filename1 filename2 block_size search_area grid_step"

    # init config class
    Conf = cc_config.Config()
    Conf.init(f1_name=sys.argv[1], f2_name=sys.argv[2], 
                block_size=int(sys.argv[3]), search_area=int(sys.argv[4]), grid_step=int(sys.argv[5]))
    Conf.self_prepare()

    global_start_time = time.time()

    # Downscale
    if Conf.rescale_apply:
        print('Rescaling...')
        Conf.img1 = rescale(Conf.img1, 1.0 / Conf.rescale_factor)
        Conf.img2 = rescale(Conf.img2, 1.0 / Conf.rescale_factor)
        print('Done!')

    # Image intensity normalization
    if Conf.image_intensity_byte_normalization:
        print('\nImage intensity rescaling (0, 255)...')
        #Conf.img1 = exposure.adjust_log(Conf.img1)
        #Conf.img2 = exposure.adjust_log(Conf.img2)

        # Rescale intensity only
        Conf.img1 = exposure.rescale_intensity(Conf.img1, out_range=(0, 255))
        Conf.img2 = exposure.rescale_intensity(Conf.img2, out_range=(0, 255))

        p2, p98 = np.percentile(Conf.img1, (2, 98))
        Conf.img1 = img_as_ubyte(exposure.rescale_intensity(Conf.img1, in_range=(p2, p98)))

        p2, p98 = np.percentile(Conf.img2, (2, 98))
        Conf.img2 = img_as_ubyte(exposure.rescale_intensity(Conf.img2, in_range=(p2, p98)))

        print('Done!')

    # Normalization
    #print('\n### Laplacian! ###\n')
    #Conf.img1 = cv2.Laplacian(Conf.img1, cv2.CV_64F, ksize=19)
    #Conf.img2 = cv2.Laplacian(Conf.img2, cv2.CV_64F, ksize=19)


    # Speckle filtering
    if Conf.speckle_filtering:
        assert (Conf.speckle_filtering and (Conf.speckle_filter_name in Conf.speckle_filter_name)), \
            '%s error: appropriate processor is not found' % Conf.speckle_filter_name

        print('\nSpeckle filtering with %s\n' % Conf.speckle_filter_name)

        if Conf.speckle_filter_name == 'Anisd':
            Conf.img1 = apply_anisd(Conf.img1, gamma=0.25, step=(1., 1.), ploton=False)
            Conf.img2 = apply_anisd(Conf.img2, gamma=0.25, step=(1., 1.), ploton=False)

    #####################
    ### Calculate Drift ###
    #####################
    print('\nStart multiprocessing...')
    nb_cpus = 10

    height, width = Conf.img1.shape
    print('Image size Height: %s px Width: %s px' % (height, width))

    # init drift calculator class
    Calc = cc_calc_drift.CalcDrift(Conf, Conf.img1, Conf.img2)
    Calc.create_arguments(height, width)

    # arg generator
    argGen = ((i) for i in range(Calc.Count))

    pool = multiprocessing.Pool(processes=nb_cpus)
    # calculate
    results = pool.map(Calc.calculate_drift, argGen)
    pool.close()
    pool.join()

    print('Done!')
    exec_t = (time.time() - global_start_time) / 60.
    print('Calculated in--- %.1f minutes ---' % exec_t)

    pref = 'dm'
    '''
    print('\nPlotting...')
    try:
        plot_arrows_from_list(pref, '%s/%s_%s_01.png' % (Conf.res_dir, pref, Conf.out_fname),
                            Conf.img1, results, arrwidth=0.0021, headwidth=2.5, flag_color=True)
        plot_arrows_from_list(pref, '%s/%s_%s_02.png' % (Conf.res_dir, pref, Conf.out_fname),
                            Conf.img2, results, arrwidth=0.0021, headwidth=2.5, flag_color=True)
        print('Plot end!')
    except:
        print('Plot FAULT!')
    '''

    #####################
    #### Filter vectors ####
    #####################
    print('\nStart outliers filtering...')

    # init result filtering class
    Filter = cc_calc_drift_filter.CalcDriftFilter(Conf)
    # filter
    Cnt = Filter.filter_outliers(results)

    # Filter land vectors
    print('\nLand mask filtering...')
    land_filtered_vectors = Filter.filter_land()
    print('Done\n')

    print('Done!')
    print('\nNumber of vectors: \n Unfiltered: %d   Filtered: %d\n' %
          (Cnt[0], Cnt[1]))

    print('\nPlotting...')
    plot_arrows('%s/01_spikes_%s_%s.png' % (Conf.res_dir, pref, Conf.out_fname), Conf.img1, Filter.xxx_f, Filter.yyy_f, Filter.uuu_f, Filter.vvv_f, Filter.ccc_f,
                arrwidth=0.002, headwidth=5.5, flag_color=True)

    plot_arrows('%s/02_spikes_%s_%s.png' % (Conf.res_dir, pref, Conf.out_fname), Conf.img2, Filter.xxx_f, Filter.yyy_f, Filter.uuu_f, Filter.vvv_f, Filter.ccc_f,
                arrwidth=0.002, headwidth=5.5, flag_color=True)

    #####################
    #### Defo calculate ####
    #####################
    print('\n### Start deformation calculation...')

    # init defo calculator class
    Defo = cc_calc_defo.CalcDefo(Conf, Calc, Filter)
    # calculate deformation from the 2D arrays
    mag_speed, divergence, curl, shear, total_deform, u_2d, v_2d = Defo.calculate_defo()

    print('\n### Success!\n')

    #########################
    # EXPORT TO GEO-FORMATS
    #########################

    files_pref = '%spx' % Conf.grid_step

    try:
        os.makedirs('%s/vec' % Conf.res_dir)
    except:
        pass

    try:
        os.makedirs('%s/defo/nc' % Conf.res_dir)
    except:
        pass

    # Vector
    export_to_vector(Conf.f1_name, Filter.xxx_f, Filter.yyy_f, Filter.uuu_f, Filter.vvv_f,
                    '%s/vec/%s_ICEDRIFT_%s.json' % (Conf.res_dir, files_pref, Conf.out_fname),
                    gridded=False, data_format='geojson')

    # Text file
    export_to_text(Filter.xxx_f, Filter.yyy_f, Filter.uuu_f, Filter.vvv_f,
                     '%s/vec/%s_ICEDRIFT_%s.txt' % (Conf.res_dir, files_pref, Conf.out_fname))

    # NPZ file
    export_to_npz(u_2d, v_2d,
                  '%s/vec/%s_ICEDRIFT_%s.npz' % (Conf.res_dir, files_pref, Conf.out_fname))

    ################
    # Geotiff
    ################
    print('\nStart making geotiff..')

    try:
        os.makedirs('%s/defo/gtiff' % Conf.res_dir)
    except:
        pass

    scale_factor = 1

    divergence_gtiff = divergence * scale_factor
    GeoT = (Calc.geotransform[0] - Conf.grid_step/2.*Calc.pixelHeight, Conf.grid_step*Calc.pixelWidth, 0.,
            Calc.geotransform[3] + Conf.grid_step/2.*Calc.pixelHeight, 0., Conf.grid_step*Calc.pixelHeight)
    NDV = np.nan

    # Get projection WKT
    gd_raster = gdal.Open(Conf.f1_name)
    Projection = gd_raster.GetProjection()

    #create_geotiff('%s/defo/gtiff/%s_ICEDIV_%s' % (Conf.res_dir, files_pref, Conf.out_fname),
    #               divergence_gtiff, NDV, u_2d.shape[0], u_2d.shape[1], GeoT, Projection, divergence_gtiff)

    create_geotiff('%s/defo/gtiff/%s_ICEDIV_%s' % (Conf.res_dir, files_pref, Conf.out_fname), divergence_gtiff, NDV, GeoT, Projection)

    #####################
    # Shear
    #####################
    shear_gtiff = shear * scale_factor
    GeoT = (Calc.geotransform[0] - Conf.grid_step / 2. * Calc.pixelHeight, Conf.grid_step * Calc.pixelWidth, 0.,
            Calc.geotransform[3] + Conf.grid_step / 2. * Calc.pixelHeight, 0., Conf.grid_step * Calc.pixelHeight)
    NDV = np.nan

    # Get projection WKT
    gd_raster = gdal.Open(Conf.f1_name)
    Projection = gd_raster.GetProjection()

    create_geotiff('%s/defo/gtiff/%s_ICESHEAR_%s' % (Conf.res_dir, files_pref, Conf.out_fname), shear_gtiff, NDV,
                   GeoT, Projection)

    ################
    # END Geotiff
    ################

    ############
    # Netcdf
    ############
    dict_deformation = {'ice_speed':  {'data': mag_speed, 'scale_factor': 1., 'units': 'cm/s'},
                       'ice_divergence': {'data': divergence, 'scale_factor': scale_factor, 'units': '1/h'},
                       'ice_curl': {'data': curl, 'scale_factor': scale_factor, 'units': '1/h'},
                       'ice_shear': {'data': shear, 'scale_factor': scale_factor, 'units': '1/h'},
                       'total_deformation': {'data': total_deform, 'scale_factor': scale_factor, 'units': '1/h'}}

    print('\nStart making netCDF for ice deformation...\n')
    make_nc('%s/defo/nc/%s_ICEDEF_%s.nc' % (Conf.res_dir, files_pref, Conf.out_fname),
            Calc.lon_2d, Calc.lat_2d, dict_deformation)
    print('Success!\n')
    ############
    # END Netcdf
    ############

    ############################
    # END EXPORT TO GEO-FORMATS
    ############################


    # Calc_img_entropy
    calc_img_entropy = False

    #ent_spikes_dm_S1A_EW_GRDM_1SDH_20150114T133134_20150114T133234_004168_0050E3_8C66_HV_S1A_EW_GRDM_1SDH_20150115T025040_20150115T025140_004176_005114_5C27_HV
    d1 = re.findall(r'\d\d\d\d\d\d\d\d\w\d\d\d\d\d\d', Conf.f1_name)[0]
    d2 = re.findall(r'\d\d\d\d\d\d\d\d\w\d\d\d\d\d\d', Conf.f2_name)[0]

    # Calculate entropy
    if calc_img_entropy:
        print('Calculate entropy')
        plt.clf()
        from skimage.util import img_as_ubyte
        from skimage.filters.rank import entropy
        entr_Conf.img1 = entropy(Conf.img1, disk(16))
        # xxx_f, yyy_f
        ff = open('%s/entropy/ent_NCC_%s_%s.txt' % (Conf.res_dir, d1, d2), 'w')
        for i in range(len(xxx_f)):
            ff.write('%7d %7.2f\n' % (i+1, np.mean(entr_Conf.img1[yyy_f[i]-Conf.grid_step:yyy_f[i]+Conf.grid_step,
                                        xxx_f[i]-Conf.grid_step:xxx_f[i]+Conf.grid_step])))
        ff.close()

        # TODO:

        plt.imshow(entr_Conf.img1, cmap=plt.cm.get_cmap('hot', 10))
        plt.colorbar()
        plt.clim(0, 10);

        plt.savefig('%s/entropy/img/ent_NCC_%s_%s.png' % (Conf.res_dir, d1, d2), bbox_inches='tight', dpi=300)

    # END