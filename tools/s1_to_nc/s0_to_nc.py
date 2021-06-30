from netCDF4 import Dataset
import numpy as np
from numpy import inf
from osgeo import gdal, osr
import time
import sys
import os
from s1denoise import Sentinel1Image
from s1denoise.utils import (cost, fit_noise_scaling_coeff, get_DOM_nodeValue, fill_gaps)

def make_nc(nc_fname, lons, lats, data):
    """
    Make netcdf4 file with sigma0

    """

    print('\nStart making nc...')

    if os.path.isfile(nc_fname):
        print('\nRemoving old NC file...')
        os.remove(nc_fname)
        print('Done\n')

    ds = Dataset('%s' % nc_fname, 'w', format='NETCDF4_CLASSIC')
    print(ds.file_format)


    # Dimensions
    y_dim = ds.createDimension('y', lons.shape[0])
    x_dim = ds.createDimension('x', lons.shape[1])
    time_dim = ds.createDimension('time', None)
    #data_dim = ds.createDimension('data', len([k for k in data.keys()]))

    # Variables
    times = ds.createVariable('time', np.float64, ('time',))
    latitudes = ds.createVariable('lat', np.float32, ('y', 'x',), zlib=True, least_significant_digit=5)
    longitudes = ds.createVariable('lon', np.float32, ('y', 'x',), zlib=True, least_significant_digit=5)

    for var_name in data.keys():
        globals()[var_name] = ds.createVariable(var_name, np.float32, ('y', 'x',),
                                                zlib=True, least_significant_digit=2)
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

def get_data(f1_name):
    gd_raster = gdal.Open(f1_name)
    geotransform = gd_raster.GetGeoTransform()
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(gd_raster.GetProjection())
    new_cs = osr.SpatialReference()
    new_cs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[-1]

    band = gd_raster.GetRasterBand(1)
    width = band.XSize
    height = band.YSize

    # Create 2D matrices for data
    data_2d = band.ReadAsArray()

    # Lats and lons
    lon_2d = np.empty((len(range(0, height, 1)), (len(range(0, width, 1)))))
    lon_2d[:] = np.nan

    lat_2d = np.empty((len(range(0, height, 1)), (len(range(0, width, 1)))))
    lat_2d[:] = np.nan

    # 2D indexes for all elements
    idxs_2d = []

    for i, idx_row in enumerate(range(lat_2d.shape[0])):
        for j, idx_col in enumerate(range(lat_2d.shape[1])):
            # Convert x and y to lon lat
            xx1 = geotransform[0] + idx_col * pixelWidth
            yy1 = geotransform[3] + idx_row * pixelHeight

            latlon = transform.TransformPoint(float(xx1), float(yy1))
            ilon = latlon[0]
            ilat = latlon[1]
            lon_2d[i, j] = ilon
            lat_2d[i, j] = ilat

    gd_raster = None

    return data_2d, lon_2d, lat_2d

def get_data_zip(f1_name, polarizations):
    '''

    :param f1_name: Sentinel-1 GRD Level1 zip file
    :return:
    '''

    s1 = Sentinel1Image(f1_name)

    inc = s1['incidence_angle']
    angularDependency = {'HH': -0.200, 'HV': -0.025}

    data = {}

    print('\nReading bands')
    for pol in polarizations:
        band_name = 'sigma0_%s' % pol
        print('\nBand: %s' % band_name)

        data_pol = s1.get_raw_sigma0_full_size(polarization=pol)

        # Remove noise if cross-polarization
        if pol == 'HV' or pol == 'VH':
            # nominal scene center angle for S-1 EW mode: 34.5 degree
            data_pol_db = 10 * np.log10(data_pol) - (angularDependency[pol] * (inc - 34.5))

            # Thermal noise removal
            #print('\nRemoving thermal noise...')
            #esa_nesz = s1.get_nesz_full_size(polarization=pol)
            #data_pol= esa_nesz
            #data_pol = fill_gaps(data_pol, data_pol <= 0)
        else:
            data_pol_db = 10 * np.log10(data_pol) - (angularDependency[pol] * (inc - 34.5))

        #data_pol = 10 * np.log10(data_pol)
        data_pol_db[data_pol_db == -inf] = np.nan

        data[band_name] = {}
        data[band_name]['data'] = data_pol_db
        data[band_name]['scale_factor'] = 1.
        data[band_name]['units'] = 'dB'
    print('Done.\n')

    print('\nGetting lons, lats...')
    lons, lats = s1.get_geolocation_grids()
    print('Done.\n')

    return data, lons, lats


polarizations = ['HH', 'HV'] #, 'HV', 'VV', 'VH']
units = ['dB'] #, 'dB', 'dB', 'dB']
pars = ['s0'] #, 's0', 's0', 's0']
scale_f = [1.] #, 1., 1., 1.]

d_data = {}

#f1_name = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridge_case_18042021/s1_geotiff/ups_HH_S1A_EW_GRDM_1SDH_20210419T214554_20210419T214654_037525_046CC1_F83C.tiff'

f1_name = sys.argv[1]
out_folder = sys.argv[2]

# Read GeoTIFF
if f1_name.endswith('tiff') or f1_name.endswith('tif'):
    data_2d, lon_2d, lat_2d = get_data(f1_name)

# Read ZIP file
if f1_name.endswith('zip'):
    d_data, lon_2d, lat_2d = get_data_zip(f1_name, polarizations)

#for i in range(len(polarizations)):
#    ikey = '%s_%s' % (pars[i], polarizations[i])
#    d_data[ikey] = {'data': data_2d, 'scale_factor': scale_f[i], 'units':  units[i]}

nc_fname = '%s/%s.nc' % (out_folder, os.path.basename(f1_name).split('.')[0])

make_nc(nc_fname, lon_2d, lat_2d, d_data)
