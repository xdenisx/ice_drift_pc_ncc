from netCDF4 import Dataset
import numpy as np
from osgeo import gdal, osr
import time
import sys

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

polarizations = ['HH'] #, 'HV', 'VV', 'VH']
units = ['dB'] #, 'dB', 'dB', 'dB']
pars = ['s0'] #, 's0', 's0', 's0']
scale_f = [1.] #, 1., 1., 1.]

d_data = {}

#f1_name = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridge_case_18042021/s1_geotiff/ups_HH_S1A_EW_GRDM_1SDH_20210419T214554_20210419T214654_037525_046CC1_F83C.tiff'

f1_name = sys.argv[1]

data_2d, lon_2d, lat_2d = get_data(f1_name)

for i in range(len(polarizations)):
    ikey = '%s_%s' % (pars[i], polarizations[i])
    d_data[ikey] = {'data': data_2d, 'scale_factor': scale_f[i], 'units':  units[i]}

make_nc('%s.nc' % os.path.basename(f1_name).split('.')[0], lon_2d, lat_2d, d_data)