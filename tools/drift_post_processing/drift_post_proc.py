import matplotlib.pyplot as plt
from pyproj import Proj, transform
import pyproj
from scipy.io import loadmat
import sys
import os
import datetime
import numpy as np
import geojson
from osgeo import gdal, osr, gdal_array, ogr, gdalconst
import sklearn.neighbors
import shapefile as sf
import re
import pyproj
import warnings
import os
from datetime import datetime, timedelta
sys.path.append("/data/rrs/seaice/esa_rosel/code_dev/ice_drift_pc_ncc/tools/geolocation_grid")
from LocationMapping import LocationMapping
warnings.filterwarnings("ignore")


class driftField:
    '''
    Class for ice drift field processing
    '''

    def __init__(self, file_path=None, path_to_tiff=None, path_to_tiff2=None,
                 land_mask_path='/home/denis/git/dev/ice_drift_pc_ncc/data/ne_50m_land.shp', step_pixels=50):
        formats = {'mat': {}, 'nc': {}}

        self.step_pixels = step_pixels
        self.path_to_tiff = path_to_tiff
        self.path_to_tiff2 = path_to_tiff2
        self.land_mask_path = land_mask_path

        if not self.path_to_tiff is None:
            pref = 'tiff_data'
            setattr(self, pref, {})
            try:
                print(f'Reading tiff file {self.path_to_tiff}...')
                self.read_tiff_data(self.path_to_tiff, pref)
                print('Done.')
            except Exception as e:
                print(f"Error while reading tiff: {e}")
        else:
            pass

        if not self.path_to_tiff2 is None:
            pref = 'tiff_data2'
            setattr(self, pref, {})
            try:
                print(f'Reading tiff file {self.path_to_tiff2}...')
                self.read_tiff_data(self.path_to_tiff2, pref)
                print('Done.')
            except Exception as e:
                print(f"Error while reading tiff: {e}")
        else:
            pass

        self.file_path = file_path
        self.get_dt_str()

        root, extension = os.path.splitext(file_path)
        self.file_format = extension[1:]

        if self.file_format in formats.keys():
            print(f'\nFile format: {self.file_format}')
            func = getattr(self, f'reader_{self.file_format}')
            self.dataset = func()
            print('The data has been successefully readed.')

            print(f'\nFiltering outliers...')
            self.outliers_filtering()
            print('Done.')
        else:
            print(f'Sorry, format {self.file_format} is not currently supported')

    def get_dt_str(self):
        '''
        Get time difference form dates in string format
        '''
        dt1, dt2 = re.findall('\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', self.file_path)
        self.dt1 = dt1
        self.dt2 = dt2

        self.dt1 = '%s-%s-%sT%s:%s:%s' % (dt1[0:4], dt1[4:6], dt1[6:8],
                                          dt1[9:11], dt1[11:13], dt1[13:15])

        self.dt1 = datetime.strptime(self.dt1, '%Y-%m-%dT%H:%M:%S')

        self.dt2 = '%s-%s-%sT%s:%s:%s' % (dt2[0:4], dt2[4:6], dt2[6:8],
                                          dt2[9:11], dt2[11:13], dt2[13:15])
        self.dt2 = datetime.strptime(self.dt2, '%Y-%m-%dT%H:%M:%S')
        self.time_diff = (self.dt2 - self.dt1).total_seconds()

    def get_geot(self, y0, x0, pixel_width, pixel_height, shift_x, shift_y):
        '''
        Create transformation object for data with desired grid cell size
        '''

        ll_corner_x = self.tiff_data['gt'][0] + (min(y0)) * pixel_width
        ll_corner_y = self.tiff_data['gt'][3] + (min(x0)) * pixel_height

        self.data['geot'] = (ll_corner_x + shift_x, self.step_pixels * pixel_width, 0.,
                             ll_corner_y + shift_y, 0., self.step_pixels * pixel_height)

    def read_tiff_data(self, file_path, pref):
        '''
        Read and store GeoTIFF metadata
        '''
        setattr(self, pref, {})
        image = gdal.Open(file_path)
        lm = LocationMapping(image.GetGeoTransform(), image.GetProjection())
        X = np.arange(image.RasterXSize)
        Y = np.arange(image.RasterYSize)
        X, Y = np.meshgrid(X, Y)
        lat, lon = lm.raster2LatLon(X.reshape((-1)), Y.reshape((-1)))
        getattr(self, pref)['lats'] = lat.reshape(image.ReadAsArray()[0].shape[0], image.ReadAsArray()[0].shape[1])
        getattr(self, pref)['lons'] = lon.reshape(image.ReadAsArray()[0].shape[0], image.ReadAsArray()[0].shape[1])
        gt = image.GetGeoTransform()
        getattr(self, pref)['gt'] = gt
        getattr(self, pref)['proj'] = image.GetProjection()
        getattr(self, pref)['shape'] = (image.RasterYSize, image.RasterXSize)
        getattr(self, pref)['data'] = image.ReadAsArray()[0]
        del image

    def reader_mat(self, glob_var_name='OUT'):
        '''
        Read data in matlab format
        '''
        if self.path_to_tiff is not None:
            data = {}
            mat = loadmat(self.file_path)
            self.data_raw = mat

            dtInHours = mat[glob_var_name]['dtInHours'][0][0][0][0]
            dtSeconds = timedelta(seconds=int(dtInHours) * 3600.).total_seconds()

            cor_coeff = mat[glob_var_name]['r'][0][0]
            cor_coeff = np.array([x[0] for x in cor_coeff])
            data['cc'] = cor_coeff

            v_complex = mat[glob_var_name]['v'][0][0]
            v_complex = np.array([x[0] for x in v_complex])

            u = v_complex.real  # longitudinal [m/s]
            v = v_complex.imag  # latitudinal [m/s]

            p1c = mat[glob_var_name]['p1rc'][0][0][::, 0]
            p1r = mat[glob_var_name]['p1rc'][0][0][::, 1]

            # DX and DY in numpy axis order (reversed!)
            mer_raw = mat[glob_var_name]['DX'][0][0]
            zonal_raw = mat[glob_var_name]['DY'][0][0]

            dy = np.array([x[0] for x in mer_raw])
            dx = np.array([x[0] for x in zonal_raw])

            data['y0'] = p1r
            data['x0'] = p1c
            data['dy'] = dy
            data['dx'] = dx

            yy1 = p1r + dy
            xx1 = p1c + dx

            xx1[xx1 < 0] = 9999
            yy1[yy1 < 0] = 9999

            data['y1'] = yy1.astype('float32')
            data['x1'] = xx1.astype('float32')

            y0_unq = np.unique(data['y0'])
            x0_unq = np.unique(data['x0'])
            ny, nx = len(y0_unq), len(x0_unq)

            # Make 2D arrays from vectors
            x_min = np.nanmin(data['x0'])
            x_max = np.nanmax(data['x0'])

            y_min = np.nanmin(data['y0'])
            y_max = np.nanmax(data['y0'])

            stp = self.step_pixels
            xx = range(x_min, x_max + stp, stp)
            yy = range(y_min, y_max + stp, stp)

            ny_un, nx_un = np.unique(yy), np.unique(xx)
            ny, nx = len(ny_un), len(nx_un)
            yy_2d, xx_2d = np.meshgrid(ny_un, nx_un)

            ddy_2d = np.empty((nx, ny,))
            ddy_2d[:] = np.nan

            ddx_2d = np.empty((nx, ny,))
            ddx_2d[:] = np.nan

            yy1_2d = np.empty((nx, ny,))
            yy1_2d[:] = np.nan

            xx1_2d = np.empty((nx, ny,))
            xx1_2d[:] = np.nan

            cc_2d = np.empty((nx, ny,))
            cc_2d[:] = np.nan

            for i in range(data['y0'].shape[0]):
                idx = np.argwhere((yy_2d == data['y0'][i]) & (xx_2d == data['x0'][i]))
                ii, jj = idx[0][0], idx[0][1]
                ddy_2d[ii, jj] = dy[i]
                ddx_2d[ii, jj] = dx[i]
                yy1_2d[ii, jj] = yy1[i]
                xx1_2d[ii, jj] = xx1[i]
                cc_2d[ii, jj] = cor_coeff[i]

            data['y0_2d'] = yy_2d
            data['x0_2d'] = xx_2d
            data['dy_2d'] = ddy_2d
            data['dx_2d'] = ddx_2d
            data['y1_2d'] = yy1_2d.astype('float32')
            data['x1_2d'] = xx1_2d.astype('float32')
            data['cc_2d'] = cc_2d

            self.data = data

            # Create geo transform object for the data
            pixel_width = self.tiff_data['gt'][1]
            pixel_height = self.tiff_data['gt'][-1]

            shift_x = -pixel_width * self.step_pixels / 2
            shift_y = -pixel_height * self.step_pixels / 2

            self.get_geot(data['y0'], data['x0'], pixel_width, pixel_height, shift_x, shift_y)

        else:
            print('Could not process the data. Please provide path to geotiff file and grid cell size.')

    def calc_defo(self, normalization=True, invert_meridional=True, fill_nan=False,
                  out_path='.', defo_params=['div', 'shear', 'curl', 'tdef', 'mag_speed'], filtered=True,
                  land_mask=True):
        '''
        Calculate deformation invariants from ice drift components
        '''

        if self.path_to_tiff is not None:
            ddy = self.data['dy_2d'].copy().astype('float')
            ddx = self.data['dx_2d'].copy().astype('float')

            if filtered:
                ddy[self.data['outliers_mask'] == False] = np.nan
                ddx[self.data['outliers_mask'] == False] = np.nan
            else:
                pass

            if land_mask:
                self.rasterize_shp()
            else:
                pass

            # Invert meridional component
            if invert_meridional:
                ddy = ddy * (-1)

            m_div = np.empty((self.data['dy_2d'].shape[0], self.data['dy_2d'].shape[1],))
            m_div[:] = np.NAN
            m_curl = np.empty((self.data['dy_2d'].shape[0], self.data['dy_2d'].shape[1],))
            m_curl[:] = np.NAN
            m_shear = np.empty((self.data['dy_2d'].shape[0], self.data['dy_2d'].shape[1],))
            m_shear[:] = np.NAN
            m_tdef = np.empty((self.data['dy_2d'].shape[0], self.data['dy_2d'].shape[1],))
            m_tdef[:] = np.NAN

            raster_cell_size_cm = 100 * (self.tiff_data['gt'][1] +
                                         abs(self.tiff_data['gt'][-1])) / 2
            defo_cell_size_cm = self.step_pixels * raster_cell_size_cm
            print(f'Raster cell size (cm): {raster_cell_size_cm}')
            print(f'Deformation product grid step size (cm) {defo_cell_size_cm}')

            if normalization:
                # Get U/V components of speed (cm/sec)
                ddx = ddx * raster_cell_size_cm
                ddy = ddy * raster_cell_size_cm
                ddx = ddx / self.time_diff
                ddy = ddy / self.time_diff
            else:
                pass

            # Calculate magnitude (speed module) (cm/s)
            mag_speed = np.hypot(ddx, ddy)

            # Print mean speed in cm/s
            print(f'Mean speed: {np.nanmean(mag_speed)} cm/s')

            # Convert from seconds to hours
            ddx = ddx * 3600
            ddy = ddy * 3600

            for i in range(1, ddx.shape[0] - 1):
                for j in range(1, ddx.shape[1] - 1):
                    # div
                    if (np.isnan(ddx[i, j + 1]) == False and np.isnan(ddx[i, j - 1]) == False and
                            np.isnan(ddy[i - 1, j]) == False and np.isnan(ddy[i + 1, j]) == False and
                            (np.isnan(ddx[i, j]) == False or np.isnan(ddy[i, j]) == False)):
                        m_div[i, j] = 0.5 * ((ddx[i, j + 1] - ddx[i, j - 1]) +
                                             (ddy[i - 1, j] - ddy[i + 1, j])) / defo_cell_size_cm

                    # Curl
                    if (np.isnan(ddy[i, j + 1]) == False and np.isnan(ddy[i, j - 1]) == False and
                            np.isnan(ddx[i - 1, j]) == False and np.isnan(ddx[i + 1, j]) == False and
                            (np.isnan(ddx[i, j]) == False or np.isnan(ddy[i, j]) == False)):
                        m_curl[i, j] = 0.5 * (ddy[i, j + 1] - ddy[i, j - 1] -
                                              ddx[i - 1, j] + ddx[i + 1, j]) / defo_cell_size_cm

                    # Shear
                    if (np.isnan(ddy[i + 1, j]) == False and np.isnan(ddy[i - 1, j]) == False and
                            np.isnan(ddx[i, j - 1]) == False and np.isnan(ddx[i, j + 1]) == False and
                            np.isnan(ddy[i, j - 1]) == False and np.isnan(ddy[i, j + 1]) == False and
                            np.isnan(ddx[i + 1, j]) == False and np.isnan(ddx[i - 1, j]) == False and
                            (np.isnan(ddx[i, j]) == False or np.isnan(ddy[i, j]) == False)):
                        dc_dc = 0.5 * (ddy[i + 1, j] - ddy[i - 1, j])
                        dr_dr = 0.5 * (ddx[i, j - 1] - ddx[i, j + 1])
                        dc_dr = 0.5 * (ddy[i, j - 1] - ddy[i, j + 1])
                        dr_dc = 0.5 * (ddx[i + 1, j] - ddx[i - 1, j])

                        m_shear[i, j] = np.sqrt(
                            (dc_dc - dr_dr) * (dc_dc - dr_dr) +
                            (dc_dr - dr_dc) * (dc_dr - dr_dc)) / defo_cell_size_cm

                    # Total deformation
                    if not np.isnan(m_shear[i, j]) and not np.isnan(m_div[i, j]):
                        m_tdef[i, j] = np.hypot(m_shear[i, j], m_div[i, j])

            if land_mask:
                print('\nApplying land mask...')
                self.data['mag_speed'] = mag_speed
                self.data['mag_speed'][self.data['land_mask'] == 255] = np.nan
                self.data['div'] = m_div
                self.data['div'][self.data['land_mask'] == 255] = np.nan
                self.data['curl'] = m_curl
                self.data['curl'][self.data['land_mask'] == 255] = np.nan
                self.data['shear'] = m_shear
                self.data['shear'][self.data['land_mask'] == 255] = np.nan
                self.data['tdef'] = m_tdef
                self.data['tdef'][self.data['land_mask'] == 255] = np.nan
                print('Done.')
            else:
                self.data['mag_speed'] = mag_speed
                self.data['div'] = m_div
                self.data['curl'] = m_curl
                self.data['shear'] = m_shear
                self.data['tdef'] = m_tdef

            # Save to geotiff files
            os.makedirs(f'{out_path}', exist_ok=True)
            base_fname = os.path.basename(self.file_path)[:-4]
            for par in defo_params:
                self.export_geotiff(self.data[par], f'{out_path}/{par}_{base_fname}.tiff')
        else:
            print('Could not process the data. Please provide a path to geotiff file and grid cell size.')

    def export_geotiff(self, data=None, fname='out.tif'):
        if not data is None:
            dataType = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)
            data[np.isnan(data)] = np.nan
            if type(dataType) != np.int:
                if dataType.startswith('gdal.GDT_') == False:
                    dataType = eval('gdal.GDT_' + dataType)

            cols = data.shape[1]
            rows = data.shape[0]

            driver = gdal.GetDriverByName('GTiff')
            outRaster = driver.Create(fname, cols, rows, 1, dataType)
            outRaster.SetGeoTransform(self.data['geot'])
            outband = outRaster.GetRasterBand(1)
            outband.WriteArray(data)
            outRaster.SetProjection(self.tiff_data['proj'])
            outband.SetNoDataValue(np.nan)
            outband.FlushCache()
            del outband
        else:
            print('No data to export.')

    def export_vector(self, data_format='geojson', out_path='.', filtered=True):
        '''
        Export vetors to geojson/shp format
        '''

        ddy = self.data['dy_2d'].copy()
        ddx = self.data['dx_2d'].copy()
        ddy[self.data['outliers_mask'] == False] = np.nan
        ddx[self.data['outliers_mask'] == False] = np.nan

        '''
        Test plotting 
        
        img1 = self.tiff_data['data']
        plt.clf()
        plt.imshow(img1, cmap='gray')
        ddy = self.data['dy_2d'].copy()
        ddx = self.data['dx_2d'].copy()
        ddy[self.data['outliers_mask'] == False] = np.nan
        ddx[self.data['outliers_mask'] == False] = np.nan

        plt.quiver(self.data['y0_2d'],
                   self.data['x0_2d'],
                   ddy,
                   ddx,
                   self.data['cc_2d'], angles='xy', scale_units='xy', scale=1, alpha=0.7, width=0.0015)
        plt.savefig('/data/rrs/seaice/esa_rosel/L_C_pairs/exp_bac/output/test.png', dpi=300)
        
        '''


        if hasattr(self, 'tiff_data'):
            geod = pyproj.Geod(ellps='WGS84')

            if data_format not in ['geojson', 'shp']:
                print('Invalid format')
                return 0

            if data_format == 'shp':
                w = sf.Writer('%s.shp' % output_path, sf.POLYLINE)
                w.field('id', 'C', '40')
                w.field('lat1', 'C', '40')
                w.field('lon1', 'C', '40')
                w.field('lat2', 'C', '40')
                w.field('lon2', 'C', '40')
                w.field('drift_m', 'C', '40')
                w.field('direction', 'C', '40')

            if data_format == 'geojson':
                print('Geojson format!')
                features = []

            x0, y0 = self.data['x0_2d'].copy(),\
                     self.data['y0_2d'].copy()

            x1, y1 = np.rint(self.data['x1_2d'].copy()).astype('int'),\
                     np.rint(self.data['y1_2d'].copy()).astype('int')

            if filtered == True:
                print('\nmasking...')
                x1[self.data['outliers_mask']==False] = -9999
                y1[self.data['outliers_mask']==False] = -9999
                print('done.\n')
            else:
                pass

            for i in range(y1.shape[0]):
                for j in range(y1.shape[1]):

                    if not np.isnan(ddy[i,j]) and not np.isnan(ddx[i,j]):
                        lon0 = self.tiff_data['lons'][x0[i, j], y0[i, j]]
                        lat0 = self.tiff_data['lats'][x0[i, j], y0[i, j]]
                        #try:
                        lon1 = self.tiff_data['lons'][x1[i, j], y1[i, j]]
                        lat1 = self.tiff_data['lats'][x1[i, j], y1[i, j]]

                        try:
                            az, az2, mag = geod.inv(lon0, lat0,
                                                    lon1, lat1)
                            mag = float(mag)
                            if az <= 180.0:
                                az = az + 360.0
                        except:
                            mag, az = 999., 999.

                        if data_format == 'shp':
                            w.line([[[lon1, lat1], [lon2, lat2]]])
                            w.record(str(i),
                                     str(lat0), str(lon0),
                                     str(lat1), str(lon1),
                                     str(mag), str(az))

                        if data_format == 'geojson':
                            if lon0 == lon1 and lat0 == lat1:
                                ft = geojson.Feature(geometry=geojson.Point([lon0, lat0]),
                                                     properties={'id': str(i + j),
                                                                 'lat1': lat0,
                                                                 'lon1': lon0,
                                                                 'drift_m': 0.,
                                                                 'azimuth': None})
                            else:
                                ft = geojson.Feature(geometry=geojson.LineString([(lon0, lat0), (lon1, lat1)]),
                                                     properties={'id': str(i + j),
                                                                 'lat1': lat0,
                                                                 'lon1': lon0,
                                                                 'lat2': lat1,
                                                                 'lon2': lon1,
                                                                 'drift_m': mag,
                                                                 'azimuth': az})
                            features.append(ft)

                    else:
                        pass
                        #except:
                        #    pass

            os.makedirs(f'{out_path}', exist_ok=True)
            if data_format == 'shp':
                try:
                    # create the PRJ file
                    print(output_path)
                    prj = open(f'{fname[:-4]}.prj', 'w')
                    prj.write(
                        '''GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]''')
                    prj.close()
                    w.save(f'{fname}')
                except Exception as e:
                    print(f"Error while saving shapefile: {e}")

            if data_format == 'geojson':
                try:
                    collection = geojson.FeatureCollection(features=features)
                    output_geojson = open(f'{out_path}/fltrd_{os.path.basename(self.file_path)[:-4]}.geojson', 'w')
                    output_geojson.write(geojson.dumps(collection))
                    output_geojson.close()
                except Exception as e:
                    print(f"Error while saving geojson: {e}")
        else:
            print('No georeferencing data found (lons, lats). Could not process the data.')

    def export_txt(self, out_path='.', filtered=True, mask=True):
        '''
        Export drift coordinates in image coordinate system (pixels) to text file
        '''

        output_txt = f'{out_path}/fltrd_{os.path.basename(self.file_path)[:-4]}.csv'

        if filtered == True:
            p1r, p1c, dy, dx = self.data['y0_2d'][self.data['outliers_mask'] == mask].ravel(), self.data['x0_2d'][
                self.data['outliers_mask'] == mask].ravel(), self.data['dy_2d'][
                                   self.data['outliers_mask'] == mask].ravel(), self.data['dx_2d'][
                                   self.data['outliers_mask'] == mask].ravel()
        else:
            p1r, p1c, dy, dx = self.data['y0_2d'].ravel(), self.data['x0_2d'].ravel(), self.data['dy_2d'].ravel(), \
                               self.data['dx_2d'].ravel()

        with open(output_txt, 'w', encoding='utf-8') as ff:
            for i in range(len(p1c)):
                ff.write('%s,%s,%s,%s\n' % (p1c[i], p1r[i], dx[i], dy[i]))

    def unit_vector(self, vector):
        ''' Returns the unit vector of the vector
        '''
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        ''' Returns the angle in radians between vectors 'v1' and 'v2':

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        '''
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.dot(v1_u, v2_u))
        if np.isnan(angle):
            if (v1_u == v2_u).all():
                return np.degrees(0.0)
            else:
                return np.degrees(np.pi)
        return np.degrees(angle)

    def length_between(self, v1, v2):
        ''' Returns length difference between vectors
        '''
        v1_length = np.hypot(v1[0], v1[1])
        v2_length = np.hypot(v2[0], v2[1])
        return abs(v1_length - v2_length)

    def outliers_filtering(self, radius=256, angle_difference=3, length_difference=3,
                           total_neighbours=15, angle_neighbours=15, length_neighbours=15,
                           th_small_length=3., artificial_vectors_filtering=True):
        '''
        Outliers filtering based on local homogenity criteria (vector direction and length)
        '''

        print(f'\nOutlier filtering parameters:\n'
              f'radius={radius}\n'
              f'angle_difference={angle_difference}\n'
              f'length_difference={length_difference}\n'
              f'total_neighbours={total_neighbours}\n'
              f'angle_neighbours={angle_neighbours}\n'
              f'length_neighbours={length_neighbours}\n'
              f'th_small_length={th_small_length}\n\n')

        # Filter out artificial vectors like vectors produced by CTU drift algorithm
        # (border effect + orthogonal vectors)
        if artificial_vectors_filtering==True:
            print('\nArtificial filtering...')
            # Filter start points with NaN values

            img1 = self.tiff_data['data']
            img2 = self.tiff_data2['data']

            ######################################
            # Start testing
            ######################################

            # Init masked arrays for handling
            mask = np.ma.make_mask(self.data['y1_2d'].copy())

            # End points outside an image
            mask[np.abs(self.data['x1_2d']) >= img1.shape[1]] = False
            mask[np.abs(self.data['y1_2d']) >= img1.shape[0]] = False

            # NaN values
            mask[np.isnan(self.data['y1_2d'])] = False
            mask[np.isnan(self.data['x1_2d'])] = False

            # 9999 values
            mask[np.isnan(self.data['y1_2d'])==9999] = False
            mask[np.isnan(self.data['x1_2d'])==9999] = False

            #y1_ma = np.ma.array(self.data['y1_2d'].copy(), mask=~mask).astype(int)
            #x1_ma = np.ma.array(self.data['y1_2d'].copy(), mask=~mask).astype(int)

            self.data['dy_2d'][~mask] = np.nan
            self.data['dx_2d'][~mask] = np.nan

            # Chack vicinity of end points
            w = 200

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i,j]:
                        # Image1
                        if np.isnan(img1[self.data['x1_2d'].astype('int')[i,j],
                                         self.data['y1_2d'].astype('int')[i,j]]):
                            self.data['dy_2d'][i,j] = np.nan
                            self.data['dx_2d'][i,j] = np.nan
                        # Image2
                        if np.isnan(img2[self.data['x1_2d'].astype('int')[i,j],
                                         self.data['y1_2d'].astype('int')[i,j]]):
                            self.data['dy_2d'][i,j] = np.nan
                            self.data['dx_2d'][i,j] = np.nan

                        # End points
                        # Image1
                        if np.isnan(img1[np.max((0, int(self.data['x1_2d'].astype('int')[i,j] - w))):np.min(
                                (int(self.data['x1_2d'].astype('int')[i,j] + w), img1.shape[0])),
                                    np.max((0, int(self.data['y1_2d'].astype('int')[i,j] - w))):np.min(
                                        (int(self.data['y1_2d'].astype('int')[i,j] + w), img1.shape[1]))]).any():
                            self.data['dx_2d'][i,j] = np.nan
                            self.data['dy_2d'][i,j] = np.nan

                        # Image2
                        if np.isnan(img2[np.max((0, int(self.data['x1_2d'].astype('int')[i,j] - w))):np.min(
                                (int(self.data['x1_2d'].astype('int')[i,j] + w), img2.shape[0])),
                                    np.max((0, int(self.data['y1_2d'].astype('int')[i,j] - w))):np.min(
                                        (int(self.data['y1_2d'].astype('int')[i,j] + w), img2.shape[1]))]).any():
                            self.data['dx_2d'][i,j] = np.nan
                            self.data['dy_2d'][i,j] = np.nan

                        # Start points
                        # Image1
                        if np.isnan(img1[np.max((0, int(self.data['x0_2d'].astype('int')[i, j] - w))):np.min(
                                (int(self.data['x0_2d'].astype('int')[i, j] + w), img1.shape[0])),
                                    np.max((0, int(self.data['y0_2d'].astype('int')[i, j] - w))):np.min(
                                        (int(self.data['y0_2d'].astype('int')[i, j] + w), img1.shape[1]))]).any():
                            self.data['dx_2d'][i, j] = np.nan
                            self.data['dy_2d'][i, j] = np.nan

                        # Image2
                        if np.isnan(img2[np.max((0, int(self.data['x0_2d'].astype('int')[i, j] - w))):np.min(
                                (int(self.data['x0_2d'].astype('int')[i, j] + w), img2.shape[0])),
                                    np.max((0, int(self.data['y0_2d'].astype('int')[i, j] - w))):np.min(
                                        (int(self.data['y0_2d'].astype('int')[i, j] + w), img2.shape[1]))]).any():
                            self.data['dx_2d'][i, j] = np.nan
                            self.data['dy_2d'][i, j] = np.nan


            ###########################################
            # End testing
            ###########################################

            """
            # Filter end points with NaN values
            # Set drift vector end points to 0 where is nan
            self.data['y1_2d'][np.isnan(self.data['y1_2d'])] = 0
            self.data['x1_2d'][np.isnan(self.data['x1_2d'])] = 0

            # Image1
            self.data['dy_2d'][np.isnan(img1[np.rint(self.data['x1_2d']).astype('int'),
                                             np.rint(self.data['y1_2d']).astype('int')])] = np.nan

            self.data['dx_2d'][np.isnan(img1[np.rint(self.data['x1_2d']).astype('int'),
                                             np.rint(self.data['y1_2d']).astype('int')])] = np.nan

            # Image2
            self.data['dy_2d'][np.isnan(img2[np.rint(self.data['x1_2d']).astype('int'),
                                             np.rint(self.data['y1_2d']).astype('int')])] = np.nan

            self.data['dx_2d'][np.isnan(img2[np.rint(self.data['x1_2d']).astype('int'),
                                             np.rint(self.data['y1_2d']).astype('int')])] = np.nan

            # Filter start points with NaN values
            # Image1
            self.data['dy_2d'][np.isnan(img1[np.rint(self.data['x0_2d']).astype('int'),
                                             np.rint(self.data['y0_2d']).astype('int')])] = np.nan

            self.data['dx_2d'][np.isnan(img1[np.rint(self.data['x0_2d']).astype('int'),
                                             np.rint(self.data['y0_2d']).astype('int')])] = np.nan

            # Image2
            self.data['dy_2d'][np.isnan(img2[np.rint(self.data['x0_2d']).astype('int'),
                                             np.rint(self.data['y0_2d']).astype('int')])] = np.nan

            self.data['dx_2d'][np.isnan(img2[np.rint(self.data['x0_2d']).astype('int'),
                                             np.rint(self.data['y0_2d']).astype('int')])] = np.nan
                                             
            

            # Iterate over all vectors and exclude with vicinity with NaNs
            w = 200

            # Exclude end points with NaN within vicinity (Image1)
            for i in range(self.data['x1_2d'].shape[0]):
                for j in range(self.data['x1_2d'].shape[1]):
                    if np.isnan(img1[np.max((0, int(self.data['x1_2d'][i, j] - w))):np.min((int(self.data['x1_2d'][i, j] + w), img1.shape[0])),
                                np.max((0, int(self.data['y1_2d'][i, j] - w))):np.min((int(self.data['y1_2d'][i, j] + w), img1.shape[1]))]).any():
                        self.data['dx_2d'][i, j] = np.nan
                        self.data['dy_2d'][i, j] = np.nan
            print('Done.\n')
            """
        else:
            pass


        y0, x0, dy, dx = self.data['y0_2d'].ravel(), self.data['x0_2d'].ravel(), self.data['dy_2d'].ravel(), self.data[
            'dx_2d'].ravel()

        vv = dy
        uu = dx

        idx_mask = []

        #  Radius based filtering
        vector_start_data = np.vstack((y0, x0)).T
        vector_start_tree = sklearn.neighbors.KDTree(vector_start_data)

        for i in range(len(x0)):
            # If vector lie on image border
            im1_y_min = np.max([0, y0[i] + vv[i]])
            im1_x_min = np.max([0, x0[i] + uu[i]])

            im1_y_max = np.min([y0[i] + vv[i], self.tiff_data['shape'][0] - 1])
            im1_x_max = np.min([x0[i] + uu[i], self.tiff_data['shape'][1] - 1])

            # !TODO: Fix this part for nan areas
            # First, check if vector vertices lie over NaN pixels of SAR images
            # X-axes from the algorithm output corresponds to numpy's Y-axis (rename it?)
            # if np.isnan(self.tiff_data['data'][self.data['x1_2d'].ravel()[i], self.data['y1_2d'].ravel()[i]]) or np.isnan(self.tiff_data2['data'][self.data['x1_2d'].ravel()[i], self.data['y1_2d'].ravel()[i]]):
            #    idx_mask.append(i)

            # if im1_y_min == 0 or im1_x_min == 0 or im1_y_max == (self.tiff_data['shape'][0] - 1) or im1_y_max == (self.tiff_data['shape'][0] - 1) or np.isnan(self.tiff_data['data'][self.data['y0_2d'].ravel()[i], self.data['x0_2d'].ravel()[i]]) or np.isnan(self.tiff_data['data'][self.data['y0_2d'].ravel()[i], self.data['x0_2d'].ravel()[i]]):
            #    idx_mask.append(i)
            # else:
            # Keep 'small' vectors (below threshold th_small_length)
            if np.hypot(uu[i], vv[i]) > th_small_length and not np.isnan(uu[i]):
                req_data = np.array((self.data['y0_2d'].ravel()[i], self.data['x0_2d'].ravel()[i])).reshape(1, -1)
                # Getting number of neighbours
                num_nn = vector_start_tree.query_radius(req_data, r=radius, count_only=True)
                # print('Number of neighbors: %s' % num_nn)

                if num_nn[0] < total_neighbours:
                    idx_mask.append(i)
                else:
                    nn = vector_start_tree.query_radius(req_data, r=radius)
                    data = np.vstack((uu[nn[0]], vv[nn[0]])).T

                    num_of_homo_NN = 0
                    num_of_length_homo_NN = 0

                    for ii in range(num_nn[0]):
                        # Angle between "this" vector and others
                        angle_v1_v2 = self.angle_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])
                        # Length between "this" vector and others
                        diff_v1_v2 = self.length_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])

                        if angle_v1_v2 <= angle_difference:
                            num_of_homo_NN = num_of_homo_NN + 1

                        if diff_v1_v2 < length_difference:
                            num_of_length_homo_NN = num_of_length_homo_NN + 1

                        # Mask two orthogonal vectors
                        if angle_v1_v2 >= 89 and angle_v1_v2 <= 91:
                            idx_mask.append(i)
                            idx_mask.append(ii)

                    if not (num_of_homo_NN >= angle_neighbours) or not (num_of_length_homo_NN >= length_neighbours):
                        idx_mask.append(i)
            else:
                pass

        tt = list(set(idx_mask))
        iidx_mask = np.array(tt)
        mask = np.full(uu.shape, True)
        try:
            mask[iidx_mask] = False
        except:
            pass
        self.data['outliers_mask'] = mask.reshape(self.data['y0_2d'].shape[0], self.data['y0_2d'].shape[1])

        uu[~mask] = np.nan
        vv[~mask] = np.nan

        # Experimental second iteration
        print('Second iteration...')
        for i in range(len(x0)):
            if not i in idx_mask:
                # Keep 'small' vectors (below threshold th_small_length)
                if np.hypot(uu[i], vv[i]) > th_small_length and not np.isnan(uu[i]):
                    req_data = np.array((self.data['y0_2d'].ravel()[i], self.data['x0_2d'].ravel()[i])).reshape(1, -1)
                    # Getting number of neighbours
                    num_nn = vector_start_tree.query_radius(req_data, r=radius, count_only=True)
                    # print('Number of neighbors: %s' % num_nn)

                    if num_nn[0] < total_neighbours:
                        idx_mask.append(i)

                    else:
                        nn = vector_start_tree.query_radius(req_data, r=radius)
                        data = np.vstack((uu[nn[0]], vv[nn[0]])).T

                        num_of_homo_NN = 0
                        num_of_length_homo_NN = 0

                        for ii in range(num_nn[0]):
                            # Angle between "this" vector and others
                            angle_v1_v2 = self.angle_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])
                            # Length between "this" vector and others
                            diff_v1_v2 = self.length_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])

                            if angle_v1_v2 <= angle_difference:
                                num_of_homo_NN = num_of_homo_NN + 1

                            if diff_v1_v2 < length_difference:
                                num_of_length_homo_NN = num_of_length_homo_NN + 1

                            # Mask two orthogonal vectors
                            if angle_v1_v2 >= 89 and angle_v1_v2 <= 91:
                                idx_mask.append(i)
                                idx_mask.append(ii)

                        if not (num_of_homo_NN >= angle_neighbours) or not (num_of_length_homo_NN >= length_neighbours):
                            idx_mask.append(i)
                else:
                    pass

        tt = list(set(idx_mask))
        iidx_mask = np.array(tt)
        mask = np.full(uu.shape, True)
        try:
            mask[iidx_mask] = False
        except:
            pass
        self.data['outliers_mask'] = mask.reshape(self.data['y0_2d'].shape[0], self.data['y0_2d'].shape[1])

        """
        y0, x0, dy, dx = self.data['y0_2d'].ravel(), self.data['x0_2d'].ravel(), self.data['dy_2d'].ravel(), self.data[
            'dx_2d'].ravel()

        vv = dy
        uu = dx

        idx_mask = []

        #  Radius based filtering
        vector_start_data = np.vstack((y0, x0)).T
        vector_start_tree = sklearn.neighbors.KDTree(vector_start_data)

        for i in range(0, len(x0), 1):
            # If vector lie on image border
            im1_y_min = np.max([0, y0[i] + vv[i]])
            im1_x_min = np.max([0, x0[i] + uu[i]])

            im1_y_max = np.min([y0[i] + vv[i], self.tiff_data['shape'][0] - 1])
            im1_x_max = np.min([x0[i] + uu[i], self.tiff_data['shape'][1] - 1])

            # !TODO: Fix this part for nan areas
            # First, check if vector vertices lie over NaN pixels of SAR images
            # X-axes from the algorithm output corresponds to numpy's Y-axis (rename it?)
            # if np.isnan(self.tiff_data['data'][self.data['x1_2d'].ravel()[i], self.data['y1_2d'].ravel()[i]]) or np.isnan(self.tiff_data2['data'][self.data['x1_2d'].ravel()[i], self.data['y1_2d'].ravel()[i]]):
            #    idx_mask.append(i)

            # if im1_y_min == 0 or im1_x_min == 0 or im1_y_max == (self.tiff_data['shape'][0] - 1) or im1_y_max == (self.tiff_data['shape'][0] - 1) or np.isnan(self.tiff_data['data'][self.data['y0_2d'].ravel()[i], self.data['x0_2d'].ravel()[i]]) or np.isnan(self.tiff_data['data'][self.data['y0_2d'].ravel()[i], self.data['x0_2d'].ravel()[i]]):
            #    idx_mask.append(i)
            # else:
            # Keep 'small' vectors (below threshold th_small_length)
            if np.hypot(uu[i], vv[i]) > th_small_length and not np.isnan(uu[i]):
                req_data = np.array((self.data['y0_2d'].ravel()[i], self.data['x0_2d'].ravel()[i])).reshape(1, -1)
                # Getting number of neighbours
                num_nn = vector_start_tree.query_radius(req_data, r=radius, count_only=True)
                # print('Number of neighbors: %s' % num_nn)

                if num_nn[0] < total_neighbours:
                    idx_mask.append(i)
                else:
                    nn = vector_start_tree.query_radius(req_data, r=radius)
                    data = np.vstack((uu[nn[0]], vv[nn[0]])).T

                    num_of_homo_NN = 0
                    num_of_length_homo_NN = 0

                    for ii in range(num_nn[0]):
                        # Angle between "this" vector and others
                        angle_v1_v2 = self.angle_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])
                        # Length between "this" vector and others
                        diff_v1_v2 = self.length_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])
                        if angle_v1_v2 <= angle_difference:
                            num_of_homo_NN = num_of_homo_NN + 1
                        if diff_v1_v2 < length_difference:
                            num_of_length_homo_NN = num_of_length_homo_NN + 1
                        # Mask two orthogonal vectors
                        if angle_v1_v2 >= 89 and angle_v1_v2 <= 91:
                            idx_mask.append(i)
                            idx_mask.append(ii)
                    if not (num_of_homo_NN >= angle_neighbours) or not (num_of_length_homo_NN >= length_neighbours):
                        idx_mask.append(i)
            else:
                pass

        tt = list(set(idx_mask))
        iidx_mask = np.array(tt)
        mask = np.full(uu.shape, True)
        mask[iidx_mask] = False
        self.data['outliers_mask'] = mask.reshape(self.data['y0_2d'].shape[0], self.data['y0_2d'].shape[1])
        """

    def rasterize_shp(self, field=None, att='data', par='dy_2d'):
        '''
        Rasterize shapefile into geotiff
        '''

        shp = ogr.Open(self.land_mask_path)
        lyr = shp.GetLayer()

        try:
            if hasattr(self, att):
                if par in getattr(self, att):
                    rows, cols = self.data[par].shape
                    driver = gdal.GetDriverByName('MEM')
                    dst_ds = driver.Create(
                        '',
                        cols,
                        rows,
                        1,
                        gdal.GDT_UInt16)
                    dst_ds.SetGeoTransform(self.data['geot'])
                    dst_ds.SetProjection(self.tiff_data['proj'])

                    if field is None:
                        gdal.RasterizeLayer(dst_ds, [1], lyr, None)
                    else:
                        OPTIONS = ['ATTRIBUTE=' + field]
                        gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=OPTIONS)

                    self.data['land_mask'] = dst_ds.ReadAsArray()
                    data, dst_ds, shp, lyr = None, None, None, None
                else:
                    print('\nError: No information on drift data shape is provided.')
            else:
                print('\nError: An object does not have attribute data.')
        except Exception as e:
            print(f"Error while rasterizing landmask shapefile: {e}")

