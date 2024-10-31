import zipfile
import os
import glob
import xml.etree.ElementTree
import numpy as np
from osgeo import gdal, osr
import xml.etree.ElementTree as ET
import re
from pathlib import Path
import shutil
import xml.etree.ElementTree
import numpy as np
from scipy import interpolate, ndimage

__author__ = "Denis Demchev"
__copyright__ = "Copyright 2024, The Sever project"

__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Denis Demchev"
__email__ = "denis.demchev@aari.ru"
__status__ = "Production"

class dataSentinel1:
    def __init__(self, file_path=None, out_path=None):
        self.file_path = file_path
        if self.file_path is not None and self.file_path.endswith('zip'):
            print('{} will be processed.'.format(self.file_path))
            self.out_path = out_path
    
            # Unzip file
            if self.out_path is not None:
                self.unzip_dir = self.out_path
            else:
                self.unzip_dir = 'tempdir'
            os.makedirs(self.unzip_dir, exist_ok=True)
            self.__unzip_file()
    
        else:
            print('A file path is not defined.')
    
    def __unzip_file(self):
        ''' Unzip data '''
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            zip_ref.extractall(self.unzip_dir)
        
        # Get polarization modes
        self.pols = {}
        search_dir = '{}/{}/measurement'.format(self.unzip_dir, os.path.basename(self.file_path).replace('zip', 'SAFE'))
        print('search dir: %s' % search_dir)
        
        for ifile in glob.glob('{}/*.tif*'.format(search_dir)):
            print(ifile)
            ipol = re.findall(r'-[a-z][a-z]-', ifile, flags=re.IGNORECASE)[1][1:-1]
            self.pols[ipol.split('.')[0]] = {}
            self.pols[ipol.split('.')[0]]['tiff_file'] = ifile
    
    def __nan_helper(self, y):
        """Helper to handle indices and logical indices of NaNs.
    
        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """
    
        return np.isnan(y), lambda z: z.nonzero()[0]
    
    def extract_metadata(self):
        ''' Extract metadata. Currently:
            lut_sz: lookup for sigma nougt
            A: gain
            offset: offset in range direction
            pixel_first_value: first pixel value
            '''
        # Find calibration file name
        self.calib_metadata = {}
    
        for ipol in self.pols.keys():
            self.calib_metadata[ipol] = {}
            dir_name =  os.path.basename(self.file_path).replace('zip', 'SAFE')
            f_dt = os.path.basename(self.file_path).split('.')[0].split('_')[4].lower()
            #calibration-s1a-ew-grd-hh-20150407t124924-20150407t125028-005378-006d48-001.xml
            f_path = glob.glob('{}/{}/annotation/calibration/calibration-*{}*{}*.xml'.format(self.unzip_dir,
                                                                                             dir_name,
                                                                                             ipol,
                                                                                             f_dt))[0]


            xml_path = f_path
            xml_element_name = 'calibrationVectorList'
            xml_attribute_name = 'sigmaNought'
            
            with gdal.Open(self.pols[ipol]['tiff_file']) as tiff:
                rows = tiff.RasterYSize
                cols = tiff.RasterXSize
            
            coefficients_rows = []
            e = xml.etree.ElementTree.parse(xml_path).getroot()
            print('reading data...')
            for noiseVectorList in e.findall(xml_element_name):
                for child in noiseVectorList:
                    for param in child:
                        if param.tag == 'pixel':
                            currentPixels = str(param.text).split()
                        if param.tag == xml_attribute_name:
                            currentValues = str(param.text).split()
                        
                    i = 0
                    currentRow = np.empty([1,cols])
                    currentRow[:] = np.nan
                    while i < len(currentPixels):
                        currentRow[0, int(currentPixels[i])] = float(currentValues[i])
                        i += 1
                    
                    currentRow = self._fill_nan(currentRow)
                    coefficients_rows.append(currentRow[0])

            ##################################################
            # Interpolate gain values for all image pixels
            ##################################################
            print('interpolating data...')
            zoom_x = float(cols) / len(coefficients_rows[0])
            zoom_y = float(rows) / len (coefficients_rows)
            sz = ndimage.zoom(coefficients_rows,[zoom_y,zoom_x])
            self.calib_metadata[ipol]['lut_sz_2D'] = sz
            self.calib_metadata[ipol]['offset'] = 0
            dn = None
    
    def calibrate_data(self, lut_name='lut_sz_2D', clip=None):
        ''' Calibrate data '''
    
        for ipol in self.pols.keys():
            print('\nCalibrating {}'.format(self.pols[ipol]['tiff_file']))
            dn = np.float64(gdal.Open(self.pols[ipol]['tiff_file']).ReadAsArray())
            sz = ((dn ** 2 + self.calib_metadata[ipol]['offset']) / self.calib_metadata[ipol][lut_name]**2)
            sz[np.isinf(sz)] = np.nan
            sz_dB = 10. * np.log10(sz)
    
            if clip is not None:
                if ipol in clip:
                    print(f'\nClipping in {clip[ipol]}')
                    np.clip(sz_dB, clip[ipol]['db_min'], clip[ipol]['db_max'], sz_dB)
                    print('Done.')
            else:
                pass
    
            self.pols[ipol][f'{lut_name}_dB'] = sz_dB
            self.pols[ipol][f'{lut_name}_dB'][np.isinf(self.pols[ipol][f'{lut_name}_dB'])] = np.nan
    
    def export_geotiff_gcp(self, output_path=None, lut_name='lut_sz_2D'):
        ''' Export calibrated data (sz, bz, gz) to geotiff format
            (unprojected file with GCPs)'''
    
        os.makedirs(output_path, exist_ok=True)
        for ipol in self.pols.keys():
            ds = gdal.Open(self.pols[ipol]['tiff_file'])
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            bands = 1
            cell_type = gdal.GDT_Float32
            driver_name = 'GTiff'
            driver = gdal.GetDriverByName(driver_name)
            gcps = ds.GetGCPs()
            gcps_projection = ds.GetGCPProjection()
            out_fname = f'''{output_path}/{os.path.basename(self.file_path).split('.')[0]}_{ipol}.tiff'''
            out_data = driver.Create(out_fname, cols, rows, bands, cell_type)
            out_data.SetGCPs(gcps, gcps_projection)
            # print(f'''### db_min: {np.nanmin(self.pols[ipol][f'{lut_name}_dB'])}''')
            # print(f'''### db_max: {np.nanmax(self.pols[ipol][f'{lut_name}_dB'])}''')
            out_data.GetRasterBand(1).WriteArray(self.pols[ipol][f'{lut_name}_dB'])
            out_data.GetRasterBand(1).SetNoDataValue(np.nan)
            out_data = None
            self.pols[ipol]['tiff_file_dB'] = f'''{output_path}/{os.path.basename(self.file_path).split('.')[
                0]}_{ipol}.tiff'''
    
    # Adapted from https://svn.osgeo.org/gdal/trunk/autotest/alg/warp.py
    def warp_with_gcps(self,
                       input_path=None,
                       output_path=None,
                       gcp_epsg=4326,
                       output_epsg=3875,
                       pixel_size=None,
                       resampling=gdal.GRA_NearestNeighbour):
        ''' Warp geotiff file with GCPs '''
    
        # Open the source dataset and add GCPs to it
        print(f'''### Open input file: {input_path}''')
        src_ds = gdal.OpenShared(input_path, gdal.GA_ReadOnly)
        gcps = src_ds.GetGCPs()
        gcp_srs = osr.SpatialReference()
        gcp_srs.ImportFromEPSG(gcp_epsg)
        gcp_crs_wkt = gcp_srs.ExportToWkt()
        src_ds.SetGCPs(gcps, gcp_crs_wkt)
    
        # Define target SRS
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(output_epsg)
        dst_wkt = dst_srs.ExportToWkt()
    
        error_threshold = 0.125  # error threshold --> use same value as in gdalwarp
    
        # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
        tmp_ds = gdal.AutoCreateWarpedVRT(src_ds,
                                          None,  # src_wkt : left to default value --> will use the one from source
                                          dst_wkt,
                                          resampling,
                                          error_threshold)
        dst_xsize = tmp_ds.RasterXSize
        dst_ysize = tmp_ds.RasterYSize
        dst_gt = tmp_ds.GetGeoTransform()
        tmp_ds = None
    
        # Now create the true target dataset
        dst_path = str(Path(output_path).with_suffix(".tiff"))
        cell_type = gdal.GDT_Float32
        dst_ds = gdal.GetDriverByName('GTiff').Create(dst_path, dst_xsize, dst_ysize, src_ds.RasterCount, cell_type)
        dst_ds.SetProjection(dst_wkt)
        dst_ds.SetGeoTransform(dst_gt)
        dst_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    
        # And run the reprojection
        clb = gdal.TermProgress
        gdal.ReprojectImage(src_ds,
                            dst_ds,
                            None,  # src_wkt : left to default value --> will use the one from source
                            None,  # dst_wkt : left to default value --> will use the one from destination
                            resampling,
                            0,  # WarpMemoryLimit : left to default value
                            error_threshold,
                            clb,  # Progress callback : could be left to None or unspecified for silent progress
                            None)  # Progress callback user data
    
        # If pixel size is not None apply downsampling
        if pixel_size is not None:
            print(f'Downsampling to {pixel_size} m')
            downsampled_dst_path = '{}/UPS_{}'.format(os.path.dirname(dst_path), os.path.basename(dst_path))
            ds_tiff = gdal.Warp(downsampled_dst_path, dst_ds, format='Gtiff', xRes=pixel_size, yRes=pixel_size)
            ds_tiff = None
            os.remove(dst_path)
            print('Done.')
        else:
            pass
    
        dst_ds = None
    
    def export_projected_geotiff(self, output_path=None, lut_name='lut_sz_2D', gcp_epsg=4326, epsg=3875,
                                 pixel_size=None,
                                 resampling=gdal.GRA_NearestNeighbour,
                                 format_filename=False,
                                 pref=None,
                                 normalize=False):
        ''' Export geotiff projected '''
    
        os.makedirs(output_path, exist_ok=True)
        # Export calibrated data to geotiff file
        self.export_geotiff_gcp(output_path=output_path, lut_name=lut_name)
    
        for ipol in self.pols.keys():
            # Warp geotiff file with GCPs to desired projection
            if not format_filename:
                out_fname = f'''{output_path}/{epsg}_{os.path.basename(self.file_path).split('.')[0]}_{ipol}.tiff'''
            else:
                # Formatted filename for the drift pipeline
                # UPS_hh_ALOS2_XX_XXXX_XXXX_20190629T143413_20190629T143423_0000324300_001001_ALOS2275382000-190629.tiff
                # Get instrument
                instriment = re.findall(r'RCM\d', os.path.basename(self.file_path))[0]
                # Get date and time
                dt_full = re.findall(r'\d\d\d\d\d\d\d\d_\d\d\d\d\d\d', os.path.basename(self.file_path))[0]
                idate, itime = dt_full.split('_')
                out_fname = f'''{output_path}/{ipol.lower()}_{instriment}_XX_XXXX_XXXX_{idate}T{itime}.tiff'''
                print(f'Generating: {out_fname}')
    
            self.warp_with_gcps(input_path=self.pols[ipol]['tiff_file_dB'],
                                output_path=out_fname,
                                gcp_epsg=gcp_epsg,
                                output_epsg=epsg,
                                pixel_size=pixel_size,
                                resampling=resampling)
    
            # We don't need a geotiff with GCPs anymore
            # print(f'''\nRemoving {self.pols[ipol]['tiff_file_dB']}''')
            os.remove(self.pols[ipol]['tiff_file_dB'])
            self.pols[ipol]['tiff_file_dB'] = None
            print('Done.')
    
    def delete_temp(self):
        '''
        Delete unziped folders
        '''
    
        for root, dirs, files in os.walk(self.unzip_dir):
            for idir in dirs:
                if idir.find(os.path.basename(self.file_path).split('.')[0])>=0:
                    print(f'''### {idir}''')
                    shutil.rmtree(f'''{root}/{idir}''')
                else:
                    pass
    
            for ifile in files:
                if ifile.find(os.path.basename(self.file_path).split('.')[0])>=0 and ifile.endswith('xml'):
                    os.remove(f'''{root}/{ifile}''')
                else:
                    pass

    def _fill_nan(self, A):
        B = A
        ok = ~np.isnan(B)
        xp = ok.ravel().nonzero()[0]
        fp = B[~np.isnan(B)]
        x  = np.isnan(B).ravel().nonzero()[0]
        B[np.isnan(B)] = np.interp(x, xp, fp)
        return B