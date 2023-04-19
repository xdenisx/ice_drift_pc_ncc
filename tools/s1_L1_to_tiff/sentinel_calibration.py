# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 01:46:12 2016

@author: ekazakov

modified: Denis Demchev

"""
import xml.etree.ElementTree
import numpy as np
from scipy import interpolate, ndimage
from osgeo import gdal
import os
import numpy as np
from osgeo import osr
from osgeo import gdalconst
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from skimage.filters import median
from skimage import exposure
from scipy.ndimage import uniform_filter

GDALWARP_PATH = 'gdalwarp '
path_to_coastline = '/Home/denemc/git/ice_drift_pc_ncc/data/ne_50m_land.shp'

def save_array_as_geotiff_gcp_mode(input_array, output_path, base_raster):
    print('writing file ' + str(output_path) + '...')
    cols = base_raster.RasterXSize
    rows = base_raster.RasterYSize
    
    bands = 1
    cell_type = gdal.GDT_Float32
    driver_name = 'GTiff'
    driver = gdal.GetDriverByName(driver_name)
    #projection = base_raster.GetProjection()
    #transform = base_raster.GetGeoTransform()
    
    #transform = gdal.GCPsToGeoTransform(base_raster.GetGCPs())

    gcps = base_raster.GetGCPs()
    #gcps_count = base_raster.GetGCPCount()
    print('\ngetting GetGCPProjection...')
    gcps_projection = base_raster.GetGCPProjection()
    print('done.\n')

    out_data = driver.Create(output_path, cols, rows, bands, cell_type)
    #out_data.SetProjection(projection)

    #out_data.SetGeoTransform (transform)
    out_data.SetGCPs(gcps, gcps_projection)

    input_array[input_array < 0] = np.nan
    input_array[input_array == 0] = np.nan
    input_array = 10 * np.log10(input_array)

    out_data.GetRasterBand(1).WriteArray(input_array)
    out_data = None
    
def resave_geotiff_with_gdalwarp (input_geotiff_path, output_geotiff_path, t_srs, nodata):
    #gdalwarp -overwrite -t_srs EPSG:4326 -dstnodata 0 -of GTiff E:/dzz_mag/bolivia/LC82310722016225LGN00/LC82310722016225LGN00_B3.TIF E:/dzz_mag/bolivia/LC82310722016225LGN00/asd.tif
    cmd = GDALWARP_PATH + '-t_srs ' + t_srs + ' -dstnodata ' + str(nodata) + ' -of GTiff ' + input_geotiff_path + ' ' + output_geotiff_path
    print(cmd)
    os.system(cmd)

def fill_nan(A):
    B = A
    ok = ~np.isnan(B)
    xp = ok.ravel().nonzero()[0]
    fp = B[~np.isnan(B)]
    x  = np.isnan(B).ravel().nonzero()[0]
    B[np.isnan(B)] = np.interp(x, xp, fp)
    return B

def get_coefficients_array (xml_path, xml_element_name, xml_attribute_name, cols, rows):
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
            
                
            currentRow = fill_nan(currentRow)
            
            coefficients_rows.append(currentRow[0])
            
    print('interpolating data...')
    zoom_x = float(cols) / len(coefficients_rows[0])
    zoom_y = float(rows) / len (coefficients_rows)
    return ndimage.zoom(coefficients_rows,[zoom_y,zoom_x])
 
def perform_radiometric_calibration(input_tiff_path, calibration_xml_path, output_tiff_path):
    
    measurement_file = gdal.Open(input_tiff_path)
    measurement_file_array = np.array(measurement_file.GetRasterBand(1).ReadAsArray().astype(np.float32))
    
    radiometric_coefficients_array = get_coefficients_array(calibration_xml_path,'calibrationVectorList','sigmaNought',measurement_file.RasterXSize,measurement_file.RasterYSize)
    print('radiometric calibration...')
    calibrated_array = (measurement_file_array * measurement_file_array) / (radiometric_coefficients_array * radiometric_coefficients_array)

    save_array_as_geotiff_gcp_mode(calibrated_array, output_tiff_path, measurement_file)

def perform_noise_correction(input_tiff_path, calibration_xml_path, noise_xml_path, output_tiff_path):
    measurement_file = gdal.Open(input_tiff_path)
    measurement_file_array = np.array(measurement_file.GetRasterBand(1).ReadAsArray().astype(np.float32))
    
    radiometric_coefficients_array = get_coefficients_array(calibration_xml_path,'calibrationVectorList','sigmaNought',measurement_file.RasterXSize,measurement_file.RasterYSize)
    noise_coefficients_array = get_coefficients_array(noise_xml_path,'noiseVectorList','noiseLut',measurement_file.RasterXSize,measurement_file.RasterYSize)
    print('noise correction...')
    noise_corrected_array = (measurement_file_array * measurement_file_array - noise_coefficients_array) / (radiometric_coefficients_array * radiometric_coefficients_array)
    save_array_as_geotiff_gcp_mode(noise_corrected_array, output_tiff_path, measurement_file)
    del measurement_file

def save_projected_geotiff(input_tiff, proj4_str, grid_res, out_tiff):
    proc_str = 'gdalwarp -of GTiff -tap -tr %s %s -t_srs \'%s\' %s %s' % (grid_res, grid_res, proj4_str, input_tiff, out_tiff)
    os.system(proc_str)

def transform_gcps(gcp_list, ct):
    new_gcp_list = []
    for gcp in gcp_list:
        # point = ogr.CreateGeometryFromWkt("POINT (%s %s)" % (gcp.GCPX, gcp.GCPY))

        # Check gdal version first
        ss = gdal.__version__
        if int(ss[0]) >= 3:
            xy_target = ct.TransformPoint(gcp.GCPY, gcp.GCPX)
        else:
            xy_target = ct.TransformPoint(gcp.GCPX, gcp.GCPY)

        new_gcp_list.append(gdal.GCP(xy_target[0], xy_target[1], 0, gcp.GCPPixel, gcp.GCPLine))  # 0 stands for point elevation
    return new_gcp_list

def get_transformation(target):
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)
    ct = osr.CoordinateTransformation(source, target)
    return ct

def scale_range (input, min, max):
    idx = np.where(~np.isnan(input))
    print(idx)
    input[idx] += -(np.nanmin(input))
    input[idx] /= np.nanmax(input) / (max - min)
    input[idx] += min
    return input

def lee_filter(band, window, var_noise=0.25):
    # band: SAR data to be despeckled (already reshaped into image dimensions)
    # window: descpeckling filter window (tuple)
    # default noise variance = 0.25
    # assumes noise mean = 0

    mean_window = uniform_filter(band, window)
    mean_sqr_window = uniform_filter(band ** 2, window)
    var_window = mean_sqr_window - mean_window ** 2

    weights = var_window / (var_window + var_noise)
    band_filtered = mean_window + weights * (band - mean_window)
    return band_filtered

def get_gdal_dataset_extent(gdal_dataset):
    x_size = gdal_dataset.RasterXSize
    y_size = gdal_dataset.RasterYSize
    geotransform = gdal_dataset.GetGeoTransform()
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_max = x_min + x_size * geotransform[1]
    y_min = y_max + y_size * geotransform[5]
    return {'xMin': x_min, 'xMax': x_max, 'yMin': y_min, 'yMax': y_max, 'xRes': geotransform[1],
            'yRes': geotransform[5]}

def get_land_mask(ds_tiff):
    ''' Rasterized land mask
    based on OSM data
    '''

    source_geotransform = ds_tiff.GetGeoTransform()
    source_projection = ds_tiff.GetProjection()
    source_extent = get_gdal_dataset_extent(ds_tiff)

    # geotransform = gdal_dataset.GetGeoTransform()
    if source_geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        # gdal_dataset = gdal.Warp('',gdal_dataset, format='MEM')
        # print gdal_dataset.RasterXSize
        # geotransform = gdal_dataset.GetGeoTransform()
        # if geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        print('Error: GDAL dataset without georeferencing')

    print('Calculating land mask')
    print('Recalculate raster to WGS84')
    ds_tiff = gdal.Warp('', ds_tiff, dstSRS='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                        format='MEM')
    print('Extracting WGS84 extent')
    extent = get_gdal_dataset_extent(ds_tiff)

    # format='MEM',

    # Check if coast data exist in data directory
    if os.path.isfile(path_to_coastline):
        print('Clipping and Rasterizing land mask to raster extent')
        land_mask_wgs84 = gdal.Rasterize('', path_to_coastline,
                                         format='MEM',
                                         outputBounds=[extent['xMin'], extent['yMin'],
                                                       extent['xMax'], extent['yMax']],
                                         xRes=extent['xRes'], yRes=extent['yRes'])
        # format='MEM',
        land_mask = gdal.Warp('', land_mask_wgs84, format='MEM', dstSRS=source_projection,
                              xRes=source_extent['xRes'], yRes=source_extent['yRes'],
                              outputBounds=[source_extent['xMin'], source_extent['yMin'],
                                            source_extent['xMax'], source_extent['yMax']])

        land_data = land_mask.GetRasterBand(1).ReadAsArray()

        del land_mask
        return land_data
    else:
        print('\nCould not find land data in %s!\n' % path_to_coastline)

def reproject_ps(tif_path, out_path, t_srs, res, disk_output=False, mask=False, supress_speckle=False):
    # 1) creating CoordinateTransformation:
    target = osr.SpatialReference()
    target.ImportFromEPSG(t_srs)
    ct = get_transformation(target)

    # 2) reading GCPs from original image and transforming them:
    print('\nOpen calibrated tiff: %s\n' % tif_path)
    ds = gdal.Open(tif_path)
    dt = ds.GetGeoTransform()

    gcp_list = ds.GetGCPs()
    new_gcp_list = transform_gcps(gcp_list, ct)

    # 3) creating an image copy and writing transformed GCPs:
    driver = gdal.GetDriverByName("VRT")
    copy_ds = driver.CreateCopy("", ds)
    copy_ds.SetGCPs(new_gcp_list, target.ExportToWkt())

    # 4) warping a copy to get rid of GCP-referencing
    #    target image will have a "true" georeference:
    clb = gdal.TermProgress

    if disk_output:
        print('\nRES: %s\n' % res)
        print('Start warping...')
        ds_wrap = gdal.Warp('', copy_ds, format="MEM", dstSRS="EPSG:%s" % t_srs,
                            xRes=res, yRes=res, multithread=True, callback=clb)
        print('Warping done.')

        # Clip data and rescale
        band = ds_wrap.GetRasterBand(1)
        arr = band.ReadAsArray()
        arr[np.isinf(arr)] = np.nan

        # Speckle filtering
        if supress_speckle == True:
            print('Supressing speckle with median filter')
            #arr = lee_filter(arr, window=16)
            arr = median(arr, np.ones((9, 9)))
            print('Done')
        else:
            pass


        [rows, cols] = arr.shape

        db_min = -35
        db_max = -5
        to_max = 255
        to_min = 1

        #arr[np.isinf(arr)] = np.nan

        #arr = np.clip(arr, from_min, from_max).astype(np.float32)
        #arr[arr > -5] = -5.
        #arr[arr < -35] = -35.

        print('\n@@@@@@@@@@ %s @@@@@@@@@@@\n' % np.nanmean(arr))

        arr[arr == 0] = np.nan

        # Save mask with nan values
        if mask:
            out_path_mask = '%s/mask_%s' % (os.path.dirname(out_path), os.path.basename(out_path))
            print('\nWriting geotiff with NaN mask %s ...' % out_path_mask)
            driver = gdal.GetDriverByName('GTiff')
            outdata = driver.Create(out_path_mask, cols, rows, 1, gdal.GDT_Byte)
            outdata.SetGeoTransform(ds_wrap.GetGeoTransform())
            outdata.SetProjection(ds_wrap.GetProjection())
            arr_mask = np.copy(arr)
            arr_mask[~np.isnan(arr)] = 255
            arr_mask[np.isnan(arr)] = 0

            ##########################
            # Create land mask
            ##########################
            print('\nApplying land mask...')
            # Get land mask
            land_mask = get_land_mask(ds_wrap)

            # Apply land mask
            print(land_mask)
            arr_mask[land_mask == 255] = 0
            print('Done.\n')
            ##########################
            # End create lamd mask
            ##########################


            outdata.GetRasterBand(1).WriteArray(arr_mask)
            # outdata.GetRasterBand(1).SetNoDataValue(0)
            outdata.FlushCache()
            outdata = None

        # Rescale
        print('\nRescaling...')
        arr_out = np.clip(arr, db_min, db_max)
        print('Rescaling done.')


        print('\nWriting geotiff...')
        driver = gdal.GetDriverByName('GTiff')
        outdata = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds_wrap.GetGeoTransform())
        outdata.SetProjection(ds_wrap.GetProjection())
        outdata.GetRasterBand(1).WriteArray(arr_out)
        #outdata.GetRasterBand(1).SetNoDataValue(0)
        outdata.FlushCache()

        outdata = None
        band = None
        ds = None

        #print('Writing geotiff done.')


        #out_ds = gdal.Translate(out_path, outdata, format='GTiff',
        #                        scaleParams=[[-35, -5, 1, 255]], noData=0, bandList=[1], outputType=gdal.GDT_Byte)
        #outdata = None
    else:
        pass
        #out_ds = gdal.Warp("", copy_ds, format="MEM", dstSRS="EPSG:3995", srcNodata=0, dstNodata=0, xRes=40, yRes=40, multithread=True, callback=clb)

    # 5) clean-up and exit
    ds = None
    ds_wrap = None
    copy_ds = None

    return 1


def get_geolocationGrid( annotation_xml_file ):
    
    geolocationGrid = None
    
    try:
        # Get root of xml tree
        root = xml.etree.ElementTree.parse(annotation_xml_file).getroot()
        # Get geolocation grid
        geolocationGrid = root.find('.//geolocationGrid')
    except Exception as e:
        print( "Problem finding the geolocationGrid!" )
        print( e )
        
    return geolocationGrid

