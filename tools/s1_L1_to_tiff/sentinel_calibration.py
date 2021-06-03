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

GDALWARP_PATH = 'gdalwarp '

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
    #gcps_count = base_raster.GetGCPCount ()
    gcps_projection = base_raster.GetGCPProjection ()

    out_data = driver.Create(output_path, cols, rows, bands, cell_type)
    #out_data.SetProjection (projection)
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
 
def perform_radiometric_calibration (input_tiff_path, calibration_xml_path, output_tiff_path):
    
    measurement_file = gdal.Open(input_tiff_path)
    measurement_file_array = np.array(measurement_file.GetRasterBand(1).ReadAsArray().astype(np.float32))
    
    radiometric_coefficients_array = get_coefficients_array(calibration_xml_path,'calibrationVectorList','sigmaNought',measurement_file.RasterXSize,measurement_file.RasterYSize)
    print('radiometric calibration...')
    calibrated_array = (measurement_file_array * measurement_file_array) / (radiometric_coefficients_array * radiometric_coefficients_array)

    save_array_as_geotiff_gcp_mode(calibrated_array, output_tiff_path, measurement_file)

def perform_noise_correction (input_tiff_path, calibration_xml_path, noise_xml_path, output_tiff_path):
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

def reproject_ps(tif_path, out_path, t_srs, res, disk_output=False):
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
        ds_wrap = gdal.Warp('', copy_ds, format="MEM", dstSRS="EPSG:%s" % t_srs,
                            xRes=res, yRes=res, multithread=True, callback=clb)

        # Clip data and rescale
        band = ds_wrap.GetRasterBand(1)
        arr = band.ReadAsArray()
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

        arr[arr==0] = np.nan

        # Rescale
        print('\nRescaling...')
        #arr_out = rescale_intensity(arr, out_range=(db_min, db_max)) #.astype(np.float32)

        arr_out = np.clip(arr, db_min, db_max)
        #arr_out = rescale_intensity(arr_out, out_range=(db_min, db_max)).astype(np.float32)

        #arr_out = scale_range(arr, -35, -5)
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

        #print('Writing geotiff done.')
        #ds = None

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