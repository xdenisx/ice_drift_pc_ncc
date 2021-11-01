#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:19:39 2021

Download ERA5 data for given image coordinates

@author: hildeman
"""

import sys
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from pathlib import Path
import re
try:
    from osgeo import gdal
except:
    import gdal

from ERA5Downloader import *
sys.path.append("../geolocation_grid")
from LocationMapping import LocationMapping


    


# If run as script                                                                                                                                                                                                 
if __name__ == "__main__":
    """
    Param 1: Path to geotiff file
    Param 2: Path to folder to store data
    Param 3: variable name
    Param 4: hours prior
    Param 5: hours after
    
    """  


    
    image_files = Path( sys.argv[1] ).expanduser().absolute()
    output_folder = Path( sys.argv[2] ).expanduser().absolute()
    
    if not image_files.exists():
        raise Exception("Image file or folder did not exist!")
    if not output_folder.exists():
        raise Exception("Output fodler does not exist!")
    if not output_folder.is_dir():
        raise Exception("Output fodler is not a directory!")


    # If image_file is a directory
    if image_files.is_dir():
        # Get list of image files
        image_files = np.concatenate( ( list(image_files.glob('./*.tiff')), list(image_files.glob('./*.tif')) ) )
    else:
        image_files = [image_files]
    
    
    # Loop through all image files 
    for image_file in image_files:
    
    
        output_file = re.sub( pattern=r'(.tiff|.tif)', repl=r'.nc', string= image_file.name )
        output_file = output_folder.joinpath( output_file )
        variable_name = sys.argv[3] 
        
        hours_prior = int(1)
        if len(sys.argv) >= 5:
            hours_prior = int(sys.argv[4])
        hours_after = int(1)
        if len(sys.argv) >= 6:
            hours_after = int(sys.argv[5])
            
        if not image_file.exists():
            raise Exception("Image file did not exist!")
            
            
        # Extract time from filename
        date_m = re.findall(r'\d{8}T\d{6}', image_file.name )
        if date_m is None:
            raise Exception( "Could not extract datetime from image file name!" )
        dt_str = '%s/%s/%sT%s:%s:%s' % (date_m[0][0:4], date_m[0][4:6], date_m[0][6:8], date_m[0][9:11], date_m[0][11:13], date_m[0][13:15])
        dt = datetime.strptime(dt_str, '%Y/%m/%dT%H:%M:%S')
        # Get time 
        dt0 = dt - timedelta(hours=hours_prior)
        dt1 = dt + timedelta(hours=hours_after)
        
        # Set years
        years = [ "%04d" % elem for elem in np.arange(dt0.year, dt1.year+1) ]
        # Set months
        months = str(dt0.month)
        if len(years) > 1:
            months = [ "%d" % elem for elem in np.arange(1, 13) ]
        else:
            months = [ "%d" % elem for elem in np.arange(dt0.month, dt1.month+1) ]
        # Set days
        days = str(dt0.day)
        if len(years) > 1 or len(months) > 1:
            days = [ "%d" % elem for elem in np.arange(1, 32) ]
        else:
            days = [ "%d" % elem for elem in np.arange(dt0.day, dt1.day+1) ]
        times = [ "%02d:00" % elem for elem in np.arange(0,24) ]
        
        
        # Open geotiff file
        image = gdal.Open(str(image_file))
        # Create Location mapping
        lm = LocationMapping( image.GetGeoTransform(), image.GetProjection() )
        # Get lon and lat values of all pixels
        X = np.arange( image.RasterXSize )
        Y = np.arange( image.RasterYSize )
        X, Y = np.meshgrid( X, Y )
        lat, lon = lm.raster2LatLon( X.reshape( (-1) ), Y.reshape( (-1) ) )
        
        # Set area of domain based on extreme values
        area = [ np.max(lat) + 1, np.min(lon) - 1, np.min(lat) - 1, np.max(lon) + 1 ]
        
        # If netcdf file exists
        if not output_file.exists():
            # Download
            download_ERA5( years, months, days, times, variable_name, str( output_file ), area = area, source = 'reanalysis-era5-single-levels' )
            
        # Load ERA5 file
        dataset = xr.open_dataset( str(output_file) )
        dataset_names = list(dataset.keys())
        # Get time stamps in datetime format
        time_instances = dataset["time"].values
        time_instances = np.array( [datetime.fromtimestamp( elem ) for elem in (time_instances - np.datetime64('1970-01-01T00:00:00') )/ np.timedelta64(1, 's') ] )
        # Sort out times of interest
        allowed_times = np.array( [ (elem >= dt0) & (elem <= dt1) for elem in time_instances ] )
        dataset = dataset.isel( time = allowed_times )
        time_instances = time_instances[allowed_times]
        
        # Get dataset of time-averaged temperatures
        
        
        
        values = [None] * len( dataset_names )
        # Loop through all keys of dataset
        for iter, variable in enumerate( dataset_names ):
            
            # Get dataarray for current variable
            values[iter] = dataset[variable]
            
            # Plot spatial
            fig = plt.figure()
            plt.imshow( values[iter].mean(dim = "time").sel( latitude = xr.DataArray( lat, dims = "points" ), longitude = xr.DataArray( lon, dims = "points" ), method = "nearest" ).values.reshape( (image.RasterYSize, image.RasterXSize) )  - 273.15 )
            plt.xlabel("X-values")
            plt.ylabel("Y-values")
            plt.title( dataset_names[iter] )
            plt.colorbar()
            # Save figure
            spatial_file = re.sub( pattern=r'(.tiff|.tif)', repl=r'', string= image_file.name )
            spatial_file = 'spatial_%s_%s.png' % (dataset_names[iter], spatial_file)
            spatial_file = output_folder.joinpath( spatial_file )
            plt.savefig( spatial_file )
            plt.close(fig)
    
            
            
            fig = plt.figure()
            max_lines = values[iter].max(dim = ("longitude", "latitude") ) - 273.15
            min_lines = values[iter].min(dim = ("longitude", "latitude") ) - 273.15
            mean_lines = values[iter].mean(dim = ("longitude", "latitude") ) - 273.15
            vert_line = ( np.min( (np.min(max_lines), np.min(min_lines), np.min(mean_lines) ) ), np.max( (np.max(max_lines), np.max(min_lines), np.max(mean_lines) ) ) )
    
            plt.vlines( dt, ymin = vert_line[0], ymax = vert_line[1], colors = "green", ls="--" )        
            plt.plot( time_instances, max_lines, color = "red", linestyle='dashed' )
            plt.plot( time_instances, min_lines, color = "blue", linestyle='dashed' )
            plt.plot( time_instances, mean_lines, color = "black", linestyle='solid' )
            plt.xlabel("Time")
            plt.ylabel("Temperature [C]")
            plt.title( dataset_names[iter] )
            # Save figure
            temporal_file = re.sub( pattern=r'(.tiff|.tif)', repl=r'', string= image_file.name )
            temporal_file = 'temporal_%s_%s.png' % (dataset_names[iter], temporal_file)
            temporal_file = output_folder.joinpath( temporal_file )
            plt.savefig( temporal_file )
            plt.close(fig)
            
        
        


