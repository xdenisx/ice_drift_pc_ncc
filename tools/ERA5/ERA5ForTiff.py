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
    Param 2: Path to store data
    Param 3: variable name
    Param 4: hours prior
    Param 5: hours after
    
    """  

    image_file = Path( sys.argv[1] ).expanduser().absolute()
    output_file = Path( sys.argv[2] ).expanduser().absolute()
    variable_name = sys.argv[3] 
    
    hours_prior = int(1)
    if len(sys.argv) > 5:
        hours_prior = int(sys.argv[4])
    hours_after = int(1)
    if len(sys.argv) > 6:
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
    
    if not output_file.exists():
        # Download
        download_ERA5( years, months, days, times, variable_name, str( output_file ), area = area, source = 'reanalysis-era5-single-levels' )
        
    # Load ERA5 file
    dataset = xr.open_dataset( str(output_file) )
    dataset_names = list(dataset.keys())
    # Get time stamps in datetime format
    times = dataset["time"].values
    times = np.array( [datetime.fromtimestamp( elem ) for elem in (times - np.datetime64('1970-01-01T00:00:00') )/ np.timedelta64(1, 's') ] )
    # Sort out times of interest
    allowed_times = np.array( [ (elem >= dt0) & (elem <= dt1) for elem in times ] )
    dataset = dataset.isel( time = allowed_times )
    times = times[allowed_times]
    
    
    
    values = [None] * len( dataset_names )
    # Loop through all keys of dataset
    plt.figure(1)
    for iter, variable in enumerate( dataset_names ):
        plt.subplot(2, len( dataset_names ), iter+1)
        # Interpolate to raster points
        values[iter] = dataset.sel( latitude = xr.DataArray( lat, dims = "points" ), longitude = xr.DataArray( lon, dims = "points" ), method = "nearest" )[ variable ]
        
        # Plot spatial
        plt.imshow( values[iter].isel( time = 0 ).values.reshape( (image.RasterXSize, image.RasterYSize) ) )
        plt.xlabel("X-values")
        plt.ylabel("Y-values")
        plt.title( dataset_names[iter] )
        plt.colorbar()
        
        # Plot temporal
        plt.subplot(2, len( dataset_names ), len( dataset_names ) + iter+1)
        plt.plot( times, np.mean(values[iter].values, axis=1) )
        plt.xlabel("Time")
        plt.ylabel("Mean temperature [K]")
        plt.title( dataset_names[iter] )
        
    plt.show()
        
        


