#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script interpoaltes and plots geolocationGridPoint variables onto the raster of the given geotiff image.

2021-10-26 Anders Gunnar Felix Hildeman

"""

import sys
import pathlib
from matplotlib import pyplot as plt
import numpy as np
from LocationMapping import LocationMapping
import xml.etree.ElementTree
from scipy import interpolate
try:
	from osgeo import gdal
except:
	import gdal



def plotInterpolatedVariable( image_path, band_name, variable_name ):
    
    image = gdal.Open( image_path )
    
    array = None
    Z = None
    
    # Get mapping between coordinates for the given image
    location_mapping = LocationMapping( image.GetGeoTransform(), image.GetProjection() )
    
    if band_name is None:
            tags = image.GetMetadata().keys()
            print( "No band name given! Possible band names are: " )
            print( list(tags) )
            return (None, None)
    
    if band_name in image.GetMetadata():
        
        xml_tree = xml.etree.ElementTree.fromstring( image.GetMetadata()[band_name] )
        
        if variable_name is None:
            tags = set()
            for elem in xml_tree.findall( './geolocationGridPointList/geolocationGridPoint/*' ):
                tags.add( elem.tag )
            print( "No variable given! Possible variables are: " )
            print( list(tags) )
            return (None, None)
        
        # Get all grid points xml elements
        gridPoints = list( xml_tree.findall( './geolocationGridPointList/geolocationGridPoint' ) )
        
        # Go through all girdPoint elements
        lat = np.zeros( (len(gridPoints)) )
        long = np.zeros( (len(gridPoints)) )
        val = np.zeros( (len(gridPoints)) )
        ok_grid_points = np.zeros( (len(gridPoints)), dtype = bool )
        for iterGridPoints, gridPoint in enumerate(gridPoints):
            # Insert current latitude and longitude value
            cur_lat = gridPoint.find('./latitude')
            cur_long = gridPoint.find('./longitude')
            cur_val = gridPoint.find('./%s' % variable_name )
            # If the latitude and longitude value actually existed
            if cur_lat is not None and cur_long is not None and cur_val is not None:
                # Insert current values in arrays
                lat[iterGridPoints] = float(cur_lat.text)
                long[iterGridPoints] = float(cur_long.text)
                val[iterGridPoints] = float(cur_val.text)
                # print( float(cur_val.text) )
                ok_grid_points[iterGridPoints] = True
                
        print( "Number of active grid points: %s" % str(np.sum(ok_grid_points)) )

        # Map geolocation points
        x,y = location_mapping.latLon2Raster( np.array(lat), np.array(long) )
        
        # Create interpolator
        f = interpolate.NearestNDInterpolator( list(zip(x[ok_grid_points], y[ok_grid_points])), val[ok_grid_points] )
        # f = interpolate.interp2d( x[ok_grid_points], y[ok_grid_points], val[ok_grid_points], kind='linear', fill_value = np.nan )
        # Interpoalte to grid
        X = np.arange( image.RasterXSize )
        Y = np.arange( image.RasterYSize )
        X,Y = np.meshgrid( X,Y )
        Z = f( X, Y )
        # Z = f( np.arange( image.RasterXSize ) , np.arange( image.RasterYSize ) )
        
        
    
    # Get band with band name
    for iterBands in range( image.RasterCount ):
        # Get current raster band
        band = image.GetRasterBand(int(iterBands+1))
        # Get name of raster band
        if band_name == band.GetDescription():
            array = band.ReadAsArray()
        
    return (Z, array)




# If run as script
if __name__ == "__main__":
	"""
	Param 1: Path to geotiff image.
	Param 2: Name of the band which data shuld be interpolate and plotted. If this parameter is not given, the script prints the set of all possible variable names.
	Param 3: Name of the geolocationGridPoint variable that should be interpolated and plotted. If this parameter is not given, the script prints the set of all possible variable names.
	"""
    
	# image path
	image_path = pathlib.Path( sys.argv[1] ).expanduser().absolute()
	# band name
	band_name = None
	if len(sys.argv) >= 3:
	    band_name = sys.argv[2]
	# variable name    
	variable_name = None
	if len(sys.argv) >= 4:
	    variable_name = sys.argv[3]

	Z, array = plotInterpolatedVariable( str(image_path), band_name, variable_name )

	if Z is not None or array is not None:    
		plt.figure(1)
        
		if Z is not None:
			plt.subplot(1,2,1)
			plt.imshow( Z )
			plt.xlabel("Raster X-dim")
			plt.ylabel("Raster Y-dim")
			plt.title(variable_name)
			plt.colorbar()
            
		if array is not None:
			plt.subplot(1,2,2)
			plt.imshow( array )
			plt.xlabel("Raster X-dim")
			plt.ylabel("Raster Y-dim")
			plt.title(band_name)
			plt.colorbar()
    		
		plt.show()

