#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 18:22:39 2021

@author: andy
"""

try:
    from osgeo import osr
except: 
    import osr

import numpy as np



class LocationMapping:

    
    def __init__( self, geotransform, projection ):
        
        xoffset, px_w, rot1, yoffset, px_h, rot2 = geotransform
        
        self.trans_matrix = np.array( [ [px_w, rot1], [rot2, px_h] ] )
        self.trans_offset = np.array( [xoffset + px_w / 2.0, yoffset + px_h / 2.0] ).reshape( (2, 1) )
        
        # get CRS from dataset 
        crs = osr.SpatialReference()
        crs.ImportFromWkt( projection )
        # create lat/long crs with WGS84 datum
        crsGeo = osr.SpatialReference()
        crsGeo.ImportFromEPSG(4326) # 4326 is the EPSG id of lat/long crs 
        self.t = osr.CoordinateTransformation(crs, crsGeo)
        self.t_inv = osr.CoordinateTransformation(crsGeo, crs)
        
    

    def raster2LatLon( self, x, y ):
        
        points = np.stack( x,y, axis = 1 )
        
        pos = np.matmul( self.trans_matrix, points.T )
        pos = pos + self.trans_offset
        
        (lat, long, z) = self.t.TransformPoints( pos.T )
        
        return (lat, long)
    
    
    def latLon2Raster( self, lat, long ):
        
        points = np.stack( lat,long, axis = 1 )
        
        (x, y, z) = self.t_inv.TransformPoints( points )
        pos = np.stack( x, y, axis=1 )
        pos = pos - self.trans_offset
        pos = np.linalg.solve( self.trans_matrix, pos.T ).T
        
        x = pos[:,0]
        y = pos[:,1]
        
        return (x, y)