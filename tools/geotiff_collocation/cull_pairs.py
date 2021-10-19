#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:10:10 2021

@author: andy
"""

import numpy as np
import tiff_areas
import re
import sys
import datetime
from pathlib import Path
import shutil


def get_time_window_index( datetime2, datetime1, time_delta ):
    return int(np.floor( float(( datetime2 - datetime1 ).total_seconds() ) / float(time_delta.total_seconds() ) ))


def get_subdirectories( path_pairs, resolution ):
    # Go through all elemetns in directory
    subdirectories = []
    for subdirectory in path_pairs.glob('*'):
        if re.match( r'^\d{3}$', subdirectory.name ) is not None:
            cur_dir = { 'directory' : subdirectory }
            
            # Get image files
            cur_dir["image_files"] = list( subdirectory.glob( 'UPS*.tif' ) ) + list( subdirectory.glob( 'UPS*.tiff' ) )
            if len(cur_dir["image_files"]) != 2:
                continue
            
            cur_dir["areas"] = tiff_areas.get_area_of_images( str(cur_dir["image_files"][0]), resolution, str(cur_dir["image_files"][1]) )
        
            # Get datetimes of files
            cur_dir["datetime"] = [None] * 2
            for image_file in cur_dir["image_files"]:
                date_m = re.findall(r'\d{8}T\d{6}', image_file.name )
                dt_str = '%s/%s/%sT%s:%s:%s' % (date_m[0][0:4], date_m[0][4:6], date_m[0][6:8], date_m[0][9:11], date_m[0][11:13], date_m[0][13:15])
                dt = datetime.datetime.strptime(dt_str, '%Y/%m/%dT%H:%M:%S')
                if cur_dir["datetime"][0] is None:
                    cur_dir["datetime"][0] = dt
                    cur_dir["datetime"][1] = dt
                else:
                    if cur_dir["datetime"][0] > dt:
                        cur_dir["datetime"][0] = dt
                    if cur_dir["datetime"][1] < dt:
                        cur_dir["datetime"][1] = dt

            # Append current directory to list of subdirectories            
            subdirectories.append( cur_dir )
            
    # Go through all subdirectories and sort them based on their highest datetime
    subdirectories.sort( key = lambda x: x["datetime"][1] )
    
    return subdirectories
    
    
def get_time_windows( subdirectories, time_delta ):    
    # Get time windows
    num_windows = 0
    if len( subdirectories ) > 0:
        num_windows = get_time_window_index( subdirectories[-1]["datetime"][1], subdirectories[0]["datetime"][1], time_delta ) + 1
    windows = [ [] for iter in range( num_windows ) ]
    for iter, subdirectory in enumerate( subdirectories ):
        # Get current pairs time window
        index = get_time_window_index( subdirectory["datetime"][1], subdirectories[0]["datetime"][1], time_delta )
        windows[ index ].append(iter)    
        
    # Sort inside each window based on overlapping areas
    for window in windows:
        window.sort( reverse = True, key = lambda x : subdirectories[x]["areas"][2] )

    return windows



# If run as script                                                                                                                                                                                                 
if __name__ == "__main__":
    """
    Param 1: Path to pairs folder
    Param 2: Resolution [m/pixel]
    Param 3: Size of temporal windows [h]
    Param 4: Should the pairs with smaller areas be removed? [1 / 0]
    """  

    if len(sys.argv) < 2:
        raise Exception("No pairs path was given!")
    	
    # Path to first image                                                                                                                                                                                        
    path_pairs = Path(sys.argv[1])
    
    if not path_pairs.is_dir(  ):
        raise Exception("Given pairs parent path does not exist!")
        
    # Get resolution
    resolution = int(100)
    if len(sys.argv) >= 3:
        resolution = int( sys.argv[2] )
        
    # Get size of temporal window
    time_delta = datetime.timedelta( hours = 24 )
    if len(sys.argv) >= 4:
        time_delta = datetime.timedelta( hours = float(sys.argv[3]) )
        
    # Get size of temporal window
    remove = False
    if len(sys.argv) >= 5:
        remove = bool(int(sys.argv[4]))
        
    # Get the subdirectories and the data
    subdirectories = get_subdirectories( path_pairs, resolution )
    
    # Print areas
    for directory in subdirectories:
        print( "Directory %s has areas %.2f, %.2f, %.2f [km^2]" % \
              ( directory["directory"].name, directory["areas"][0] * 1e-6, directory["areas"][1] * 1e-6, directory["areas"][2] * 1e-6 ) \
              )
    
    # Get the time windows
    windows = get_time_windows( subdirectories, time_delta )
    
    for iter, window in enumerate( windows ):
        
        if len(window) == 0:
            continue
        
        print( "Current time window occupies the time slot %s - %s and involves the pairs:" % ( \
                subdirectories[0]["datetime"][1] + time_delta * iter, \
                subdirectories[0]["datetime"][1] + time_delta * (iter + 1) ) )
        print( [ subdirectories[iter2]["directory"].name for iter2 in window ] )
        # If should remove unecessary pairs
        if remove:
            for dir_path in [ subdirectories[iter]["directory"] for iter in window[1:] ]:
                try:
                    cur_path = dir_path.name
                    shutil.rmtree(dir_path)
                    print("Removed %s" % cur_path)
                except Exception as e:
                    print(e)
                    

    
    
        
        
        
        
    
    
    
        
    
        
        
    