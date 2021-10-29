#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download ERA5 data



Created on Mon Nov 25 14:46:15 2019

@author: hildeman
"""


# Imports
import numpy as np
import cdsapi
import sys




def download_ERA5( year, month, day, time, variable_names, output_file, area = None, source = 'reanalysis-era5-single-levels' ):
    """
        Download ERA5 data of choice to file
    """
    
    # Create cdsapi client object
    c = cdsapi.Client()

    print( "Variable names: %s" % variable_names )
    print( "Years: %s" % str(year) )
    print( "Months: %s" % str(month))
    print( "Days: %s" % str(day) )
    print( "Times: %s" % str(time) )

    request_dict = {
            'product_type':'reanalysis',
            'format':'netcdf',
            'variable': variable_names,
            'month': month,
            'year':year,
            'day': day,
            'time': time
        }
    if area is not None:
        print( "Area: %s" % str(area) )
        request_dict["area"] = area
    
    # Create request for file and wait
    c.retrieve( source, request_dict, output_file )
    


# If run as script                                                                                                                                                                                                 
if __name__ == "__main__":
    """
    Param 1: Path to store data
    Param 2: year
    Param 3: month
    Param 4: day
    Param 5: time
    Param 6: variable name
    """  

    output_file = sys.argv[1] 
    year = sys.argv[2] 
    month = sys.argv[3] 
    day = sys.argv[4] 
    time = sys.argv[5] 
    variable_name = sys.argv[6] 

    download_ERA5( year, month, day, time, variable_name, output_file )

        
    
    
    
