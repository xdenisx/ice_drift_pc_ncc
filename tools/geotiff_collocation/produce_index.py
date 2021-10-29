#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:57:19 2021

@author: andy
"""

import pathlib
import csv
import sys
import re
from datetime import datetime



def produceIndex( parent_folder ):
    """
    Produce a .csv file indexing the pair IDs with important parameters
    """

    files_pref = 'UPS'
    
    with open(str(parent_folder.joinpath("index.csv")), mode = 'w' ) as output_file:
        output_writer = csv.writer( output_file, delimiter=',', quotechar='"' )
        output_writer.writerow( [ 'ID', 'Mode of file 1', 'Mode of file 2', "Time difference [h]", \
        	'Start date of file 1', 'Start time of file 1', 'End date of file 1', 'End time of file 1', \
        	'Start date of file 2', 'Start time of file 2', 'End date of file 2', 'End time of file 2', \
        	'File1', 'File2' ] )
        
        # Loop through all subdirectories
        for child in parent_folder.iterdir():
        
            # Locate files
            files = list( child.glob( files_pref + '_*.tiff'  ) )
            if len(files) != 2:
                continue
            
            # Sort names by datetime
            files.sort(key=lambda x: str(x.name).split('_')[6], reverse=False)
            
            # Extract mode
            mode1 = str(files[0].name).split('_')[2]
            mode2 = str(files[1].name).split('_')[2]
            # Extract date and time
            start_date_and_time1 = re.match( r'(\d{8})T(\d{6})', str(files[0].name).split('_')[6] )
            start_date_and_time1 = '%s/%s/%sT%s:%s:%s' % ( start_date_and_time1.group(1)[0:4],  start_date_and_time1.group(1)[4:6], start_date_and_time1.group(1)[6:8], \
            	start_date_and_time1.group(2)[0:2], start_date_and_time1.group(2)[2:4], start_date_and_time1.group(2)[4:6], )
            start_date_and_time1 = datetime.strptime( start_date_and_time1, '%Y/%m/%dT%H:%M:%S')            
            end_date_and_time1 = re.match( r'(\d{8})T(\d{6})', str(files[0].name).split('_')[7] )
            end_date_and_time1 = '%s/%s/%sT%s:%s:%s' % ( end_date_and_time1.group(1)[0:4],  end_date_and_time1.group(1)[4:6], end_date_and_time1.group(1)[6:8], \
            	end_date_and_time1.group(2)[0:2], end_date_and_time1.group(2)[2:4], end_date_and_time1.group(2)[4:6], )
            end_date_and_time1 = datetime.strptime( end_date_and_time1, '%Y/%m/%dT%H:%M:%S')           
            start_date_and_time2 = re.match( r'(\d{8})T(\d{6})', str(files[1].name).split('_')[6] )
            start_date_and_time2 = '%s/%s/%sT%s:%s:%s' % ( start_date_and_time2.group(1)[0:4],  start_date_and_time2.group(1)[4:6], start_date_and_time2.group(1)[6:8], \
            	start_date_and_time2.group(2)[0:2], start_date_and_time2.group(2)[2:4], start_date_and_time2.group(2)[4:6], )
            start_date_and_time2 = datetime.strptime( start_date_and_time2, '%Y/%m/%dT%H:%M:%S')           
            end_date_and_time2 = re.match( r'(\d{8})T(\d{6})', str(files[1].name).split('_')[7] )
            end_date_and_time2 = '%s/%s/%sT%s:%s:%s' % ( end_date_and_time2.group(1)[0:4],  end_date_and_time2.group(1)[4:6], end_date_and_time2.group(1)[6:8], \
            	end_date_and_time2.group(2)[0:2], end_date_and_time2.group(2)[2:4], end_date_and_time2.group(2)[4:6], )
            end_date_and_time2 = datetime.strptime( end_date_and_time2, '%Y/%m/%dT%H:%M:%S')
 
            # Write a new row corresponding to the current pair parameters           
            cur_row = []           
            cur_row.append( str(child.name) )
            cur_row.append( mode1 )
            cur_row.append( mode2 )
            cur_row.append( "%.2f" % ( (end_date_and_time2 - start_date_and_time1).total_seconds() / 3600 ) )
            cur_row.append( start_date_and_time1.strftime("%Y-%m-%d") )
            cur_row.append( start_date_and_time1.strftime("%H:%M:%S") )
            cur_row.append( end_date_and_time1.strftime("%Y-%m-%d") )
            cur_row.append( end_date_and_time1.strftime("%H:%M:%S") )
            cur_row.append( start_date_and_time2.strftime("%Y-%m-%d") )
            cur_row.append( start_date_and_time2.strftime("%H:%M:%S") )
            cur_row.append( end_date_and_time2.strftime("%Y-%m-%d") )
            cur_row.append( end_date_and_time2.strftime("%H:%M:%S") )
            cur_row.append( files[0].name )
            cur_row.append( files[1].name )
            output_writer.writerow( cur_row )

	
    return




if __name__ == "__main__":
    
    # Get parameters
    out_path = sys.argv[1]

    # Produce csv index over all pairs
    produceIndex( pathlib.Path(out_path).expanduser().absolute() )