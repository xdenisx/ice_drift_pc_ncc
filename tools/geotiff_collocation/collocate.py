try:
    import gdal
except:
    from osgeo import gdal
import sys
import pathlib
import os
import numpy as np
from RasterAdjuster import RasterAdjuster
import re
from datetime import datetime, timedelta
import csv

def findRasterIntersect(raster1, raster2):
    """ 
    load data
    """
    band1 = raster1.GetRasterBand(1)
    band2 = raster2.GetRasterBand(1)
    gt1 = raster1.GetGeoTransform()
    gt2 = raster2.GetGeoTransform()

    # find each image's bounding box
    # r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * raster1.RasterXSize), gt1[3] + (gt1[5] * raster1.RasterYSize)]
    r2 = [gt2[0], gt2[3], gt2[0] + (gt2[1] * raster2.RasterXSize), gt2[3] + (gt2[5] * raster2.RasterYSize)]
    print('\t1 bounding box: %s' % str(r1))
    print('\t2 bounding box: %s' % str(r2))

    # find intersection between bounding boxes
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]
    if r1 != r2:
        print('\t** different bounding boxes **')
        # check for any overlap at all...
        if (intersection[2] < intersection[0]) or (intersection[1] < intersection[3]):
            intersection = None
            print('\t***no overlap***')
            return
        else:
            print('\tintersection:', intersection)
            left1 = int(round((intersection[0] - r1[0]) / gt1[1]))  # difference divided by pixel dimension
            top1 = int(round((intersection[1] - r1[1]) / gt1[5]))
            col1 = int(round((intersection[2] - r1[0]) / gt1[1])) - left1  # difference minus offset left
            row1 = int(round((intersection[3] - r1[1]) / gt1[5])) - top1

            left2 = int(round((intersection[0] - r2[0]) / gt2[1]))  # difference divided by pixel dimension
            top2 = int(round((intersection[1] - r2[1]) / gt2[5]))
            col2 = int(round((intersection[2] - r2[0]) / gt2[1])) - left2  # difference minus new left offset
            row2 = int(round((intersection[3] - r2[1]) / gt2[5])) - top2

            # print '\tcol1:',col1,'row1:',row1,'col2:',col2,'row2:',row2
            if col1 != col2 or row1 != row2:
                print("*** MEGA ERROR *** COLS and ROWS DO NOT MATCH ***")
            # these arrays should now have the same spatial geometry though NaNs may differ
            array1 = band1.ReadAsArray(left1, top1, col1, row1)
            array2 = band2.ReadAsArray(left2, top2, col2, row2)

    else:  # same dimensions from the get go
        col1 = raster1.RasterXSize  # = col2
        row1 = raster1.RasterYSize  # = row2
        array1 = band1.ReadAsArray()
        array2 = band2.ReadAsArray()

    return array1, array2, col1, row1, intersection

def check_save_pair(f1, f2, out_path, id_pair, intersect_ratio, maximum_drift_speed = 0):
    """
    save pair of images collocated into 'out_path' in subdirectory 'id_pair'
    """

    extension = 0
    dt1 = None
    dt2 = None

    # Get datetimes of files
    date_m = re.findall(r'\d{8}T\d{6}', f1)
    if not date_m is None:
        dt_str = '%s/%s/%sT%s:%s:%s' % (date_m[0][0:4], date_m[0][4:6], date_m[0][6:8], date_m[0][9:11], date_m[0][11:13], date_m[0][13:15])
        dt1 = datetime.strptime(dt_str, '%Y/%m/%dT%H:%M:%S')
    date_m = re.findall(r'\d{8}T\d{6}', f2)
    if not date_m is None:
        dt_str = '%s/%s/%sT%s:%s:%s' % (date_m[0][0:4], date_m[0][4:6], date_m[0][6:8], date_m[0][9:11], date_m[0][11:13], date_m[0][13:15])
        dt2 = datetime.strptime(dt_str, '%Y/%m/%dT%H:%M:%S')
    # Get extension based on time difference and maximum driftspeed
    if (dt1 is not None) and (dt2 is not None):
        extension = np.abs(float(maximum_drift_speed)) * np.abs(float( (dt1 - dt2).total_seconds() ))
    
    try:
        print('\nStart adjusment...')
        adjuster = RasterAdjuster(f1, f2, intersection_extension = extension)
        # Get arrays of rasters
        array1 = adjuster.raster1.ReadAsArray()
        array2 = adjuster.raster2.ReadAsArray()
        # Acquire the area of non-empty pixels in the rasters
        area1 = ~( np.isnan(array1) | (array1 == 0) )
        area2 = ~( np.isnan(array2) | (array2 == 0) )
        intersection_area = area1 & area2
        # Acquire the number of pixels for intersection
        intersection_area = np.sum( intersection_area )
        area1 = np.sum( area1 )
        area2 = np.sum( area2 )
        # If some of the images do not have any information
        if ( area1 == 0 ):
            print( "\nNo active pixels after cut away on file: %s" % os.path.basename(f1) )
            return 0
        if ( area2 == 0 ):
            print( "\nNo active pixels after cut away on file: %s" % os.path.basename(f2) )
            return 0
        # If intersection area is too small
        if ( intersection_area < intersect_ratio * np.min( (area1, area2) ) ):
            print( "\nToo small of an overlap for images: \n%s and \n%s" % ( os.path.basename(f1), os.path.basename(f2) ) )
            return 0
            
        print('\n### Start making pair... ###')
        # Create dir for a pir
        try:
            os.makedirs('%s/%03d' % (out_path, id_pair))
        except:
            print("Failed to make directory: %s/%03d" % (out_path, id_pair) )
            return 0
            
        adjuster.export(raster1_export_path='%s/%03d/%s' % (out_path, id_pair, os.path.basename(f1)),
                        raster2_export_path='%s/%03d/%s' % (out_path, id_pair, os.path.basename(f2)),
                        mask_export_path='%s/%03d/mask_%s' % (out_path, id_pair, os.path.basename(f1)))
        print('Adjusment done.\n')
        return 1
    except Exception as e:
        print(e)
        return 0







def acquireID( parent_folder ):
    """
    Get the next pair number ID in given folder
    """

    curID = int(0)

    path = pathlib.Path(parent_folder)
    for child in path.iterdir():
        curNameAsNum = None
        try:
            curNameAsNum = int(str(child.name))   
        except:
            continue
        if curNameAsNum >= curID:
            curID = curNameAsNum + 1

    return curID

def isPairAlreadyPresent( file1, file2, parent_folder ):
    """
    True if folder with pair of files already exists
    """

    path = pathlib.Path(parent_folder)
    ifile1 = pathlib.Path(file1)
    ifile2 = pathlib.Path(file2)
    for child in path.iterdir():
        if child.joinpath(ifile1.name).exists():
            if child.joinpath(ifile2.name).exists():
                return True

    return False



def produceIndex( parent_folder ):
    """
    Produce a .csv file indexing the pair IDs with important parameters
    """

    polarization = 'hh'
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
            files = list( child.glob( files_pref + '_' + polarization + '*.tiff'  ) )
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
            output_writer.writerow( [ str(child.name), mode1, mode2, \
            	"%.2f" % ( (end_date_and_time2 - start_date_and_time1).total_seconds() / 3600 ), \
            	start_date_and_time1.strftime("%Y-%m-%d"), start_date_and_time1.strftime("%H:%M:%S"), \
            	end_date_and_time1.strftime("%Y-%m-%d"), end_date_and_time1.strftime("%H:%M:%S"), \
            	start_date_and_time2.strftime("%Y-%m-%d"), start_date_and_time2.strftime("%H:%M:%S"), \
            	end_date_and_time2.strftime("%Y-%m-%d"), end_date_and_time2.strftime("%H:%M:%S"), \
            	str(files[0].name), str(files[1].name) ] )

	
    return



if __name__ == "__main__":
    
    # Get parameters
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    days_lag = float(sys.argv[3])
    in_path2 = None
    if len( sys.argv ) >= 5:
    	in_path2 = sys.argv[4]
    days_minimum_lag = float(0)
    if len( sys.argv ) >= 6:
    	days_minimum_lag = float(sys.argv[5])
    intersect_ratio = float(0.34)
    if len( sys.argv ) >= 7:
    	intersect_ratio = float(sys.argv[6])
    max_drift_speed = float(0.4)
    if len( sys.argv ) >= 8:
    	max_drift_speed = float(sys.argv[7])
    
    try:
        os.makedirs(out_path)
    except:
        pass
    
    polarization = 'hh'
    files_pref = 'UPS'
    
    
    # go through the first in path tree and sort out the names
    f_names = []
    for root, d_names, ff_names in os.walk(in_path):
        for fname in ff_names:
            #f_names.sort()
            f_names.append('%s/%s' % (root, fname))
    f_names = [ff for ff in f_names if (ff.endswith('tiff') and polarization in ff)]
    f_names.sort(key=lambda x: os.path.basename(x).split('_')[6], reverse=True)

    # go through the second in path tree and sort out the names    
    f_names2 = f_names
    if in_path2 is not None:
        f_names2 = []
        for root, d_names, ff_names in os.walk(in_path2):
            for fname in ff_names:
                #f_names.sort()
                f_names2.append('%s/%s' % (root, fname))
        f_names2 = [ff for ff in f_names2 if (ff.endswith('tiff') and polarization in ff)]
        f_names2.sort(key=lambda x: os.path.basename(x).split('_')[6], reverse=True)            
    
    # Go through all geotiff images first path
    for f_name in f_names:
        if os.path.basename(f_name).startswith(files_pref) and os.path.basename(f_name).endswith('tiff') \
                and f_name.find(polarization) > 0:
            ifile = f_name
            date_m = re.findall(r'\d{8}T\d{6}', f_name)
    
            if not date_m is None:
                dt0_str = '%s/%s/%sT%s:%s:%s' % (date_m[0][0:4], date_m[0][4:6], date_m[0][6:8],
                                                 date_m[0][9:11], date_m[0][11:13],date_m[0][13:15])
    
                # Date time of a current file
                dt0 = datetime.strptime(dt0_str, '%Y/%m/%dT%H:%M:%S')
    
                # Date time of a current file minus time lag
                dt0_lag = dt0 - timedelta(days=days_lag)
                dt0_lag_plus = dt0 + timedelta(days=days_lag)
                dt0_minimum_lag = dt0 - timedelta(days=days_minimum_lag)
                dt0_minimum_lag_plus = dt0 + timedelta(days=days_minimum_lag)
    
                # try to find files within i days
                for f_name2 in f_names2:
                    if f_name2 != f_name:
                        ifile2 = f_name2 #'%s/%s' % (root, f_name2)
    
                        date_m2 = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', f_name2)
    
                        dt_i_str = '%s/%s/%sT%s:%s:%s' % (date_m2[0][0:4], date_m2[0][4:6], date_m2[0][6:8],
                                                         date_m2[0][9:11], date_m2[0][11:13], date_m2[0][13:15])
                        dt_i = datetime.strptime(dt_i_str, '%Y/%m/%dT%H:%M:%S')
    
                        # If the i date within current time gap
                        if ( ( (dt_i <= dt0_minimum_lag) and (dt_i >= dt0_lag) ) or ( (dt_i >= dt0_minimum_lag_plus) and (dt_i <= dt0_lag_plus) ) ):
    
                            # See if pair is already represented
                            if not isPairAlreadyPresent( ifile, ifile2, out_path ):

                                # Get ID 
                                id_pair = acquireID(out_path) 
                                # Save pair 
                                res = check_save_pair(ifile, ifile2, out_path, id_pair, intersect_ratio, maximum_drift_speed = max_drift_speed)
    
                                if res == 1:
                                    print('\nTime lag is %.1f [hours]' % abs((dt_i - dt0).total_seconds() / 3600))
                                    print('\nMaking pair %03d ... ' % id_pair)
    
                                print('Done.\n')
                    else:
                        pass



    # Produce csv index over all pairs
    produceIndex( pathlib.Path(out_path).expanduser().absolute() )



