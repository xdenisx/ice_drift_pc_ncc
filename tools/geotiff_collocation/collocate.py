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

def check_save_pair(f1, f2, out_path, id_pair, polarizations, intersect_ratio, maximum_drift_speed = 0):
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

        adjuster1 = RasterAdjuster(f1, f1)
        adjuster2 = RasterAdjuster(f2, f2)
        bandNames = []
        bands1 = []
        bands2 = []
        metadata1 = []
        metadata2 = []
        
        

        # Loop through all polarizations
        try:

            for polarization in polarizations:
                # print("Going through polarization: %s" % polarization)
                # See if current polarization exists
                f1_polarization = re.sub(r"UPS_\w{2}_", "UPS_" + polarization + "_", f1  )
                f2_polarization = re.sub(r"UPS_\w{2}_", "UPS_" + polarization + "_", f2  )
                if not (os.path.isfile(f1_polarization) and os.path.isfile(f2_polarization)):
                    # If it does'nt exist in both
                    continue
                
                bandNames.append( polarization )
                # Acquire cropped band from current polarization to list of bands for file 1
                tempRaster = gdal.Open( f1_polarization )
                adjuster1.initFromRasters( adjuster1.raster1, tempRaster )
                # Get metadata
                metadata = tempRaster.GetMetadata()
                if "geolocationGrid" in metadata.keys():
                    metadata1.append( metadata["geolocationGrid"] )
                else:
                    metadata1.append("")
                # Acquire cropped band from current polarization to list of bands for file 2
                tempRaster = gdal.Open( f2_polarization )
                adjuster2.initFromRasters( adjuster2.raster1, tempRaster )
                # Get metadata
                metadata = tempRaster.GetMetadata()
                if "geolocationGrid" in metadata.keys():
                    metadata2.append( metadata["geolocationGrid"] )
                else:
                    metadata2.append("")

                # Acquire cropped band from current polarization to list of bands for file 2
                bands1.append( adjuster1.raster2.GetRasterBand(1).ReadAsArray() )
                bands2.append( adjuster2.raster2.GetRasterBand(1).ReadAsArray() )
                
                
                
                
        except Exception as e:
            print("Failed to crop all bands to same raster!")
            raise e
            return 0
        if len( bands1 ) == 0:
            return 0
        print("Bands: %s" % str(bandNames))
        
        
        polar_name = bandNames[0]
        # If more than one polarization 
        if len( bandNames ) > 1:
            polar_name = 'XX'
        # Rename file name after polarization
        f1 = re.sub(r"UPS_\w{2}_", "UPS_" + polar_name + "_", f1  )
        f2 = re.sub(r"UPS_\w{2}_", "UPS_" + polar_name + "_", f2  )
        
        raster1_export_path='%s/%03d/%s' % (out_path, id_pair, os.path.basename(f1))
        raster2_export_path='%s/%03d/%s' % (out_path, id_pair, os.path.basename(f2))
        mask_export_path='%s/%03d' % (out_path, id_pair)
        # If file already exists
        if os.path.isfile( raster1_export_path ):
            return 0

        try:
            # Populate all bands
            tempRaster = gdal.GetDriverByName('GTiff').Create( "%s/temp1.tiff" % out_path , adjuster1.raster1.RasterXSize, adjuster1.raster1.RasterYSize, len(bands1), gdal.GDT_Float32 )
            tempRaster.SetGCPs( adjuster1.raster1.GetGCPs(), adjuster1.raster1.GetGCPProjection() )
            tempRaster.SetGeoTransform( adjuster1.raster1.GetGeoTransform() )
            tempRaster.SetProjection( adjuster1.raster1.GetProjectionRef() )
            for iter in range(len(bands1)):
                tempRaster.GetRasterBand(int(iter+1)).WriteArray(bands1[iter])
                tempRaster.GetRasterBand(int(iter+1)).SetDescription(bandNames[iter])
            tempRaster.SetMetadata( dict( zip(["Band " + str(iter) for iter in range(len(bandNames))], bandNames ) ) )
            adjuster1.initFromRasters( tempRaster, tempRaster )

            tempRaster2 = gdal.GetDriverByName('GTiff').Create( "%s/temp2.tiff" % out_path , adjuster2.raster1.RasterXSize, adjuster2.raster1.RasterYSize, len(bands2), gdal.GDT_Float32 )
            tempRaster2.SetGCPs( adjuster2.raster1.GetGCPs(), adjuster2.raster1.GetGCPProjection() )
            tempRaster2.SetGeoTransform( adjuster2.raster1.GetGeoTransform() )
            tempRaster2.SetProjection( adjuster2.raster1.GetProjectionRef() )
            for iter in range(len(bands2)):
                tempRaster2.GetRasterBand(int(iter+1)).WriteArray(bands2[iter])
                tempRaster2.GetRasterBand(int(iter+1)).SetDescription(bandNames[iter])
            tempRaster.SetMetadata( dict( zip(["Band " + str(iter) for iter in range(len(bandNames))], bandNames ) ) )
            adjuster2.initFromRasters( tempRaster2, tempRaster2 )
        except Exception as e:
            print("Failed to add all bands to rasters!")
            raise e
            return 0


        print('\nStart adjusment...')

        adjuster1.initFromRasters( adjuster1.raster1, adjuster2.raster1, intersection_extension = extension)
        # Get arrays of rasters
        array1 = adjuster1.get_raster1_as_array()
        array2 = adjuster1.get_raster2_as_array()
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
            print( "\nNo active pixels after cut away on file: %s" % os.path.basename(f1_polarization) )
            return 0
        if ( area2 == 0 ):
            print( "\nNo active pixels after cut away on file: %s" % os.path.basename(f2_polarization) )
            return 0
        # If intersection area is too small
        if ( intersection_area < intersect_ratio * np.min( (area1, area2) ) ):
            print( "\nToo small of an overlap for images: \n%s and \n%s" % ( os.path.basename(f1_polarization), os.path.basename(f2_polarization) ) )
            return 0
            
        print('\n### Start making pair... ###')
        # Create dir for a pir
        try:
            os.makedirs('%s/%03d' % (out_path, id_pair))
        except:
            print("Failed to make directory: %s/%03d" % (out_path, id_pair) )
            return 0
            
        adjuster1.export(raster1_export_path=raster1_export_path,
                         mask1_fname=f'''mask_{os.path.basename(f1)}''',
                         raster2_export_path=raster2_export_path,
                         mask2_fname=f'''mask_{os.path.basename(f2)}''',
                         mask_export_path=mask_export_path,
                         normalize=normalize)
        
        # Insert metadata
        metadata = {}
        for iter, bandName in enumerate( bandNames ):
            metadata[bandName] = metadata1[iter]
        raster1 = gdal.Open( raster1_export_path )
        raster1.SetMetadata( metadata )
        raster1.FlushCache()
        metadata = {}
        for iter, bandName in enumerate( bandNames ):
            metadata[bandName] = metadata2[iter]
        raster2 = gdal.Open( raster2_export_path )
        raster2.SetMetadata( metadata )
        raster2.FlushCache()
        
        
        
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
    regexp1 = re.match( r'(^UPS_)(\w{2})(_.*$)', ifile1.name )
    if regexp1 is None:
        return False
    
    ifile2 = pathlib.Path(file2)
    regexp2 = re.match( r'(^UPS_)(\w{2})(_.*$)', ifile2.name )
    if regexp2 is None:
        return False
    
    for child in path.iterdir():
        
        file1_exists = False
        file2_exists = False
        
        list_files = child.glob('*')
        for cur_file in list_files:
            regexpCur = re.match( r'(^UPS_)(\w{2})(_.*$)', cur_file.name )
            if regexpCur is not None:
                try:
                    if regexpCur.group(3) == regexp1.group(3):
                        file1_exists = True
                    if regexpCur.group(3) == regexp2.group(3):
                        file2_exists = True
                except: 
                    pass
            if file1_exists and file2_exists:
                return True

    return False






if __name__ == "__main__":
    print('starting')
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
    normalize = False
    if len( sys.argv ) >= 8:
        normalize = bool(sys.argv[8])
        print('Data will be normalized to 0 255')
    
    try:
        os.makedirs(out_path)
    except:
        pass
    
    polarizations = ['HH', 'VV', 'HV', 'VH', 'N', 'hh', 'hv', 'vh', 'vv']
    #polarizations = [ x.lower() for x in polarizations ]
    files_pref = 'UPS'

    # go through the first in path tree and sort out the names
    f_names = []
    for root, d_names, ff_names in os.walk(in_path):
        for fname in ff_names:
            #f_names.sort()
            f_names.append('%s/%s' % (root, fname))
    f_names = [ff for ff in f_names if (ff.endswith('tiff') and any( polarization in ff for polarization in polarizations)) ]
    f_names.sort(key=lambda x: os.path.basename(x).split('_')[6], reverse=True)
    print(f'f_names: {f_names}')

    # go through the second in path tree and sort out the names    
    f_names2 = f_names
    if in_path2 is not None:
        f_names2 = []
        for root, d_names, ff_names in os.walk(in_path2):
            for fname in ff_names:
                #f_names.sort()
                f_names2.append('%s/%s' % (root, fname))
        f_names2 = [ff for ff in f_names2 if (ff.endswith('tiff') and any( polarization in ff for polarization in polarizations) ) ]
        f_names2.sort(key=lambda x: os.path.basename(x).split('_')[6], reverse=True)
    
    # Go through all geotiff images first path
    for f_name in f_names:
        if os.path.basename(f_name).startswith(files_pref) and os.path.basename(f_name).endswith('tiff'):
            if not any( polarization in f_name for polarization in polarizations ):
                continue

            ifile = f_name
            date_m = re.findall(r'\d{8}T\d{6}', f_name)
            print(date_m)
            print('####')
            
            if not date_m is None:
                dt0_str = '%s/%s/%sT%s:%s:%s' % (date_m[0][0:4], date_m[0][4:6], date_m[0][6:8],
                                                 date_m[0][9:11], date_m[0][11:13],date_m[0][13:15])
    
                # Date time of a current file
                dt0 = datetime.strptime(dt0_str, '%Y/%m/%dT%H:%M:%S')
                print(dt0)
    
                # Date time of a current file minus time lag
                dt0_lag = dt0 - timedelta(days=days_lag)
                dt0_lag_plus = dt0 + timedelta(days=days_lag)
                dt0_minimum_lag = dt0 - timedelta(days=days_minimum_lag)
                dt0_minimum_lag_plus = dt0 + timedelta(days=days_minimum_lag)
    
                # try to find files within i days
                for f_name2 in f_names2:
                    print('#### fname2')
                    if f_name2 != f_name:
                        ifile2 = f_name2 #'%s/%s' % (root, f_name2)
                        # If not the same polarization in both files
                        if not any( [( (polarization in f_name) and (polarization in f_name2) ) for polarization in polarizations] ):
                            continue 
    
                        date_m2 = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', f_name2)
    
                        dt_i_str = '%s/%s/%sT%s:%s:%s' % (date_m2[0][0:4], date_m2[0][4:6], date_m2[0][6:8],
                                                         date_m2[0][9:11], date_m2[0][11:13], date_m2[0][13:15])
                        dt_i = datetime.strptime(dt_i_str, '%Y/%m/%dT%H:%M:%S')

    
                        # If the i date within current time gap
                        if ( ( (dt_i <= dt0_minimum_lag) and (dt_i >= dt0_lag) ) or ( (dt_i >= dt0_minimum_lag_plus) and (dt_i <= dt0_lag_plus) ) ):
                            print(f'dt_i {dt_i}')
    
                            # See if pair is already represented
                            if not isPairAlreadyPresent( ifile, ifile2, out_path ):
                                print('making...')

                                # Get ID 
                                id_pair = acquireID(out_path) 
                                # print('\nFile 1: %s \nFile 2: %s' % (ifile, ifile2) )
                                # Save pair 
                                res = check_save_pair(ifile, ifile2, out_path, id_pair, polarizations, intersect_ratio, maximum_drift_speed = max_drift_speed)
    
                                if res == 1:
                                    print('\nTime lag is %.1f [hours]' % abs((dt_i - dt0).total_seconds() / 3600))
                                    print('\nMaking pair %03d ... ' % id_pair)
                                    print('Done.\n')
                    else:
                        pass


    # Remove temporary image files if existing
    if os.path.isfile("%s/temp1.tiff" % out_path):
        try:
            os.remove("%s/temp1.tiff" % out_path)
        except:
            pass
    if os.path.isfile("%s/temp2.tiff" % out_path):
        try:
            os.remove("%s/temp2.tiff" % out_path)
        except:
            pass

    



