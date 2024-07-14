import os
import re
from sentinel_calibration import *
import sys
import glob
from os.path import expanduser
import shutil
import math
try:
    from osgeo import gdal
except:
    import gdal
import xml.etree.ElementTree

home = expanduser("~")

in_path = sys.argv[1]
out_path = sys.argv[2]
# date in output file path
f_date_in_path = sys.argv[3]
proj_epsg = 5041
if len(sys.argv) >= 6:
    proj_epsg=int(sys.argv[5])

normalize_255 = False
if len(sys.argv) >= 7:
    normalize_255 = bool(sys.argv[5])
    print('Image will be normalized!')

# !TODO: remove    
remove_zip = True
print('Remove zip file: %s' % remove_zip)


reproject = True
save_metadata = True

polarizations = ['HH', 'VV', 'HV', 'VH']
polarizations = [x.lower() for x in polarizations]
modes = ['EW', 'IW']

# GRD resolutions
resolution = {}
tiff_res = math.ceil(float(sys.argv[4]))
resolution['GRDM'] = tiff_res
resolution['GRDH'] = tiff_res #math.ceil(tiff_res/2)


# Land and no data mask
f_mask = False

print('\nTarget polarizations: %s\n' % polarizations)

# Try to make temp dir
try:
    os.makedirs('%s/temp' % home)
except:
    print("Failed to create temp directory!")

if not os.path.exists(out_path):
    os.makedirs(out_path)

for root, d_names, f_names in os.walk(in_path):
    print(root, d_names, f_names)
    f_names = [fn for fn in f_names if fn.lower().endswith('zip')]
    f_names.sort(key=lambda x: os.path.basename(x).split('_')[4])

    for f_name in f_names:
        #if f_name == f_names[0]: #'20210511' in f_name:
        ifile = '%s/%s' % (root, f_name)
        path_to_safe_file = '%s/temp/%s.SAFE' % (home, os.path.basename(ifile.split('.')[0]))

        # Check file(s) exist
        m = re.findall(r'\d\d\d\d\d\d\d\dT', f_name[:-4])[0][:-1]
        if not os.path.isfile( '%s/%s/UPS_*_%s.tiff' % (out_path, m, f_name[:-4]) ):

            # Unzip
            shutil.rmtree(path_to_safe_file, ignore_errors=True)
            idate = os.path.basename(ifile).split('_')[4]
            unzip = 'unzip %s -d %s/temp' % (ifile, home)
            os.system(unzip)

            for pol in polarizations:

                if f_date_in_path == '1':
                    if not os.path.exists('%s/%s' % (out_path, m)):
                        os.makedirs('%s/%s' % (out_path, m))
                    out_tiff_name = '%s/%s/UPS_%s_%s.tiff' % (out_path, m, pol, f_name[:-4])
                else:
                    out_tiff_name = '%s/UPS_%s_%s.tiff' % (out_path, pol, f_name[:-4])

                try:
                    pol_file = glob.glob('%s/temp/%s.SAFE/measurement/*%s*.tiff' %
                                         (home, os.path.basename(ifile)[:-4], pol))[0]
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                    print('\n%s file found!\n' % pol)
                    print('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                except:
                    try:
                        pol_file = glob.glob('%s/temp/%s/measurement/*%s*.tiff' %
                                             (home, os.path.basename(ifile)[:-4], pol))[0]
                        print('\n%s file found!\n' % pol)
                    except:
                        pol_file = ''

                if pol_file != '':
                    calib_fname = glob.glob('%s/annotation/calibration/calibration*%s*.xml' % (path_to_safe_file, pol))[0]
                    tmp_fname = os.path.basename(ifile)[:-4]
                    m = re.findall(r'\d\d\d\d\d\d\d\dT', tmp_fname)[0][:-1]
                    os.makedirs('%s' % out_path, exist_ok=True)

                    if f_date_in_path == '1':
                        if not os.path.exists('%s/%s' % (out_path, m)):
                            os.makedirs('%s/%s' % (out_path, m))
                        out_calib_name = '%s/%s/_%s_%s.tiff' % (out_path, m, pol, tmp_fname)
                    else:
                        out_calib_name = '%s/_%s_%s.tiff' % (out_path, pol, tmp_fname)

                    # Calibration
                    perform_radiometric_calibration(pol_file, calib_fname, out_calib_name)

                    if reproject:
                        print('\nReprojecting...')
                        if 'GRDM' in out_tiff_name:
                            res = resolution['GRDM']
                            reproject_ps(out_calib_name, out_tiff_name, proj_epsg, res, disk_output=True,
                                         mask=f_mask, supress_speckle=True, normalize_255=normalize_255)

                        elif 'GRDH' in out_tiff_name:
                            res = resolution['GRDH']
                            reproject_ps(out_calib_name, out_tiff_name, proj_epsg, res, disk_output=True,
                                         mask=f_mask, supress_speckle=True, normalize_255=normalize_255)

                        # Delete calibrated unprojected tiff
                        try:
                            pass
                            os.remove(out_calib_name)
                        except:
                            pass
                        print('Reprojecting done.\n')
                        
                    if save_metadata:
                        
                        # Get all xml files in annotation
                        metadata_files = glob.glob('%s/annotation/*.xml' % (path_to_safe_file))
                        # Loop through all xml files
                        for metadata_file in metadata_files:
                            # If current xml file has a name identical to current pol-file
                            tempRegexp1 = re.match( "^(.*).tiff$", os.path.basename( pol_file ) )
                            tempRegexp2 = re.match( "^(.*).xml$", os.path.basename( metadata_file ) )
                            if ( tempRegexp1.group(1) == tempRegexp2.group(1) ):
                                
                                print( "\nSaving metadata" )
                                # Get geolocationGrid
                                geolocationGrid = get_geolocationGrid( metadata_file )
                                # Remove pixel and line elements
                                for gridPoint in geolocationGrid.findall(".//geolocationGridPoint"):
                                    for child in gridPoint.findall("./*"):
                                        if (child.tag == "pixel" or child.tag == "line"):
                                            gridPoint.remove( child )
                                
                                # If exists file 
                                if os.path.exists(out_calib_name):
                                
                                    # Open file
                                    opened_file = gdal.Open( out_calib_name )
                                    # Get metadata
                                    metadata = opened_file.GetMetadata( )
                                    # Insert geolocationGrid
                                    metadata["geolocationGrid"] = xml.etree.ElementTree.tostring( geolocationGrid )
                                    # Write to metadata
                                    opened_file.SetMetadata( metadata )
                                    # Flush to file
                                    opened_file.FlushCache()
                                
                                # If exists file 
                                if os.path.exists(out_tiff_name):
                                    # Open file
                                    opened_file = gdal.Open( out_tiff_name )
                                    # Get metadata
                                    metadata = opened_file.GetMetadata( )
                                    # Insert geolocationGrid
                                    metadata["geolocationGrid"] = xml.etree.ElementTree.tostring( geolocationGrid )
                                    # Write to metadata
                                    opened_file.SetMetadata( metadata )
                                    # Flush to file
                                    opened_file.FlushCache()
                                
                                
                                

            # Remove unziped SAFE folder
            shutil.rmtree(path_to_safe_file, ignore_errors=True)
            if remove_zip:
            	os.remove(ifile)
            else:
            	pass
            

        else:
            print('Error with %s' % os.path.basename(out_tiff_name))
