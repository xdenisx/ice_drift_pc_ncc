from sentinel_calibration import *
import sys
import glob
from os.path import expanduser
from pathlib import Path
import shutil

home = expanduser("~")

in_path = sys.argv[1] # /data/rrs/s1/fs
out_path = sys.argv[2]
grid_res = sys.argv[3]

reproject = False
polarizations = ['HH', 'VV']
polarizations = [x.lower() for x in polarizations]

print('\nTarget polarizations: %s\n' % polarizations)

proj4_str = '+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'

# Try to make temp dir
try:
    os.makedirs('%s/temp' % home)
except:
    pass

if not os.path.exists(out_path):
    os.makedirs(out_path)

for root, d_names, f_names in os.walk(in_path):
    print(root, d_names, f_names)
    for f_name in f_names:
        ifile = '%s/%s' % (root, f_name)
        path_to_safe_file = '%s/temp/%s.SAFE' % (home, os.path.basename(ifile.split('.')[0]))
        shutil.rmtree(path_to_safe_file, ignore_errors=True)

        # Unzip
        idate = os.path.basename(ifile).split('_')[4]
        unzip = 'unzip %s -d %s/temp' % (ifile, home)
        os.system(unzip)

        for pol in polarizations:
            try:
                pol_file = glob.glob('%s/temp/%s.SAFE/measurement/*%s*.tiff' %
                                     (home, os.path.basename(ifile)[:-4], pol))[0]
                print('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                print(pol_file)
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                print('\n%s file found!\n' % pol)
            except:
                try:
                    pol_file = glob.glob('%s/temp/%s/measurement/*%s*.tiff' %
                                         (home, os.path.basename(ifile)[:-4], pol))[0]
                    print('\n%s file found!\n' % pol)
                except:
                    pol_file = ''

            if pol_file != '':
                calib_fname = glob.glob('%s/annotation/calibration/calibration*%s*.xml' % (path_to_safe_file, pol))[0]
                out_calib_name = '%s/%s_%s.tiff' % (out_path, os.path.basename(ifile)[:-4], pol)
                perform_radiometric_calibration(pol_file, calib_fname, out_calib_name)

                if reproject:
                    # Reproject and save to tiff in UPS
                    out_tiff = '%s/ups_%s' % (os.path.dirname(out_calib_name), os.path.basename(out_calib_name))

                    if not os.path.isfile(out_tiff):
                        # Reproject geotiff
                        save_projected_geotiff(out_calib_name, proj4_str, grid_res, out_tiff)
                        # Delete calibrated unprojected tiff
                        os.remove(out_calib_name)
                    else:
                        os.remove(out_calib_name)
                        print('\nFile: %s exist!\n' % out_tiff)

        # Remove unziped SAFE folder
        shutil.rmtree(path_to_safe_file, ignore_errors=True)