import os
import re
from sentinel_calibration import *
import sys
import glob
from os.path import expanduser
from pathlib import Path
import shutil

home = expanduser("~")

in_path = sys.argv[1] # /data/rrs/s1/fs
out_path = sys.argv[2]

reproject = True

polarizations = ['HH', 'VV']
polarizations = [x.lower() for x in polarizations]

# GRD resolutions
resolution = {}
resolution['GRDM'] = 200.
resolution['GRDH'] = 20.

print('\nTarget polarizations: %s\n' % polarizations)

#proj4_str = '+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'

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
        #if f_name == f_names[0]: #'20210511' in f_name:
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
                os.makedirs('%s/%s' % (out_path, m), exist_ok=True)
                out_calib_name = '%s/%s/_%s_%s.tiff' % (out_path, m, pol, tmp_fname)

                try:
                    os.remove(out_calib_name)
                except:
                    print("Error while deleting file ", out_calib_name)

                try:
                    perform_radiometric_calibration(pol_file, calib_fname, out_calib_name)
                except:
                    pass

                if reproject:
                    print('\nReptojecting...')
                    # Reproject and save to tiff in UPS
                    out_tiff_name = '%s/%s/UPS_%s_%s.tiff' % (out_path, m, pol, tmp_fname)

                    if not os.path.isfile(out_tiff_name):
                        if 'GRDM' in out_tiff_name:
                            res = resolution['GRDM']
                            try:
                                reproject_ps(out_calib_name, out_tiff_name, 3995, res, disk_output=True)
                            except:
                                print('Can not reproject')

                        elif 'GRDH' in out_tiff_name:
                            res = resolution['GRDH']
                            try:
                                reproject_ps(out_calib_name, out_tiff_name, 3995, res, disk_output=True)
                            except:
                                print('Can not reproject')

                        # Delete calibrated unprojected tiff
                        try:
                            os.remove(out_calib_name)
                        except:
                            pass
                        print('Done.\n')
                    else:
                        os.remove(out_calib_name)
                        print('\nFile: %s exist!\n' % out_tiff_name)

        # Remove unziped SAFE folder
        shutil.rmtree(path_to_safe_file, ignore_errors=True)