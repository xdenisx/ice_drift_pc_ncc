import os
import re
from sentinel_calibration import *
import sys
import glob
from os.path import expanduser
import shutil

home = expanduser("~")

in_path = sys.argv[1]
out_path = sys.argv[2]

reproject = True

polarizations = ['HH', 'VV']
polarizations = [x.lower() for x in polarizations]

# GRD resolutions
resolution = {}
resolution['GRDM'] = 200.
resolution['GRDH'] = 20.

print('\nTarget polarizations: %s\n' % polarizations)

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

        # Check projected tiff file exist
        m = re.findall(r'\d\d\d\d\d\d\d\dT', f_name[:-4])[0][:-1]
        out_tiff_name = '%s/%s/UPS_%s_%s.tiff' % (out_path, m, polarizations[0], f_name[:-4])

        if not os.path.isfile(out_tiff_name) and 'EW' in f_name:
            # Unzip
            shutil.rmtree(path_to_safe_file, ignore_errors=True)
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

                    # Calibration
                    perform_radiometric_calibration(pol_file, calib_fname, out_calib_name)

                    if reproject:
                        print('\nReptojecting...')
                        if 'GRDM' in out_tiff_name:
                            res = resolution['GRDM']
                            reproject_ps(out_calib_name, out_tiff_name, 3995, res, disk_output=True,
                                         mask=False)

                        elif 'GRDH' in out_tiff_name:
                            res = resolution['GRDH']
                            reproject_ps(out_calib_name, out_tiff_name, 3995, res, disk_output=True,
                                         mask=False)

                        # Delete calibrated unprojected tiff
                        try:
                            os.remove(out_calib_name)
                        except:
                            pass
                        print('Reprojecting done.\n')

            # Remove unziped SAFE folder
            shutil.rmtree(path_to_safe_file, ignore_errors=True)
        else:
            print('File %s exist' % os.path.basename(out_tiff_name))
