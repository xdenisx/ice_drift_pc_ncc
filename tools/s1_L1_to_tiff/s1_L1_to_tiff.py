from sentinel_calibration import *
import sys
import glob
from os.path import expanduser
from pathlib import Path
import shutil

#ifile = '/data/rrs/s1/fs/s1b/ew/zip/S1B_EW_GRDM_1SDH_20200301T083237_20200301T083346_020496_026D68_5471.zip'

home = expanduser("~")

in_path = sys.argv[1] # /data/rrs/s1/fs
out_path = sys.argv[2]

proj4_str = '+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
grid_res = 100.


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

        # Remove unziped SAFE folder
        shutil.rmtree(path_to_safe_file, ignore_errors=True)

        # Unzip
        print('###############')
        print('\nUNZIPPING...\n')
        print('###############')
        idate = os.path.basename(ifile).split('_')[4]
        unzip = 'unzip %s -d %s/temp' % (ifile, home)
        os.system(unzip)
        print('###############')
        print('\nUNZIPPING END!\n')
        print('###############')

        try:
            hh_file = glob.glob('%s/temp/%s.SAFE/measurement/*hh*.tiff' % (home, os.path.basename(ifile)[:-4]))
            hh_file = hh_file[0]
            print('\nHH file found!\n')
        except:
            try:
                hh_file = glob.glob('%s/temp/%s/measurement/*hh*.tiff' % (home, os.path.basename(ifile)[:-4]))
                hh_file = hh_file[0]
                print('\nHH file found!\n')
            except:
                hh_file = ''
                print('\nHH file NOT found!\n')
        try:
            hv_file = glob.glob('%s/temp/%s.SAFE/measurement/*hv*.tiff' % (home, os.path.basename(ifile)[:-4]))
            hv_file = hv_file[0]
            print('\nHV file found!\n')
        except:
            try:
                hv_file = glob.glob('%s/temp/%s/measurement/*hv*.tiff' % (home, os.path.basename(ifile)[:-4]))
                hv_file = hv_file[0]
                print('\nHV file found!\n')
            except:
                hv_file = ''
                print('\nHV file NOT found!\n')

        try:
            if hh_file!='':
                calib_fname = glob.glob('%s/annotation/calibration/calibration*hh*.xml' % path_to_safe_file)[0]
                out_calib_name = '%s/HH_%s.tiff' % (out_path, os.path.basename(ifile)[:-4])
                perform_radiometric_calibration(hh_file, calib_fname, out_calib_name)

                # Reproject and save to tiff
                out_tiff = '%s/ups_%s' % (os.path.dirname(out_calib_name), os.path.basename(out_calib_name))

                if not os.path.isfile(out_tiff):
                    # Reproject geotiff
                    save_projected_geotiff(out_calib_name, proj4_str, grid_res, out_tiff)
                    # Delete calibrated unprojected tiff
                    os.remove(out_calib_name)
                else:
                    os.remove(out_calib_name)
                    print('\nFile: %s exist!\n' % out_tiff)

            if hv_file!='':
                calib_fname = glob.glob('%s/annotation/calibration/calibration*hv*.xml' % path_to_safe_file)[0]
                out_calib_name = '%s/HV_%s.tiff' % (out_path, os.path.basename(ifile)[:-4])
                perform_radiometric_calibration(hv_file, calib_fname, out_calib_name)

                # Reproject and save to tiff
                out_tiff = '%s/ups_%s' % (os.path.dirname(out_calib_name), os.path.basename(out_calib_name))
                save_projected_geotiff(out_calib_name, proj4_str, grid_res, out_tiff)

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
        except:
            # Remove unziped SAFE folder
            shutil.rmtree(path_to_safe_file, ignore_errors=True)



