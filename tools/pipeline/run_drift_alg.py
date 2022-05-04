'''

The script runs ice drift algorithm script (ice_drift_alg.m) and store the results
in mat file with a mask name: 'Chalmers_drift_date1-date2.mat' (see 'out_fname' variable)

You should specify the output grid step size by modifying 'grid_step'

path1 and path2 are paths to geotiff files

maxSpeed is the internal algorithm variable to limit search area

Example: run run_drift_alg.py grid_step path/output/folder

Created: 06-03-2021

'''

import os
import re
import gdal
import sys

# Output grid step oin pixels
gridStep = sys.argv[1]

# Output path
out_path = sys.argv[2]

path1 = '/data/rrs/s1/test/clip_HH_S1B_EW_GRDM_1SDH_20200301T083237_20200301T083346_020496_026D68_5471_adjusted.tif'
path2 = '/data/rrs/s1/test/clip_HH_S1B_EW_GRDM_1SDH_20200302T073529_20200302T073629_020510_026DD5_27F9_adjusted.tif'

# Get time difference
dt1 = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', os.path.basename(path1))[0]
dt2 = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', os.path.basename(path2))[0]

# m/s
maxSpeed = 0.4 # m/s

# get pixel size (in meters) from Geotiff file
ras = gdal.Open(path1)
pixelSz = ras.GetGeoTransform()[1]

# Output mat filename
out_fname = 'Chalmers_drift_%s-%s.mat' % (dt1, dt2)

# Run the drift algorithm
os.system("""matlab -nosplash -nodesktop -wait -r "ice_drift_alg(\'%s\', \'%s\', false, %.f, %.f, %.f, \'%s\'); quit();" """ %
          (path1, path2, pixelSz, gridStep, maxSpeed, out_fname))

