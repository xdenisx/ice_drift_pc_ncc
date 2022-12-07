import os
from datetime import datetime, timedelta

td = datetime.today()
td_yest = datetime.today() - timedelta(days=1)
dt_str = '%s%02d%02d' % (td.year, td.month, td.day)

dt_str_yest = '%s-%02d-%02d' % (td_yest.year, td_yest.month, td_yest.day)

dt_str = '/mnt/s1/arctic/zip/%s' % dt_str

os.makedirs(dt_str, exist_ok=True)
#-c -179,60:179:89
sh_str = '/home/denis/git/ice_drift_pc_ncc/tools/s1_esa_download/dhusget.sh -u XXXXX -p XXXXX -s %sT00:00:00.000Z -m Sentinel-1 -T GRD -F \'( polarisationmode:HH,HH HV ) AND (footprint:\"Intersects(POLYGON((0.0 89.0, 179.0 89.0, 179.0 60.0, 0.0 60.0, 0.0 89.0)))\" )\' -o product -O %s' % \
         (dt_str_yest, dt_str)
os.system(sh_str)
