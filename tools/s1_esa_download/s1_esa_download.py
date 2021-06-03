import os
from datetime import datetime

td = datetime.today()
dt_str = '%s%02d%02d' % (td.year, td.month, td.day)

dt_str = '/mnt/s1/arctic/zip/%s' % dt_str

os.makedirs(dt_str, exist_ok=True)
#-c -179,60:179:89
sh_str = './dhusget.sh -u xdenis88x -p pdp20den20 -s 2021-05-31T06:00:00.000Z -m Sentinel-1 -T GRD -F \'( polarisationmode:HH,HH HV ) AND (footprint:\"Intersects(POLYGON((0.0 89.0, 179.0 89.0, 179.0 60.0, 0.0 60.0, 0.0 89.0)))\" )\' -o product -O %s' % dt_str
os.system(sh_str)
