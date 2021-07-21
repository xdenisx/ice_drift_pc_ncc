import numpy as np
import datetime
import os
import glob
from scipy import nanmean
import sys
import pygeoj

from osgeo import osr

# get the existing coordinate system
old_cs = osr.SpatialReference()
old_cs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')


# create the new coordinate system
new_cs_proj4 = "+proj=stere +lat_0=90 +lat_ts=75 +lon_0=150 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
new_cs = osr.SpatialReference()
new_cs.ImportFromProj4(new_cs_proj4)
# create a transform object to convert between coordinate systems
transform = osr.CoordinateTransformation(old_cs, new_cs)

def json_to_txt(jsonfile):
    drift_json = pygeoj.load(jsonfile)
    lats1 = []
    lons1 = []
    lats2 = []
    lons2 = []
    x1 = []
    y1 = []
    drift_m = []
    n=0
    for feature in drift_json:
        n+=1
        print ('new feature', n)
        lat1 = feature._data['properties']['lat1']
        lon1 = feature._data['properties']['lon1']
        xy = transform.TransformPoint(lon1, lat1)
        lats1.append(feature._data['properties']['lat1'])
        lons1.append(feature._data['properties']['lon1'])
        lats2.append(feature._data['properties']['lat2'])
        lons2.append(feature._data['properties']['lon2'])
        x1.append(xy[0])
        y1.append(xy[1])
        drift_m.append(np.float(feature._data['properties']['drift_m']))

    return lats1,lons1,lats2,lons2,x1,y1,drift_m


INDIR = '/path/to/drift_output/json'
OUTDIR = '/path/to/output_dir/'

from os import listdir
from os.path import isfile, join
flist= [f for f in listdir(INDIR) if isfile(join(INDIR, f))]

for i in range(len(flist)):
    basename = flist[i][:-5]
    drift_infile = INDIR+flist[i]
    lats1,lons1,lats2,lons2,x1,y1,drift_m = json_to_txt(drift_infile)
    table = np.column_stack((lats1,lons1,lats2,lons2,x1,y1,drift_m))
    np.savetxt(OUTDIR+basename+'_ms.txt', table, header = 'lats1 lons1 lats2 lon2 x1 y1 drift_m', fmt='%1.2f')

    ind = np.where(np.array(drift_m)<150)
    x = np.array(x1)[ind]
    y = np.array(y1)[ind]
    f = open(OUTDIR+basename+'_alpha_proj.txt', 'w')
    f.writelines(str(len(x))+'\n')
    for i in range(len(x)):
        f.write('{} {}\n'.format(y[i], x[i]))
    f.close()
