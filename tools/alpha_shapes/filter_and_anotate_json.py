import os
from scipy import nanmean
import sys
import pygeoj
import sys

from osgeo import osr

import make_attributes

# get the existing coordinate system
old_cs = osr.SpatialReference()
old_cs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')


# create the new coordinate system
new_cs_proj4 = "+proj=stere +lat_0=90 +lat_ts=75 +lon_0=150 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
new_cs = osr.SpatialReference()
new_cs.ImportFromProj4(new_cs_proj4)
# create a transform object to convert between coordinate systems
transform = osr.CoordinateTransformation(old_cs, new_cs)

def json_to_txt(drift_json):
    lats1 = []
    lons1 = []
    lats2 = []
    lons2 = []
    x1 = []
    y1 = []
    drift_m = []
    cc = []
    n=0
    for feature in drift_json:
        n+=1
        lat1 = feature._data['properties']['lat1']
        lon1 = feature._data['properties']['lon1']
        xy = transform.TransformPoint(lon1, lat1)
        lats1.append(feature._data['properties']['lat1'])
        lons1.append(feature._data['properties']['lon1'])
        lats2.append(feature._data['properties']['lat2'])
        lons2.append(feature._data['properties']['lon2'])
        x1.append(xy[0])
        y1.append(xy[1])
        drift_m.append(float(feature._data['properties']['drift_m']))

    return lats1,lons1,lats2,lons2,x1,y1,drift_m


def filter(drift_json, drift_upper_bound, cor_lower_bound, attributes):
    filtered_json = pygeoj.new()
    n = 0
    for feature in drift_json:
        if float(feature._data['properties']['drift_m']) < drift_upper_bound and float(feature._data['properties']['cor']) > cor_lower_bound:
            drift = float(feature._data['properties']['drift_m'])
            cor=float(feature._data['properties']['cor'])
            geom = pygeoj.Geometry(type='Point', coordinates=
                    feature._data['geometry']['coordinates'][0]
            )
            properties = {
                **feature._data['properties'],
                **attributes
            }
            new_feature = pygeoj.Feature(geometry=geom, properties=properties)

            filtered_json.add_feature(obj=new_feature)
            n += 1
    
    print(f'num filtered {n}')
    return filtered_json, n


drift_infile = sys.argv[1]
print('\nproccessing ' + drift_infile)
attributes = make_attributes.make_attributes(drift_infile)

drift_json = pygeoj.load(drift_infile)

filtered_json, num_features = filter(
    drift_json, 
    drift_upper_bound=150, 
    cor_lower_bound=0.55,
    attributes=attributes)
if num_features > 0:
    filtered_json.save(drift_infile+'.filtered.json')
else:
    print('No features left after filtering')

#lats1,lons1,lats2,lons2,x1,y1,drift_m = json_to_txt(drift_json)
#table = np.column_stack((lats1,lons1,lats2,lons2,x1,y1,drift_m))
#np.savetxt(drift_infile+'._ms.txt', table, header = 'lats1 lons1 lats2 lon2 x1 y1 drift_m', fmt='%1.2f')

#ind = np.where(np.array(drift_m)<150)
#x = np.array(x1)[ind]
#y = np.array(y1)[ind]
#f = open(drift_infile+'._alpha_proj.txt', 'w')
#f.writelines(str(len(x))+'\n')
#for i in range(len(x)):
#    f.write('{} {}\n'.format(y[i], x[i]))
#f.close()
