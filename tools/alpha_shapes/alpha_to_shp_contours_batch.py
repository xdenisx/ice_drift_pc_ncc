from osgeo import ogr
import numpy as np
import collections
import os

import alpha_to_shp_contours


def list_files(path):
    all = (os.path.join(path, item) for item in os.listdir(path))
    return [item for item in all if os.path.isfile(item)]

INDIR = '/home/vsel/res/ESS/P2/dec2018/out_alpha/'
OUTDIR = '/home/vsel/res/ESS/P2/dec2018/shp_contour/'


flist = list_files(INDIR)

for i in range(len(flist)):
    basename = os.path.basename(flist[i])
    alpha_file = flist[i]
    shp_file = OUTDIR+basename[:-4]+'_contour.shp'
    prj_file = OUTDIR+basename[:-4]+'_contour_selfmade.prj'
    print(f'handling {alpha_file}')
    segments = alpha_to_shp_contours.read_alpha_segments(alpha_file)
    contours = alpha_to_shp_contours.make_contours(segments)
    alpha_to_shp_contours.write_shape_file(shp_file, prj_file, contours)