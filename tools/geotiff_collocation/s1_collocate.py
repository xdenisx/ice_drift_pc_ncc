try:
    import gdal
except:
    from osgeo import gdal
import sys
import os
import numpy as np
from RasterAdjuster import RasterAdjuster
import re
from datetime import datetime, timedelta

def findRasterIntersect(raster1, raster2):
    # load data
    band1 = raster1.GetRasterBand(1)
    band2 = raster2.GetRasterBand(1)
    gt1 = raster1.GetGeoTransform()
    gt2 = raster2.GetGeoTransform()

    # find each image's bounding box
    # r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * raster1.RasterXSize), gt1[3] + (gt1[5] * raster1.RasterYSize)]
    r2 = [gt2[0], gt2[3], gt2[0] + (gt2[1] * raster2.RasterXSize), gt2[3] + (gt2[5] * raster2.RasterYSize)]
    print('\t1 bounding box: %s' % str(r1))
    print('\t2 bounding box: %s' % str(r2))

    # find intersection between bounding boxes
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]
    if r1 != r2:
        print('\t** different bounding boxes **')
        # check for any overlap at all...
        if (intersection[2] < intersection[0]) or (intersection[1] < intersection[3]):
            intersection = None
            print('\t***no overlap***')
            return
        else:
            print('\tintersection:', intersection)
            left1 = int(round((intersection[0] - r1[0]) / gt1[1]))  # difference divided by pixel dimension
            top1 = int(round((intersection[1] - r1[1]) / gt1[5]))
            col1 = int(round((intersection[2] - r1[0]) / gt1[1])) - left1  # difference minus offset left
            row1 = int(round((intersection[3] - r1[1]) / gt1[5])) - top1

            left2 = int(round((intersection[0] - r2[0]) / gt2[1]))  # difference divided by pixel dimension
            top2 = int(round((intersection[1] - r2[1]) / gt2[5]))
            col2 = int(round((intersection[2] - r2[0]) / gt2[1])) - left2  # difference minus new left offset
            row2 = int(round((intersection[3] - r2[1]) / gt2[5])) - top2

            # print '\tcol1:',col1,'row1:',row1,'col2:',col2,'row2:',row2
            if col1 != col2 or row1 != row2:
                print("*** MEGA ERROR *** COLS and ROWS DO NOT MATCH ***")
            # these arrays should now have the same spatial geometry though NaNs may differ
            array1 = band1.ReadAsArray(left1, top1, col1, row1)
            array2 = band2.ReadAsArray(left2, top2, col2, row2)

    else:  # same dimensions from the get go
        col1 = raster1.RasterXSize  # = col2
        row1 = raster1.RasterYSize  # = row2
        array1 = band1.ReadAsArray()
        array2 = band2.ReadAsArray()

    return array1, array2, col1, row1, intersection

def check_save_pair(f1, f2, id_pair):
    image1_ds = gdal.Open(f1)
    image2_ds = gdal.Open(f2)

    gt = image1_ds.GetGeoTransform()
    pixel_area = abs(gt[1] / 1000. * gt[-5] / 1000.)  # [km]

    try:
        image1_isect_array, image2_isect_array, col, row, isect_bb = findRasterIntersect(image1_ds, image2_ds)
        intersect_area = pixel_area * col * row
        print('\nIntersect area for:\n%s\n%s\n\n %.1f [km2]' %
              (os.path.basename(f1), os.path.basename(f2), intersect_area))

        if intersect_area > 100000:
            print('\n### Start making pair... ###')
            # Create dir for a pir
            try:
                os.makedirs('%s/%02d' % (out_path, id_pair))
            except:
                pass

            print('\nStart adjusment...')
            adjuster = RasterAdjuster(f1, f2)
            adjuster.export(raster1_export_path='%s/%02d/%s' % (out_path, id_pair, os.path.basename(f1)),
                            raster2_export_path='%s/%02d/%s' % (out_path, id_pair, os.path.basename(f2)),
                            mask_export_path='%s/%02d/mask_%s' % (out_path, id_pair, os.path.basename(f1)))
            print('Adjusment done.\n')
            return 1
        else:
            return 0
    except:
        return 0

    del image1_ds
    del image2_ds

in_path = sys.argv[1]
out_path = sys.argv[2]
days_lag = int(sys.argv[3])

try:
    os.makedirs(out_path)
except:
    pass

polarization = 'hh'
files_pref = 'UPS'
id_pair = 0

for root, d_names, f_names in os.walk(in_path):
    #f_names.sort()
    f_names = [ff for ff in f_names if ff.endswith('tiff')]
    f_names.sort(key=lambda x: os.path.basename(x).split('_')[6], reverse=True)

    while len(f_names) != 0:
        f_name = f_names[0]
        f_names.pop(0)

        if f_name.startswith(files_pref) and f_name.endswith('tiff') and f_name.find(polarization) > 0:
            ifile = '%s/%s' % (root, f_name)
            date_m = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', f_name)

            if not date_m is None:
                dt0_str = '%s/%s/%sT%s:%s:%s' % (date_m[0][0:4], date_m[0][4:6], date_m[0][6:8],
                                                 date_m[0][9:11], date_m[0][11:13],date_m[0][13:15])

                # Date time of a current file
                dt0 = datetime.strptime(dt0_str, '%Y/%m/%dT%H:%M:%S')

                # Date time of a current file minus time lag
                dt0_lag = dt0 - timedelta(days=days_lag)

                # try to find files within i days
                print()
                for f_name2 in f_names:
                    ifile2 = '%s/%s' % (root, f_name2)

                    date_m2 = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', f_name2)

                    dt_i_str = '%s/%s/%sT%s:%s:%s' % (date_m2[0][0:4], date_m2[0][4:6], date_m2[0][6:8],
                                                     date_m2[0][9:11], date_m2[0][11:13], date_m2[0][13:15])
                    dt_i = datetime.strptime(dt_i_str, '%Y/%m/%dT%H:%M:%S')

                    # If the i date within current time gap
                    if dt_i >= dt0_lag and dt_i < dt0:
                        print('\nTime lag is %.1f [hours]' % abs((dt_i-dt0).total_seconds()/3600))
                        print('\nMaking pair %02d ... ' % id_pair)
                        res = check_save_pair(ifile, ifile2, id_pair)
                        if res == 1:
                            id_pair += 1
                        print('Done.\n')
