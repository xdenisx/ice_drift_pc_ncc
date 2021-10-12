import glob
from osgeo import gdal, osr
import sys
import re
import os
from geojson import Polygon, Feature, FeatureCollection, dump

def convert_pt(x, y, prj_src):
    '''
    Get coordinates of point in geographical projection

    :param x: X-coordinate
    :param y: Y-coordinate
    :param prj_src: source projection in WKT format
    :return: a pair of geographical projected coordinates (lon, lat)
    '''

    p1 = osr.SpatialReference()
    p1.ImportFromWkt(prj_src)
    p2 = osr.SpatialReference()
    p2.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(p1, p2)
    return transform.TransformPoint(x, y)


def get_bbox(raster1):
    '''
    Find bounding box of GeoTiff in geographical projection coordinates

    :param raster1:
    :return:
    '''
    gt1 = raster1.GetGeoTransform()

    # find image's bounding box
    # r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * raster1.RasterXSize), gt1[3] + (gt1[5] * raster1.RasterYSize)]
    #print('\t1 bounding box: %s' % str(r1))

    prj_srs = ds.GetProjection()

    xy_ul = convert_pt(r1[0], r1[1], prj_srs)
    xy_br = convert_pt(r1[2], r1[3], prj_srs)

    return [xy_ul[0], xy_ul[1], xy_br[0], xy_br[1]]

def get_dt(fname):
    '''
    Get date and time from a filename

    :param fname: str file name
    :return: extracted datetime
    '''

    m = re.findall(r'\d{8}\w\d{6}', fname)
    if m[0]:
        return m[0]
    else:
        return None

# Path to input gtiff files
input_gtiff_files = sys.argv[1]

# Output path for downloading
out_s1_path = sys.argv[2]

# Acquire maximum number of hours apart from image that a s1 image should be retrieved
max_hours = "24"
if len(sys.argv) >= 4:
    max_hours = sys.argv[3]
min_hours = "0"
if len(sys.argv) >= 5:
    min_hours = sys.argv[4]
platform = "Sentinel-1A"
if len(sys.argv) >= 6:
    platform = sys.argv[5]
polarization = "HH+HV"
if len(sys.argv) >= 7:
    polarization = sys.argv[6]

# Temporary json file for searching
temp_json_fname = f'{out_s1_path}/temp.json'

# Create directory for meta links
os.makedirs( f'{out_s1_path}/metalinks', exist_ok=True)

# Loop through all input files
for root, dirs, files in os.walk(input_gtiff_files): #, topdown=False
    for fname in files:
        if fname.endswith('tiff') or fname.endswith('tif'):
            print(f'\nStart processing: {os.path.join(root, fname)}...\n')

            ds = gdal.Open(os.path.join(root, fname))

            # Find bounding box in geographical coordinates (EPSG:4326)
            bbox = get_bbox(ds)
            bbox = [[(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3]), (bbox[0], bbox[1])]]

            # Get date and time
            dt = get_dt(fname)
            del ds

            poly = Polygon(bbox)

            features = []
            features.append(Feature(geometry=poly, properties={"test": "test"}))

            feature_collection = FeatureCollection(features)
            with open( temp_json_fname, 'w') as f:
                dump(feature_collection, f)

            # Download metadata
            download_str = f'python3 ../s1_asf_download/asf_meta_download.py {out_s1_path} {temp_json_fname} {platform} {dt[0:8]} {dt[9:]} {min_hours} {max_hours} EW GRD_MD {polarization} '
            os.system(download_str)

            try:
                os.remove(temp_json_fname)
            except:
                pass

# Download data
try:
    meta_file = f'{out_s1_path}/metalinks/*.metalink*'
    meta_file = glob.glob( meta_file )
    print(meta_file)
    for iterFile in meta_file:
        download_str = f'python3 ../s1_asf_download/download_all.py {iterFile} {out_s1_path}'
        os.system(download_str)
except:
    print('\nError: A meta file does not exist or error during downloading.\n')
print(f'\nDone.\n')

