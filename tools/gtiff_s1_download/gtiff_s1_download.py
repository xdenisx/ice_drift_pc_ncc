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
    return transform.TransformPoint(y, x)

def GetExtent(ds):
    """ Return list of corner coordinates from a gdal Dataset """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    return (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)

def ReprojectCoords(coords,src_srs,tgt_srs):
    """ Reproject a list of x,y coordinates. """
    trans_coords=[]
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(y, x)
        trans_coords.append([x,y])

    return trans_coords

def get_bbox(ds):
    '''
    Find bounding box of GeoTiff in geographical projection coordinates
    :param ds:
    :return bounding box:
    '''

    ext = GetExtent(ds)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    # tgt_srs=osr.SpatialReference()
    # tgt_srs.ImportFromEPSG(4326)
    tgt_srs = src_srs.CloneGeogCS()
    geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)
    print('\t1 bounding box: %s' % str(geo_ext))

    return geo_ext #[xy_ul[1], xy_ul[0], xy_br[1], xy_br[0]]

def get_dt(fname):
    '''
    Get date and time from a filename

    :param fname: str file name
    :return: extracted datetime
    '''

    m = re.findall(r'\d{8}\w\d{6}', fname)
    if len(m) == 0:
        return None
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
platform = "Sentinel-1B"
if len(sys.argv) >= 6:
    platform = sys.argv[5]
polarization = "HH+HV"
if len(sys.argv) >= 7:
    polarization = sys.argv[6]

aq_mode = 'EW'
grd_mode = 'GRD_MD'

# Temporary json file for searching
temp_json_fname = f'{out_s1_path}/temp.json'

# Create directory for meta links
os.makedirs(f'{out_s1_path}/metalinks', exist_ok=True)

# Loop through all input files
for root, dirs, files in os.walk(input_gtiff_files): #, topdown=False
    for fname in files:
        if fname.endswith('tiff') or fname.endswith('tif'):
            print(f'\nStart processing: {os.path.join(root, fname)}...\n')
            ds = gdal.Open(os.path.join(root, fname))
            bbox = get_bbox(ds)
            bbox = [[(bbox[0][1], bbox[0][0]), (bbox[1][1], bbox[1][0]), (bbox[2][1], bbox[2][0]),
                    (bbox[3][1], bbox[3][0]), (bbox[0][1], bbox[0][0])]]

            dt = get_dt(fname)
            if dt is None:
                continue
            del ds

            poly = Polygon(bbox)

            features = []
            features.append(Feature(geometry=poly, properties={"test": "test"}))

            feature_collection = FeatureCollection(features)
            with open(temp_json_fname, 'w') as f:
                dump(feature_collection, f)

            # Download metadata
            download_str = f'python3 ../s1_asf_download/asf_meta_download.py {out_s1_path} {temp_json_fname} {platform} {dt[0:8]} {dt[9:]} {min_hours} {max_hours} {aq_mode} {grd_mode} {polarization}'
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
    print(f'\nDone.\n')
except:
    print('\nError: A meta file does not exist or error during downloading.\n')


