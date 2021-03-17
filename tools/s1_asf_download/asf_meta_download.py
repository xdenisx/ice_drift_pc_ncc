# -*- coding: utf-8 -*-

##########################################
# 17/03/2021
# Created by Denis Demchev
##########################################

'''
    RETRIEVE AND DOWNLOAD METALINK FILE FROM ASF API FOR S1
'''

# Example usage: run asf_download.py /PATH/TO/GEO/FILE Sentinel-1B 20210201 20210203 EW HH+HV GRD_MD


import os
import json
import sys
from datetime import datetime
#import shapefile
import argparse

def parse_args():
    """ Parse input arguments """

    parser = argparse.ArgumentParser(description='Download metalink file from Alaska SAR Faility server')

    # Mandatory arguments
    parser.add_argument('geo_file', help='Path to geojson/shapefile with a geometry')
    parser.add_argument('platform', choices=['Sentinel-1A', 'Sentinel-1B'], help='Sentinel-1A/Sentinel-1B')
    parser.add_argument('date1', help='Date 1 (YYYYMMDD)')
    parser.add_argument('date2', help='Date 2 (YYYYMMDD)')
    parser.add_argument('mode', help='Acquisition mode (EW/IW)', choices=['EW', 'IW'])
    parser.add_argument('polarisation', help='Polarization (HH+HV/VV+VH)', choices=['HH+HV', 'VV+VH'])
    parser.add_argument('grd_level', help='Processing level (GRD mode)', choices=['GRD_MD', 'GRD_HD'])

    # Optional arguments
    parser.add_argument('orbit_num', help='Orbital number', type=int, nargs='?')
    parser.add_argument('flight_direction', help='Flight direction (ascending/descending)',
                        choices=['Ascending', 'Descending'], nargs='?')

    return parser.parse_args()

asf_url = 'https://api.daac.asf.alaska.edu/services/search'

args = parse_args()

# Chake dates format
try:
    dt1 = datetime.strptime(args.date1, '%Y%m%d')
    dt2 = datetime.strptime(args.date2, '%Y%m%d')
    if dt2 < dt1:
        print('Error: date2 > date1!\nStop executing\n')
except:
    print('\nError: date1 or date2 format is wrong!\nMust be: YYYYMMDD\nStop executing\n')

# Modify polarization format
if args.polarisation.find('+') > 0:
    polarisation = '%s%%2b%s' % (args.polarisation.split('+')[0], args.polarisation.split('+')[1])

# Metalink output file name
current_directory = os.getcwd()
metafile_path = '%s/metalinks' % current_directory

try:
    os.makedirs()
except:
    pass

fname_meta = '%s/%s_%s%02d%02d%02d%02d-%s%02d%02d%02d%02d.metalink' % \
             (metafile_path, args.platform,  dt1.year, dt1.month, dt1.day, dt1.hour, dt1.minute,
              dt2.year, dt2.month, dt2.day, dt2.hour, dt2.minute)


# Check geo file geometry
with open(args.geo_file) as f:

    # If GeoJSON
    if f.name.endswith('json'):
        data = json.load(f)
        try:
            geo_file_geom = data['type']
            geo_file_geom in ['Point', 'FeatureCollection']
        except:
            raise ValueError('\nError: A GeoJSON file must contain POINT or POLYGON geometry\n')

    # If shapefile
    '''
    if f.name.endswith('shp'):
        shape = shapefile.Reader(f)
        feature = shape.shapeRecords()[0]
        coord_str = 'point%28' + '%.1f' % feature.shape.points[0][0] + '+' \
                    + '%.1f' % feature.shape.points[0][1] + '%29'
    '''

coord_str = ''

if geo_file_geom == 'Point':
    coord_str = 'point%28' + '%.1f' % data['coordinates'][0] + '+' \
                + '%.1f' % data['coordinates'][1] + '%29'

    str_download = "wget --no-check-certificate -O %s %s/param?intersectsWith=" \
                       "%s\\&platform=%s\&start=%s-%02d-%02dT%02d:%02d:00UTC\\&end=%s-%02d-%02dT23:59:59UTC\\" \
                       "&beamMode=%s\\&polarization=%s\\&processingLevel=%s\\&output=metalink" \
                       % (fname_meta, asf_url, coord_str, args.platform, dt1.year, dt1.month, dt1.day, dt1.hour, dt1.minute,
                          dt2.year, dt2.month, dt2.day, args.mode, polarisation, args.grd_level)

    print('\nStart downloading...')
    os.system(str_download)
    print('Success\n\n'
          'Result stored into the file:\n %s' % fname_meta)


if geo_file_geom == 'FeatureCollection':
    for feature in data['features']:
        # List of the coordinates should be reversed as clockwise order is needed
        for coord in feature['geometry']['coordinates'][0]:
            #print(coord_str)
            coord_str += '%.2f,%.2f,' % (coord[0], coord[1])
    coord_str = coord_str[:-1]

    str_download = "wget --no-check-certificate -O %s %s/param?polygon=" \
                   "%s\\&platform=%s\&start=%s-%02d-%02dT%02d:%02d:00UTC\\&end=%s-%02d-%02dT23:59:59UTC\\" \
                   "&beamMode=%s\\&polarization=%s\\&processingLevel=%s\\&output=metalink" \
                   % (fname_meta, asf_url, coord_str, args.platform, dt1.year, dt1.month, dt1.day, dt1.hour, dt1.minute,
                      dt2.year, dt2.month, dt2.day, args.mode, polarisation, args.grd_level)

    print('\nStart downloading...')
    os.system(str_download)
    print('Success\n\n'
          'Result stored into the file:\n %s' % fname_meta)