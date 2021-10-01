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
from datetime import datetime, timedelta
#import shapefile
import argparse

def parse_args():
    """ Parse input arguments """

    parser = argparse.ArgumentParser(description='Download metalink file from Alaska SAR Faility server')

    # Mandatory arguments
    parser.add_argument('temp_dir', help='Path to a directory to create temporary files in')
    parser.add_argument('geo_file', help='Path to geojson/shapefile with a geometry')
    parser.add_argument('platform', choices=['Sentinel-1A', 'Sentinel-1B', 'Sentinel-1A,Sentinel-1B'], help='Sentinel-1A/Sentinel-1B')
    parser.add_argument('date', help='Date (YYYYMMDD)')
    parser.add_argument('time', help='Time  (hhmmss)')
    parser.add_argument('min_hours_delta', help='Hours (hhmmss)')
    parser.add_argument('max_hours_delta', help='Hours (hhmmss)')
    parser.add_argument('mode', help='Acquisition mode (EW/IW)', choices=['EW', 'IW'])
    #parser.add_argument('polarisation', help='Polarization (HH+HV/VV+VH)', choices=['HH+HV', 'VV+VH'])
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
    date = datetime.strptime(args.date, '%Y%m%d')
    date = datetime.combine( date, datetime.strptime(args.time, '%H%M%S').time() )
    dt1 = date + timedelta( hours = int(args.min_hours_delta) )
    dt2 = date + timedelta( hours = int(args.max_hours_delta) )
except Exception as e:
    print(e)
    print('\nError: date and/or time format is wrong !\nMust be: YYYYMMDD and HHMMSS \nStop executing\n')




# Modify polarization format
#if args.polarisation.find('+') > 0:
#    polarisation = '%s%%2b%s' % (args.polarisation.split('+')[0], args.polarisation.split('+')[1])

# Metalink output file name
metafile_path = '%s/metalinks' % args.temp_dir

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

if geo_file_geom == 'FeatureCollection':

    for feature in data['features']:
        try:
            # List of the coordinates should be reversed as clockwise order is needed
            for coord in feature['geometry']['coordinates'][0]:
                if len(coord_str) > 0:
                    coord_str += ','
                coord_str += '%.2f%s%.2f' % (coord[1], '%20', coord[0])
        except:
            for coord in feature['geometry']['coordinates'][0][0]:
                if len(coord_str) > 0:
                    coord_str += ','
                coord_str += '%.2f%s%.2f' % (coord[1], '%20', coord[0])
    #coord_str = coord_str[:-1]
    coord_str = 'polygon%28%28' + coord_str + '%29%29'
    #print('Coord_str: ' + coord_str)

str_download = "wget --no-check-certificate -O %s %s/param?" % (fname_meta, asf_url)
str_download += "intersectsWith=%s" % coord_str
str_download += "\&platform=%s" % args.platform
str_download += "\&start=%s-%02d-%02dT%02d:%02d:00UTC\&end=%s-%02d-%02dT23:59:59UTC" % ( dt1.year, dt1.month, dt1.day, dt1.hour, dt1.minute, dt2.year, dt2.month, dt2.day )
str_download += "\&beamMode=%s\&processingLevel=%s\&output=metalink" % (args.mode, args.grd_level)

print('\nStart downloading...')
os.system(str_download)
print('Success\n\n'
      'Result stored into the file:\n %s' % fname_meta)
