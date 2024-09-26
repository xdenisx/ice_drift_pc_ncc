##########################################################
# Script to create geofile for data gathering
##########################################################
import argparse
import geojson
import sys

def parse_args():
    """ Parse input arguments """

    parser = argparse.ArgumentParser(description='Make a geojson file to gather data from Alaska SAT Faility server')

    # Mandatory arguments
    parser.add_argument('geometry_type', choices=['point', 'polygon'], help='Type of geometry')
    parser.add_argument('geometry', help='Coordinates of geometry object (point or polygon).\nIf Point: lon,lat\nif Polygon: lon0 lat0,lon1 lat1,...,lon0 lat0')
    parser.add_argument('output_file', help='Path to output geojson with a geometry')

    return parser.parse_args()

# Parse arguments
args = parse_args()

if args.geometry_type == 'point':
    # Create a Point object
    # s = '0,79'
    geom = geojson.Point((float(args.geometry.split(',')[0]), float(args.geometry.split(',')[1])))
elif args.geometry_type == 'polygon'::
    # Create Polygon object
    # s = '0 79,0 80,10 80,0 79'
    geom = geojson.Polygon[tuple(int(i) for i in x.split()) for x in args.geometry.split(',')]

with open(args.output_file, 'w') as file:
    geojson.dump(geom, file)

# Now, let's read the data back from the file to verify
with open(args.output_file, 'r') as file:
    geojson_data = geojson.load(file)
print(geojson_data)