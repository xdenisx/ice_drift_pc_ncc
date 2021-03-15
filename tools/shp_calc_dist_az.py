from shapely.geometry import mapping, shape
import fiona
import pyproj

def calc_distance_azimuth(lon1, lat1, lon2, lat2):
    ''' Calculation of distance and azimuth direction (0 - to North)
        between two points from coordinates given in decimal degrees
    '''

    geod = pyproj.Geod(ellps='WGS84')
    az, az2, mag = geod.inv(lon1, lat1, lon2, lat2)

    if az < .0:
        az = az + 360.0

    print('Magnitude: %.2f [m]\nAzimuth: %.2f\n' % (mag, az))
    return mag, az


# Define path to folder with shapefile and input output(modified) file names
dir_path = '/PATH/TO/DIRECTORY'
input_file = '%s/f1_name.shp' % dir_path
output_file = '%s/modified_f1_name.shp' % dir_path

# Read the original Shapefile
with fiona.collection(input_file, 'r') as input:
    # The output has the same schema
    schema = input.schema.copy()
    input_crs = input.crs

    # Create new fields to put distance and azimuth direction
    schema['properties']['Distance_m'] = 'float'
    schema['properties']['Azimuth'] = 'float'

    # write a new shapefile
    with fiona.collection(output_file, 'w', 'ESRI Shapefile', schema, input_crs) as output:
        for elem in input:
            # Get coordinates from shapefile feature
            lon1, lat1 = elem['geometry']['coordinates'][0][0], elem['geometry']['coordinates'][0][1]
            lon2, lat2 = elem['geometry']['coordinates'][1][0], elem['geometry']['coordinates'][1][1]

            # Calculate distance and azimuth
            mag, az = calc_distance_azimuth(lon1, lat1, lon2, lat2)

            elem['properties']['Distance_m'] = mag
            elem['properties']['Azimuth'] = az

            output.write({'properties': elem['properties'], 'geometry': mapping(shape(elem['geometry']))})