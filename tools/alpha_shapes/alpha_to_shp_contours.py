from osgeo import ogr
import numpy as np
import collections
import sys

import make_attributes

def pop_segment(verts_to_seg, segment):
    verts_to_seg[segment[0]].remove(segment)
    if len(verts_to_seg[segment[0]]) == 0:
        verts_to_seg.pop(segment[0])
    verts_to_seg[segment[1]].remove(segment)
    if len(verts_to_seg[segment[1]]) == 0:
        verts_to_seg.pop(segment[1])


def make_contours(segments):
    if len(segments) == 0:
        return []

    verts_to_seg = collections.defaultdict(list)
    for segment in segments:
        if segment[0] == segment[1]:
            continue
        verts_to_seg[segment[0]].append(segment)
        verts_to_seg[segment[1]].append(segment)

    contours = []
    contour = []

    while len(verts_to_seg) > 0:
        if len(contour) == 0:
            some_vert = next(iter(verts_to_seg.keys()))
            contour.append(some_vert)

        seg_to_append = verts_to_seg[contour[-1]][0]
        if seg_to_append[0] == contour[-1]:
            contour.append(seg_to_append[1])
        else:
            contour.append(seg_to_append[0])
        pop_segment(verts_to_seg, seg_to_append)

        if contour[-1] not in verts_to_seg:
            contours.append(contour)
            contour = []

    return contours


def make_shape_line(contour):
    line = ogr.Geometry(type=ogr.wkbLineString)
    for xy in contour:
        line.AddPoint_2D(xy[0],xy[1])
    return line


def add_fields_to_layer(layer, field_names):
    for f in field_names:
        field = ogr.FieldDefn(f, ogr.OFTString)
        field.SetWidth(200)
        layer.CreateField(field)


def add_attributes_to_feature(feature, attributes):
    for k, v in attributes.items():
        feature.SetField(k, v)


def write_shape_file(shp_file, prj_file, contours, attributes):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.CreateDataSource(shp_file)
    srs = ogr.osr.SpatialReference()
    srs.ImportFromProj4("+proj=stere +lat_0=90 +lat_ts=75 +lon_0=150 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    layer = datasource.CreateLayer('layerName', geom_type=ogr.wkbLineString, srs=srs)
    add_fields_to_layer(layer, (k for k, _ in attributes.items()))
    
    for contour in contours:
        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry( make_shape_line(contour))
        add_attributes_to_feature(feature, attributes)
        layer.CreateFeature(feature)
        feature = None

    # Save and close DataSource
    datasource = None

    # Save prj file
    esri_output = srs.ExportToWkt()
    with open(prj_file, 'a') as f:
        f.write(esri_output)


def read_alpha_segments(alpha_file):
    with open(alpha_file, 'r') as infile:
        num_seg = int(infile.readline())
        segments = []
        for i in range(num_seg):
            l = infile.readline()
            c = l.split(' ')

            segments.append(((float(c[1]), float(c[0])), (float(c[3]), float(c[2]))))

    return segments


def make_shape_file(f):
    alpha_file = f
    print(f'making shape file for {alpha_file}')
    shp_file = f+'.contour.shp'
    prj_file = f+'.contour_selfmade.prj'
    segments = read_alpha_segments(alpha_file)
    contours = make_contours(segments)
    attributes = make_attributes.make_attributes(alpha_file)
    write_shape_file(shp_file, prj_file, contours, attributes)

# TODO add source files attributes
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('expected input file')

    make_shape_file(sys.argv[1])