# -*- coding: utf-8 -*-
"""
RasterAdjuster class
Prerequisites: gdal>=2.1

Performs adjusting of two rasters (extent, resolution)

Author: Eduard Kazakov (silenteddie@gmail.com)
Last modification: 2017-08-03
"""

try:
    import gdal, ogr
except:
    from osgeo import gdal, ogr
import json, os

import numpy as np

path_to_coastline = '../../data/ne_50m_land.shp'

class RasterAdjuster():
    
    def __init__(self,raster1_path,raster2_path, intersection_extension = 0):
        
        # Store input pathes
        self.__raster1_path = raster1_path
        self.__raster2_path = raster2_path
        
        # Read datasets
        self.raster1 = gdal.Open(raster1_path)
        self.raster2 = gdal.Open(raster2_path)
        
        self.initFromRasters( self.raster1, self.raster2, intersection_extension )



    def initFromRasters( self, raster1, raster2, intersection_extension = 0 ):

        self.raster1 = raster1
        self.raster2 = raster2

        # From GCP to projected
        if self.__check_gcp_raster(self.raster1):
            self.raster1 = self.__gcp_raster_to_projected(self.raster1)
        if self.__check_gcp_raster(self.raster2):
            self.raster2 = self.__gcp_raster_to_projected(self.raster2)
        
        # Reproject second dataset to projection of first dataset
        if self.raster1.GetProjection() != self.raster2.GetProjection():
            self.raster2 = self.__reproject_raster_to_projection(self.raster2,self.raster1.GetProjection())
        
        # Get extents
        self.raster1_extent = self.__extent_to_wkt_polygon(self.__get_extent(self.raster1))
        self.raster2_extent = self.__extent_to_wkt_polygon(self.__get_extent(self.raster2))
        
        # Get intersection
        self.intersection = self.__intersect_two_wkt_polygons(self.raster1_extent,self.raster2_extent)
        # TODO: if intersection is empty
        
        # Extend intersection if requested 
        if intersection_extension > 0: 
            try:
                # Get geometry from intersection polygon
                intersection_geom = ogr.CreateGeometryFromWkt(self.intersection)
                ring = intersection_geom.GetGeometryRef(0)
                # Acquire extreme values
                extremes = [ ring.GetPoint(0)[0], ring.GetPoint(0)[1], ring.GetPoint(0)[0], ring.GetPoint(0)[1] ]
                for iterPoints in range(ring.GetPointCount()):
                    pt = ring.GetPoint( iterPoints )
                    extremes[0] = np.min( [extremes[0], pt[0]] )
                    extremes[1] = np.min( [extremes[1], pt[1]] )
                    extremes[2] = np.max( [extremes[2], pt[0]] )
                    extremes[3] = np.max( [extremes[3], pt[1]] )
                extremes = np.array( extremes ).reshape((2,2))
                extremes = np.mean( extremes, axis = 0 )
                # Extend all polygon points accordingly
                for iterPoints in range(ring.GetPointCount()):
                    pt = ring.GetPoint( iterPoints )
                    pt = np.array([pt[0], pt[1]])

                    if pt[0] >= np.mean( extremes[0] ):
                        pt[0] = pt[0] + intersection_extension
                    else:
                        pt[0] = pt[0] - intersection_extension
                    if pt[1] >= np.mean( extremes[1] ):
                        pt[1] = pt[1] + intersection_extension
                    else:
                        pt[1] = pt[1] - intersection_extension
                    ring.SetPoint( iterPoints, pt[0], pt[1] )
                self.intersection = intersection_geom.ExportToWkt()
            except Exception as e:
                print(str(e))
        
        # cut raster1 to intersection
        self.raster1 = self.__crop_raster_by_polygon_wkt(self.raster1,self.intersection)
        
        # project raster2 to current domain of raster1
        self.raster2 = self.__project_raster_to_existing_raster_domain(self.raster2,self.raster1)
        
        
    def set_resolution(self,xRes,yRes):
        self.raster1 = gdal.Warp('',self.raster1,format='MEM',xRes=xRes,yRes=yRes)
        self.raster2 = gdal.Warp('',self.raster2,format='MEM',xRes=xRes,yRes=yRes)
    
    def set_projection(self,projection):
        source_projection = self.__get_projection(self.raster1)
        self.raster1 = gdal.Warp('', self.raster1, srcSRS=source_projection, dstSRS=projection, format='MEM')
        self.raster2 = gdal.Warp('', self.raster2, srcSRS=source_projection, dstSRS=projection, format='MEM')
    
    def get_raster1_as_array(self,band_number=1):
        return self.raster1.GetRasterBand(band_number).ReadAsArray()
    
    def get_raster2_as_array(self,band_number=1):
        return self.raster2.GetRasterBand(band_number).ReadAsArray()
        
    def export(self,raster1_export_path=None, raster2_export_path=None, mask_export_path=None):
        if not raster1_export_path:
            raster1_export_path = self.__update_path(self.__raster1_path)
        if not raster2_export_path:
            raster2_export_path = self.__update_path(self.__raster2_path)
            
        self.__save_raster_to_gtiff(self.raster1, raster1_export_path)
        self.__save_raster_to_gtiff(self.raster2, raster2_export_path)
        self.__save_mask_to_gtiff(self.raster1, self.raster2, mask_export_path)
    
    '''Service functions'''
        
        
    def __reproject_raster_to_projection(self,raster,dest_projection):
        source_projection = self.__get_projection(raster)
        output_raster = gdal.Warp('', raster, srcSRS=source_projection, dstSRS=dest_projection, format='MEM')
        return output_raster
    
    def __get_projection(self,raster):
        return raster.GetProjection()
    
    def __get_extent(self,raster):
        geoTransform = raster.GetGeoTransform()
        xMin = geoTransform[0]
        yMax = geoTransform[3]
        xMax = xMin + geoTransform[1] * raster.RasterXSize
        yMin = yMax + geoTransform[5] * raster.RasterYSize
        return {'xMax':xMax,'xMin':xMin,'yMax':yMax,'yMin':yMin}
    
    def __extent_to_wkt_polygon(self,extent):
        return 'POLYGON ((%s %s,%s %s,%s %s,%s %s,%s %s))' % (extent['xMin'],extent['yMin'],extent['xMin'],extent['yMax'],
                                                                extent['xMax'],extent['yMax'],extent['xMax'],extent['yMin'],
                                                                extent['xMin'],extent['yMin'])
    
    
    
    def __intersect_two_wkt_polygons(self,polygon_wkt1,polygon_wkt2):
        polygon1 = ogr.CreateGeometryFromWkt(polygon_wkt1)
        polygon2 = ogr.CreateGeometryFromWkt(polygon_wkt2)
        intersection = polygon1.Intersection(polygon2)
        return intersection.ExportToWkt()
    
    def __check_gcp_raster(self,raster):
        if raster.GetGCPCount():
            return True
        else:
            return False
    
    def __gcp_raster_to_projected(self,raster):
        output_raster = gdal.Warp('', raster, format='MEM')
        return output_raster
    
    def __create_memory_ogr_datasource_with_wkt_polygon(self,polygon_wkt):
        drv = ogr.GetDriverByName('MEMORY') 
        source = drv.CreateDataSource('memData') 
        layer = source.CreateLayer('l1',geom_type=ogr.wkbPolygon)
        
        feature_defn = layer.GetLayerDefn()
        feature = ogr.Feature(feature_defn)
        geom = ogr.CreateGeometryFromWkt(polygon_wkt)
        feature.SetGeometry(geom)
        layer.CreateFeature(feature)
        layer.SyncToDisk()
        return source   
    
    def __json_polygon_to_extent(self, polygon_json):
        x_list = []
        y_list = []
        for pair in json.loads(polygon_json)['coordinates']:
            x_list.append(pair[0])
            y_list.append(pair[1])
        return {'xMax':max(x_list),'xMin':min(x_list),'yMax':max(y_list),'yMin':min(y_list)}
    
    def __crop_raster_by_polygon_wkt(self,raster,polygon_wkt):
        geom = ogr.CreateGeometryFromWkt(polygon_wkt)
        extent_json = geom.GetBoundary().ExportToJson()
        extent = self.__json_polygon_to_extent(extent_json)
        output_raster = gdal.Warp('', raster, outputBounds = [extent['xMin'],extent['yMin'],extent['xMax'],extent['yMax']], format = 'MEM')
        return output_raster
    
    def __project_raster_to_existing_raster_domain(self,raster,domain):
        extent = self.__get_extent(domain)
        xSize = domain.RasterXSize
        ySize = domain.RasterYSize
        output_raster = gdal.Warp('',raster,outputBounds = [extent['xMin'],extent['yMin'],extent['xMax'],extent['yMax']],width=xSize, height=ySize, format='MEM')
        return output_raster
            
    def __update_path (self,path):
         folder = os.path.dirname(path)
         filename = os.path.basename(path)
         filename_raw = filename.split('.')[0]
         extension = filename.split('.')[1]
         return os.path.join(folder,filename_raw+'_adjusted.'+extension)
    
    def __save_raster_to_gtiff(self,raster,gtiff_path):
        driver = gdal.GetDriverByName("GTiff")
        dataType = raster.GetRasterBand(1).DataType
        dataset = driver.Create(gtiff_path, raster.RasterXSize, raster.RasterYSize, raster.RasterCount, dataType)
        dataset.SetProjection(raster.GetProjection())
        dataset.SetGeoTransform(raster.GetGeoTransform())
        i = 1
        while i<= raster.RasterCount:
            data = raster.GetRasterBand(i).ReadAsArray()
            data[data == 0] = np.nan
            # print('\nNumber of 0: %s\n' % len(data[data == 0]))
            dataset.GetRasterBand(i).WriteArray(data)
            dataset.GetRasterBand(i).SetDescription(raster.GetRasterBand(i).GetDescription())
            i+=1
        del dataset

    def __get_gdal_dataset_extent(self, gdal_dataset):
        x_size = gdal_dataset.RasterXSize
        y_size = gdal_dataset.RasterYSize
        geotransform = gdal_dataset.GetGeoTransform()
        x_min = geotransform[0]
        y_max = geotransform[3]
        x_max = x_min + x_size * geotransform[1]
        y_min = y_max + y_size * geotransform[5]
        return {'xMin': x_min, 'xMax': x_max, 'yMin': y_min, 'yMax': y_max, 'xRes': geotransform[1],
                'yRes': geotransform[5]}

    def __get_land_mask(self, ds_tiff):
        ''' Rasterized land mask
        based on OSM data
        '''

        source_geotransform = ds_tiff.GetGeoTransform()
        source_projection = ds_tiff.GetProjection()
        source_extent = self.__get_gdal_dataset_extent(ds_tiff)

        # geotransform = gdal_dataset.GetGeoTransform()
        if source_geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            # gdal_dataset = gdal.Warp('',gdal_dataset, format='MEM')
            # print gdal_dataset.RasterXSize
            # geotransform = gdal_dataset.GetGeoTransform()
            # if geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            print('Error: GDAL dataset without georeferencing')

        print('Calculating land mask')
        print('Recalculate raster to WGS84')
        ds_tiff = gdal.Warp('', ds_tiff, dstSRS='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                            format='MEM')
        print('Extracting WGS84 extent')
        extent = self.__get_gdal_dataset_extent(ds_tiff)

        # format='MEM',

        # Check if coast data exist in data directory
        if os.path.isfile(path_to_coastline):
            print('Clipping and Rasterizing land mask to raster extent')
            land_mask_wgs84 = gdal.Rasterize('', path_to_coastline,
                                             format='MEM',
                                             outputBounds=[extent['xMin'], extent['yMin'],
                                                           extent['xMax'], extent['yMax']],
                                             xRes=extent['xRes'], yRes=extent['yRes'])
            # format='MEM',
            land_mask = gdal.Warp('', land_mask_wgs84, format='MEM', dstSRS=source_projection,
                                  xRes=source_extent['xRes'], yRes=source_extent['yRes'],
                                  outputBounds=[source_extent['xMin'], source_extent['yMin'],
                                                source_extent['xMax'], source_extent['yMax']])

            land_data = land_mask.GetRasterBand(1).ReadAsArray()

            del land_mask
            return land_data
        else:
            print('\nCould not find land data in %s!\n' % path_to_coastline)

    def __save_mask_to_gtiff(self, raster1, raster2, gtiff_path):
        driver = gdal.GetDriverByName("GTiff")
        dataType = gdal.GDT_Byte

        dataset = driver.Create(gtiff_path, raster1.RasterXSize, raster1.RasterYSize, raster1.RasterCount, dataType)
        dataset.SetProjection(raster1.GetProjection())
        dataset.SetGeoTransform(raster1.GetGeoTransform())

        i = 1
        arr1 = raster1.GetRasterBand(i).ReadAsArray()
        arr2 = raster2.GetRasterBand(i).ReadAsArray()

        mask_array = np.copy(arr1)
        mask_array[:, :] = 255
        mask_array[np.isnan(arr1)] = 0
        mask_array[np.isnan(arr2)] = 0

        #print(mask_array)

        ##########################
        # Create land mask
        ##########################
        print('\nApplying land mask...')
        # Get land mask
        print(raster1)
        land_mask = self.__get_land_mask(raster1)

        # Apply land mask
        print('###################')
        mask_array[land_mask == 255] = 0
        print('Done.\n')
        ##########################
        # End create lamd mask
        ##########################

        while i <= raster1.RasterCount:
            dataset.GetRasterBand(i).WriteArray(mask_array)
            i += 1
        del dataset
