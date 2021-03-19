# -*- coding: utf-8 -*-
"""
RasterAdjuster class
Prerequisites: gdal>=2.1

Performs adjusting of two rasters (extent, resolution)

Author: Eduard Kazakov (silenteddie@gmail.com)
Last modification: 2017-08-03
"""

import gdal, ogr
import json, os

class RasterAdjuster():
    
    def __init__(self,raster1_path,raster2_path):
        
        # Store input pathes
        self.__raster1_path = raster1_path
        self.__raster2_path = raster2_path
        
        # Read datasets
        self.raster1 = gdal.Open(raster1_path)
        self.raster2 = gdal.Open(raster2_path)
        
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
        
    def export(self,raster1_export_path=None, raster2_export_path=None):
        if not raster1_export_path:
            raster1_export_path = self.__update_path(self.__raster1_path)
        if not raster2_export_path:
            raster2_export_path = self.__update_path(self.__raster2_path)
            
        self.__save_raster_to_gtiff(self.raster1,raster1_export_path)
        self.__save_raster_to_gtiff(self.raster2,raster2_export_path)
    
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
            dataset.GetRasterBand(i).WriteArray(raster.GetRasterBand(i).ReadAsArray())
            i+=1
        del dataset    
