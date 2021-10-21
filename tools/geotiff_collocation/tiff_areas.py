
import sys
import numpy as np
import os
try:
	import gdal
except:
	from osgeo import gdal

def get_area_of_array( image_array, resolution ):
	nans = (image_array == 0) | np.isnan(image_array)
	return np.sum(~nans) * (resolution**2)


def get_area_of_images( path_img1, resolution, path_img2 = None ):

    # Open first image
    img1 = gdal.Open( path_img1 )
    img1_raster = img1.GetRasterBand(1)	
    img1_array = img1.ReadAsArray()	
    img1_active_area = get_area_of_array( img1_array, resolution )
    
    # If second image exists
    img2_active_area = None
    img12_active_area = None
    if path_img2 is not None:
        # Open image 2
        img2 = gdal.Open( path_img2 )
        img2_raster = img2.GetRasterBand(1)	
        img2_active_area = get_area_of_array( img2_raster.ReadAsArray(), resolution )
        
        # Compute overlap between images
        img12_active_area = get_area_of_array( \
            (img1_raster.ReadAsArray() != 0) & ~np.isnan(img1_raster.ReadAsArray()) & \
            (img2_raster.ReadAsArray() != 0) & ~np.isnan(img2_raster.ReadAsArray()), \
            resolution )
            
    return ( img1_active_area, img2_active_area, img12_active_area )



# If run as script                                                                                                                                                                                                 
if __name__ == "__main__":
	"""
	Param 1: Path to image 1
	Param 2: Resolution [m/pixel]
    Param 3: Path to second image
	"""  

	if len(sys.argv) < 2:
		raise Exception("No image path given!")
	
	# Path to first image                                                                                                                                                                                        
	path_img1 = sys.argv[1] 
	if not os.path.isfile(path_img1):
		raise Exception("Image path did not correspond to an existing file!")
	
	# resolution [meters per pixel]
	resolution = float(100)
	if len(sys.argv) >= 3:
		resolution = float(sys.argv[2])
	
	# Path to second image                                                                                                                                                                                        
	path_img2 = None
	if len(sys.argv) >= 4:
		path_img2 = sys.argv[3] 
		if not os.path.isfile(path_img2):
			path_img2 = None
	
    # Get areas	
	img1_active_area, img2_active_area, img12_active_area = get_area_of_images( path_img1, resolution, path_img2 )
    
	print( "Area of image 1: %s [km^2]" % str(img1_active_area * 1e-6) )
	# If second image exists
	if img2_active_area is not None:
		print( "Area of image 2: %s [km^2]" % str(img2_active_area * 1e-6) )
		print( "Area overlapping between image 1 and 2: %s [km^2]" % str(img12_active_area * 1e-6) )







