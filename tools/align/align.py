"""
This script gives functionality to align two geotiff images given a velocity vector field.

2021-09-2 Anders Gunnar Felix Hildeman

"""


import pathlib
from skimage import transform
try:
	from osgeo import gdal
except:
	import gdal, osr
import numpy as np
import csv
import sys
import LocationMapping
# import glob
# import os
# from matplotlib.path import Path
# from scipy.spatial import ConvexHull









def performAlignment( path1, path2, deformation_path, output_path, transform_type = "piecewise-affine", padding = "constant", polynomial_order = 2 ):
	"""
	Function for aligning two images given a deformation field and then save it to output path.
	"""

	# Open first image and acquire raster as array
	image1 = gdal.Open(str(path1))
	geotransform1 = image1.GetGeoTransform()
	proj1 = image1.GetProjection()
	cols1 = image1.RasterXSize
	rows1 = image1.RasterYSize
	num_bands1 = image1.RasterCount
	location_mapping1 = LocationMapping( geotransform1, proj1 )
	
	# Open second  image and acquire raster as array
	image2 = gdal.Open(str(path2))
	geotransform2 = image2.GetGeoTransform()
	proj2 = image2.GetProjection()
	cols2 = image2.RasterXSize
	rows2 = image2.RasterYSize
	num_bands2 = image2.RasterCount
	location_mapping2 = LocationMapping( geotransform2, proj2 )
	
	# Get deformation data as original pixel locations in image1 and new pixel locations in image1
	try: 
		with open( deformation_path ) as csvfile:
			csvreader = csv.reader(csvfile)
			deformations = np.array( [ row for row in csvreader ] ).astype(float)

	except:
		return 1
	deformations = deformations[ ~np.any( np.isnan(deformations), axis=1 ), : ]
	orig_locs = deformations[:,[1,0]]
	new_locs = deformations[:,[3,2]] + orig_locs
	
	# Acquire transformation given by deformation
	try:
		print("Computing transformation.")
		trans = None
		invTrans = None
		if transform_type == "polynomial":
			trans = transform.estimate_transform( transform_type, src = orig_locs, dst = new_locs, order = polynomial_order )
			invTrans = transform.estimate_transform( transform_type, dst = orig_locs, src = new_locs, order = polynomial_order )
		else:
			trans = transform.estimate_transform( transform_type, src = orig_locs, dst = new_locs )
			invTrans = trans.inverse

	except:
		return 1

	# # Acquire convex hulls 
	# print("Acquiring convex hulls.")
	# maskOrig = None
	# maskNew = None
	# try:
	# 	# Get convex hull of image 1
	# 	ch = ConvexHull( new_locs )
	# 	y, x = np.meshgrid( np.arange(rows1), np.arange(cols1) )
	# 	y, x = np.transpose(y), np.transpose(x)
	# 	x, y = x.flatten(), y.flatten()
	# 	points = np.stack( (x,y), axis=1 )
	# 	p = Path( new_locs[ch.vertices, :] )
	# 	grid = p.contains_points( points )
	# 	maskNew = grid.reshape(array2.shape)

	# 	# Get convex hull of image 2
	# 	ch = ConvexHull( orig_locs )
	# 	y, x = np.meshgrid( np.arange(rows1), np.arange(cols1) )
	# 	y, x = np.transpose(y), np.transpose(x)
	# 	x, y = x.flatten(), y.flatten()
	# 	points = np.stack( (x,y), axis=1 )
	# 	p = Path( new_locs[ch.vertices, :] )
	# 	grid = p.contains_points( points )
	# 	maskOrig = grid.reshape(array2.shape)

	# 	del ch, x, y, points, p, grid
	# except:
	# 	return 1

	
	# Transform image1 raster array
	try:
		print("Warping " + str(path1.name))
        
        # Create new geotiff file
		new_path1 = output_path / ("Aligned_" + str(path1.name))
		GTdriver = gdal.GetDriverByName('GTiff')
		opts = [ "COMPRESS=LZW", "BIGTIFF=YES" ]
		out1 = GTdriver.Create( str(new_path1), cols1, rows1, num_bands1, gdal.GDT_Float32, opts )
		out1.SetGeoTransform(geotransform1)
		out1.SetProjection(proj1)
        
		for iterBands in range( num_bands1 ):
			band1 = image1.GetRasterBand(int(iterBands+1))
			array1 = band1.ReadAsArray()
			array1[array1 == 0] = np.nan        
			array_warped = transform.warp( array1, invTrans, mode = padding, cval = np.nan )
			array_warped[array_warped == 0] = np.nan
            # array_warped[~maskOrig] = np.nan
		
    		# Write alignment to new file
			band_out = out1.GetRasterBand(int(iterBands+1))
			band_out.WriteArray(array_warped)
			out1.FlushCache()
		del out1

	except Exception as e:
		print(str(e))
		return 1
	
	# Transform image2 raster array
	try:
		print("Warping " + str(path2.name))
        
        # Create new geotiff file
		new_path2 = output_path / ("Aligned_" + str(path2.name))
		GTdriver = gdal.GetDriverByName('GTiff')
		opts = [ "COMPRESS=LZW", "BIGTIFF=YES" ]
		out2 = GTdriver.Create( str(new_path2), cols2, rows2, num_bands2, gdal.GDT_Float32, opts )
		out2.SetGeoTransform(geotransform2)
		out2.SetProjection(proj2)
        
		for iterBands in range( num_bands2 ):
		    band2 = image2.GetRasterBand( int(iterBands+1) )
		    array2 = band2.ReadAsArray()
		    array2[array2 == 0] = np.nan        
		    array_warped = transform.warp( array2, trans, mode = padding, cval = np.nan )
		    array_warped[array_warped == 0] = np.nan
    		# array_warped[~maskNew] = np.nan
        
    		# Write alignment to new file
		    band_out = out2.GetRasterBand( int( iterBands+1 ) )
		    band_out.WriteArray(array_warped)
		    out2.FlushCache()
		del out2

	except Exception as e:
		print(str(e))
		return 1


	return 0



def handlePair( image_path, deformation_path, output_path, transform_type, poly_order  ):
	"""
	Param 1: Path to directory to image pair.
	Param 2: Path to main directory of drift results for specific pair.
	Param 3: Path to place output files.
	"""

	# Get path to image pair
	image_path = sorted( list(image_path.glob('UPS_*.tiff' )), key = lambda x: x.name.split('_')[6] )
	if len(image_path) != 2:
		raise Exception("Did not exist exactly 2 UPS_hh files!")
	path1 = image_path[0]
	print(path1)
	path2 = image_path[1]
	print(path2)

	# Get path to deformation
	deformation_path = list(deformation_path.glob( 'output/*.csv' ))
	if len(deformation_path ) != 1:
		print( "Did not find .csv file for %s" % str(deformation_path)  )
		return 
	deformation_path = deformation_path[0]
	# Create directory 
	try:
		output_path.mkdir()
	except:
		return
	
	# Align images and save 
	performAlignment( path1, path2, deformation_path, output_path, transform_type = transform_type, padding = "constant", polynomial_order = poly_order )



# If run as script
if __name__ == "__main__":
	"""
	Param 1: Path to parent directory of all collocated image pairs
	Param 2: Path to parent directory of all retrieved drifts
	Param 3: Path to parent directory intended for all outputs
	Param 4: Which transform type to use.
	Param 5: Order of polynomial if polynomial transformation is used.
	"""

	# Get transform type
	transform_type = "piecewise-affine"
	if len(sys.argv) >= 5:
		transform_type = sys.argv[4]
	# Get polynomial order
	poly_order = "3"
	if len(sys.argv) >= 6:
		poly_order = sys.argv[5]
	poly_order = int(poly_order)
	print("Aligning using method: %s" % transform_type )

	# Root image path
	image_path = pathlib.Path( sys.argv[1] ).expanduser().absolute()
	# Root deformation path
	deformation_path = pathlib.Path( sys.argv[2] ).expanduser().absolute()
	# Get path to output images
	output_path = pathlib.Path( sys.argv[3] ).expanduser().absolute()

	# Loop through all directories of root directory
	for iter_path in image_path.iterdir():
		if iter_path.is_dir():
			# Get directory name
			dir_name = iter_path.name
			
			# Find directory with the same name in deformation path
			iter_deform_path = deformation_path.joinpath(dir_name)
			if (iter_deform_path.exists()):
				iter_output_path = output_path.joinpath( dir_name )
				# If subdirectory already exists
				if (iter_output_path.exists()):
					# skip 
					continue
					# # Remove old version
					# for iterremove in iter_output_path.glob('*'):
					# 	if iterremove.is_file():
					# 		iterremove.unlink()
					# iter_output_path.rmdir()
				print( "Running pair #" + str(iter_output_path.name) )

				# Call for alignment
				handlePair( iter_path, iter_deform_path, iter_output_path, transform_type, poly_order )
				print("")







