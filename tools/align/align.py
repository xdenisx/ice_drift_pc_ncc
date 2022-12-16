#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script gives functionality to align two geotiff images given a velocity vector field.

2021-09-2 Anders Gunnar Felix Hildeman

"""


import pathlib
from skimage.transform import PiecewiseAffineTransform, warp

try:
	from osgeo import gdal
except:
	import gdal
import numpy as np
import os
import csv
import sys
import glob
import xml.etree.ElementTree
sys.path.append("../geolocation_grid")
from LocationMapping import LocationMapping

# import glob
# import os
# from matplotlib.path import Path
# from scipy.spatial import ConvexHull


def warping(path, output_path, image, trans, padding):
	# Create new geotiff file
	new_path = output_path / ("Aligned_" + str(path.name))
	GTdriver = gdal.GetDriverByName('GTiff')
	opts = ["COMPRESS=LZW", "BIGTIFF=YES"]
	out = GTdriver.Create(str(new_path), image.RasterXSize, image.RasterYSize, image.RasterCount, gdal.GDT_Float32,
						  opts)
	out.SetGeoTransform(image.GetGeoTransform())
	out.SetProjection(image.GetProjection())
	metadata = image.GetMetadata()

	# Get mapping between coordinates for the given image
	location_mapping = LocationMapping(image.GetGeoTransform(), image.GetProjection())

	# Preallocate output array
	array = np.zeros((image.RasterYSize, image.RasterXSize, image.RasterCount), dtype=float)

	for iterBands in range(image.RasterCount):

		# Get current raster band
		band = image.GetRasterBand(int(iterBands + 1))
		# Get name of raster band
		band_name = band.GetDescription()
		# Get raster array from current band
		array[:, :, iterBands] = band.ReadAsArray()
		# Get corresponding band from out raster
		band_out = out.GetRasterBand(int(iterBands + 1))
		# Set name of band for output
		band_out.SetDescription(band_name)

		# If raster band name exists in metadata, warp the geolocationGridPoints
		if (band_name in metadata.keys()):
			# Get xml from current metadata
			try:
				xml_tree = xml.etree.ElementTree.fromstring(metadata[band_name])
				if (xml_tree.tag == 'geolocationGrid'):
					# Get all grid points xml elements
					gridPoints = list(xml_tree.findall('./geolocationGridPointList/geolocationGridPoint'))
					# Go through all girdPoint elements
					lat = np.zeros((len(gridPoints)))
					long = np.zeros((len(gridPoints)))
					ok_grid_points = np.zeros((len(gridPoints)), dtype=bool)
					for iterGridPoints, gridPoint in enumerate(gridPoints):
						# Insert current latitude and longitude value
						cur_lat = gridPoint.find('./latitude')
						cur_long = gridPoint.find('./longitude')
						# If the latitude and longitude value actually existed
						if cur_lat is not None and cur_long is not None:
							# Insert current values in arrays
							lat[iterGridPoints] = float(cur_lat.text)
							long[iterGridPoints] = float(cur_long.text)
							ok_grid_points[iterGridPoints] = True

					# Map geolocation points
					x, y = location_mapping.latLon2Raster(np.array(lat), np.array(long))
					points = np.stack((x, y), axis=1)
					points = trans(points)
					ok_grid_points = ok_grid_points & np.all(points >= 0, axis=1)
					lat, long = location_mapping.raster2LatLon(points[:, 0], points[:, 1])

					# Go through all grid points again
					for iterGridPoints, gridPoint in enumerate(gridPoints):
						# If current grid point is marked as 'ok'
						if ok_grid_points[iterGridPoints]:
							gridPoint.find('./latitude').text = str(lat[iterGridPoints])
							gridPoint.find('./longitude').text = str(long[iterGridPoints])
						else:
							# Otherwise, remove
							xml_tree.find('./geolocationGridPointList').remove(gridPoint)

					# Write new metadata
					metadata[band_name] = xml.etree.ElementTree.tostring(xml_tree)


			except Exception as e:
				print(e)
				return 1

	array[array == 0] = np.nan

	array_warped = warp(array, trans, mode='constant')  # , mode = padding, cval = np.nan )

	array_warped[array_warped == 0] = np.nan
	# array_warped[~maskOrig] = np.nan

	for iterBands in range(image.RasterCount):
		# Get corresponding band from out raster
		band_out = out.GetRasterBand(int(iterBands + 1))
		# Write aligned array to output
		band_out.WriteArray(array_warped[:, :, iterBands])

	out.SetMetadata(metadata)
	out.FlushCache()
	del out

	return 0

def performAlignment( path1, path2, deformation_path, output_path, transform_type = "piecewise-affine", padding = "constant", polynomial_order = 2, back_alignment = True ):
	"""
	Function for aligning two images given a deformation field and then save it to output path.
	"""

	# Open first image and acquire raster as array
	image1 = gdal.Open(str(path1))

	# Open second  image and acquire raster as array
	image2 = gdal.Open(str(path2))

	# Get deformation data as original pixel locations in image1 and new pixel locations in image1
	#try:
	with open( deformation_path ) as csvfile:
		csvreader = csv.reader(csvfile)
		deformations = np.array( [ row for row in csvreader ] ).astype(float)
	#except:
#		return 1
	deformations = deformations[ ~np.any( np.isnan(deformations), axis=1 ), : ]
	orig_locs = deformations[:,[1,0]]
	new_locs = deformations[:,[3,2]] + orig_locs

	# Acquire transformation given by deformation
	if transform_type == "piecewise-affine":
		print(f'\nComputing {transform_type} transformation...')
		p = PiecewiseAffineTransform()
		p.estimate(src=orig_locs, dst=new_locs)
		print(f'Done.\n')
	else:
		raise Exception(f'Sorry, {transform_type} transform is not implemented in the current version')

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

		if warping(path1, output_path, image1, p.inverse, padding):
			return 1

		if back_alignment == True:
			print('\nStart back-alignment...')
			aligned_images = glob.glob(f'{output_path}/Aligned*.tif*')
			aligned_images.sort(key=lambda x: os.path.basename(x).split('_')[7])
			print(f'Aligned image: {aligned_images[0]}')
			image_aligned_back = gdal.Open(aligned_images[0])
			output_ba_path = output_path / 'back_alignment'
			os.makedirs(output_ba_path, exist_ok=True)
			warping(path1, output_ba_path, image_aligned_back, p, padding)
			print('Done.')
		else:
			print('No back alignment')

	except Exception as e:
		print(str(e))
		return 1

	# Transform image2 raster array
	try:
		print("Warping " + str(path2.name))

		if warping(path2, output_path, image2, p, padding):
			return 1

	except Exception as e:
		print(str(e))
		return 1

	return 0



def handlePair( image_path, deformation_path, output_path, transform_type, poly_order, back_alignment=True ):
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

	os.makedirs(output_path, exist_ok=True)

	# Align images and save
	performAlignment( path1, path2, deformation_path, output_path, transform_type = transform_type, padding = "constant", polynomial_order = poly_order, back_alignment = True )


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
			#print(iter_deform_path)

			if (iter_deform_path.exists()):
				iter_output_path = output_path.joinpath( dir_name )
				# If subdirectory already exists
				if (iter_output_path.exists()):
					pass
					# skip
					#continue
					# # Remove old version
					# for iterremove in iter_output_path.glob('*'):
					# 	if iterremove.is_file():
					# 		iterremove.unlink()
					# iter_output_path.rmdir()
				print( "Running pair #" + str(iter_output_path.name) )

				# Call for alignment
				handlePair( iter_path, iter_deform_path, iter_output_path, transform_type, poly_order, back_alignment=True )
				print("")







