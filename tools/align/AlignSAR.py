#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script gives functionality to align two geotiff images given a velocity vector field.

2021-09-02 Anders Gunnar Felix Hildeman
2022-12-19 Denis Demchev

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
sys.path.append(r'../geolocation_grid')
from LocationMapping import LocationMapping
import time

class Alignment:
	"""Class for a pair of images alignment by displecements compensation """

	def __init__(self, img1_path=None, img2_path=None, displacement_path=None,
				 transform_type="piecewise-affine", out_path=r'./',
				 transform_master=False, geocoded=False, proj=None):

		self.transform_type = transform_type
		self.img1_path = img1_path
		self.img2_path = img2_path
		self.displacement_path = displacement_path
		self.out_path = out_path
		self.transform_master = transform_master
		self.geocoded = geocoded
		self.proj = proj

		print(f'Alignment method: {transform_type}')
		os.makedirs(self.out_path, exist_ok=True)

		if not (self.img1_path and self.img2_path and self.displacement_path) is None:
			self.performAlignment(padding='constant', back_alignment=False)
		else:
			raise TypeError(
				"Please specify all neccesseaury path to data (Image 1, Image 2 and a path to displacement data)")

	def performAlignment(self, padding, back_alignment):
		"""
		Function for aligning two images given a deformation field and then save it to output path.
		"""

		# Open first image and acquire raster as array
		image1 = gdal.Open(str(self.img1_path))

		# Open second  image and acquire raster as array
		image2 = gdal.Open(str(self.img1_path))

		# Get deformation data as original pixel locations in image1 and new pixel locations in image1
		with open(self.displacement_path) as csvfile:
			csvreader = csv.reader(csvfile)
			displacements = np.array([row for row in csvreader]).astype(float)

			# If coordinates are geocoded then covert them to image coordinates
			if self.geocoded == True:
				lm = LocationMapping(image1.GetGeoTransform(), image1.GetProjection())
				print('\nConvert geocoded cooridnates to raster coordinates...')
				displacements = displacements[~np.any(np.isnan(displacements), axis=1), :]
				orig_locs = displacements[:, [0, 1]]
				new_locs = displacements[:, [2, 3]]
				c0, r0 = lm.latLon2Raster( orig_locs[:,1].reshape( (-1) ), orig_locs[:,0].reshape( (-1) ) )
				c1, r1 = lm.latLon2Raster( new_locs[:,1].reshape( (-1) ), new_locs[:,0].reshape( (-1) ) )

				orig_locs = np.stack((c0, r0)).T
				new_locs = np.stack((c1, r1)).T

				print(orig_locs)
				print(new_locs)

				print('Done.\n')
			else:
				# Select only not NaN data for displacements
				displacements = displacements[~np.any(np.isnan(displacements), axis=1), :]
				orig_locs = displacements[:, [1, 0]]
				new_locs = displacements[:, [3, 2]] + orig_locs

		# Acquire transformation given by deformation
		if self.transform_type == "piecewise-affine":
			print(f'\nEstimating {self.transform_type} transformation...')
			p = PiecewiseAffineTransform()
			p.estimate(src=orig_locs, dst=new_locs)
			self.p = p
			print(f'Done.')
		else:
			raise Exception(f'Sorry, {self.transform_type} transform is not implemented in the current version')

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
			print(f'\nWarping {self.img1_path}')
			start_time = time.time()

			if self.warping(self.img1_path, image1, self.p.inverse, padding):
				exec_time = (time.time() - start_time) / 60.
				print("Done in {:0.1f} minutes\n".format(exec_time))
				return 1

			''' TODO:
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
				pass
			'''
		except Exception as e:
			print(str(e))
			return 1

		# Transform image2 raster array
		if self.transform_master == True:
			try:
				print(f'\nWarping {self.img2_path}')
				start_time = time.time()
				if self.warping(self.img2_path, image2, self.p, padding):
					exec_time = (time.time() - start_time) / 60.
					print("Done in {:0.1f} minutes\n".format(exec_time))
					return 1

			except Exception as e:
				print(str(e))
				return 1
		else:
			pass

		return 0

	def warping(self, path, image, trans, padding):
		''' Perform warping of a pair of GeoTIFF files based on transformation '''

		# Create new geotiff file
		new_path = f'{self.out_path}/Aligned_{os.path.basename(path)}'

		GTdriver = gdal.GetDriverByName('GTiff')
		opts = ["COMPRESS=LZW", "BIGTIFF=YES"]
		out = GTdriver.Create(new_path, image.RasterXSize, image.RasterYSize, image.RasterCount, gdal.GDT_Float32,
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