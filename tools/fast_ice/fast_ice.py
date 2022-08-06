import matplotlib.pyplot as plt
from skimage import measure
from osgeo import gdal
import numpy as np
import os
import sys

sys.path.append("/data/rrs/seaice/esa_rosel/code/ice_drift_pc_ncc/tools/geolocation_grid")
from LocationMapping import LocationMapping
import geojson


class get_fast_ice:
	'''
	Class for fast ice deliniation from ice drift data
	'''

	def __init__(self, npz_file_path=None, tiff_file_path=None, stp=50, th_pixels=5):
		self.npz_file_path = npz_file_path
		self.tiff_file_path = tiff_file_path
		self.stp = stp
		self.th_pixels = th_pixels

		npz = np.load(npz_file_path)
		self.ice_drift_magnitude = np.hypot(npz['u'], npz['v'])

		self.image = gdal.Open(self.tiff_file_path)
		lm = LocationMapping(self.image.GetGeoTransform(), self.image.GetProjection())

		# Get lon and lat values of all pixels
		X = np.arange(self.image.RasterXSize)
		Y = np.arange(self.image.RasterYSize)
		X, Y = np.meshgrid(X, Y)
		lat, lon = lm.raster2LatLon(X.reshape((-1)), Y.reshape((-1)))
		self.lats = lat.reshape(self.image.ReadAsArray()[0].shape[0], self.image.ReadAsArray()[0].shape[1])
		self.lons = lon.reshape(self.image.ReadAsArray()[0].shape[0], self.image.ReadAsArray()[0].shape[1])

		# Separate stable (fast) ice (1) and drift ice (0) into two classes based on the threshold
		self.ice_drift_magnitude[self.ice_drift_magnitude <= self.th_pixels] = 1
		self.ice_drift_magnitude[self.ice_drift_magnitude > self.th_pixels] = 0
		self.ice_drift_magnitude[np.isnan(self.ice_drift_magnitude)] = 0

		self.data = self.ice_drift_magnitude

		# Find contours at a constant value of n pixels
		print('\nFinding contours...')
		self.contours = measure.find_contours(self.data, 0)
		print('Done')

	def plot_contours(self):
		'''
		Plot obtained contours
		'''
		# Display the image and plot all contours found
		fig, ax = plt.subplots()
		ax.imshow(self.data, cmap='jet')

		for contour in self.contours:
			ax.plot(contour[:, 1], contour[:, 0], linewidth=5)

	def export_json(self, out_fname):
		'''
		Export result to Geojson file
		'''

		f_dir = os.path.dirname(out_fname)
		os.makedirs(f_dir, exist_ok=True)

		for ch, contour in enumerate(self.contours):
			b_lons = []
			b_lats = []

			for i in range(len(contour)):
				b_lons.append(self.lons[int(contour[i][0] * self.stp + self.stp),
										int(contour[i][1] * self.stp + self.stp)])
				b_lats.append(self.lats[int(contour[i][0] * self.stp + self.stp),
										int(contour[i][1] * self.stp + self.stp)])

			# Define Polygon string
			poly_str = [[x for x in zip(*[iter(b_lons), iter(b_lats)])]]
			features = geojson.Polygon(poly_str)

			# feature is a shapely geometry type
			geom_in_geojson = geojson.Feature(geometry=features, properties={})

			out_fname = f'{f_dir}/{ch}_fast_ice.geojson'
			with open(out_fname, 'w') as outfile:
				geojson.dump(geom_in_geojson, outfile)


