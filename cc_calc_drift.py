import numpy as np
from skimage.feature import match_template
from osgeo import gdal, osr, gdal_array

class CalcDrift(object):
	def __init__(self, Conf, img1, img2):
		self.Conf = Conf
		self.img1 = img1
		self.img2 = img2

	def create_arguments(self, height, width):
		block_size = self.Conf.block_size
		search_area = self.Conf.search_area
		grid_step = self.Conf.grid_step
		f1_name = self.Conf.f1_name

		# Arrays for lines_1, rows_1, lines_2, rows_2, corr. coefficients
		# Get size of arrays first
		num_lines = len(range(block_size, height + block_size, grid_step))
		num_rows = len(range(block_size, width + block_size, grid_step))

		arr_lines_1 = np.empty([num_lines, num_rows])
		arr_lines_1[:] = np.nan
		arr_lines_2 = np.empty([num_lines, num_rows])
		arr_lines_2[:] = np.nan

		arr_rows_1 = np.empty([num_lines, num_rows])
		arr_rows_1[:] = np.nan
		arr_rows_2 = np.empty([num_lines, num_rows])
		arr_rows_2[:] = np.nan

		arr_cc_max = np.empty([num_lines, num_rows])
		arr_cc_max[:] = np.nan

		# Lists for parallel caclulations
		ll_line_0 = []
		ll_row_0 = []

		# First image bounds
		ll_Lt0 = []
		ll_Rt0 = []
		ll_Lt1 = []
		ll_Rt1 = []

		# Second image bounds
		ll_Li0 = []
		ll_Ri0 = []
		ll_Li1 = []
		ll_Ri1 = []

		ll_im1_name = []
		ll_im2_name = []

		# Create 2D matrices for data
		x_2d = np.empty((len(range(grid_step, height - block_size, grid_step)),
						len(range(grid_step, width - block_size, grid_step),)))
		x_2d[:] = np.nan

		y_2d = np.empty((len(range(grid_step, height - block_size, grid_step)),
						len(range(grid_step, width - block_size, grid_step),)))
		y_2d[:] = np.nan

		# Lats and lons
		lon_2d = np.empty((len(range(grid_step, height - block_size, grid_step)),
						len(range(grid_step, width - block_size, grid_step),)))
		lon_2d[:] = np.nan

		lat_2d = np.empty((len(range(grid_step, height - block_size, grid_step)),
						len(range(grid_step, width - block_size, grid_step),)))
		lat_2d[:] = np.nan

		# dX and dY
		u_2d = np.empty((len(range(grid_step, height - block_size, grid_step)),
						len(range(grid_step, width - block_size, grid_step),)))
		u_2d[:] = np.nan

		v_2d = np.empty((len(range(grid_step, height - block_size, grid_step)),
						len(range(grid_step, width - block_size, grid_step),)))
		v_2d[:] = np.nan

		# Median vectors
		u_2d_med = np.empty((len(range(grid_step, height - block_size, grid_step)),
						len(range(grid_step, width - block_size, grid_step),)))
		u_2d_med[:] = np.nan

		v_2d_med = np.empty((len(range(grid_step, height - block_size, grid_step)),
						len(range(grid_step, width - block_size, grid_step),)))
		v_2d_med[:] = np.nan

		# Prepare geotransform object for data
		# Get 1st geotiff file
		gd_raster = gdal.Open(f1_name)
		geotransform = gd_raster.GetGeoTransform()
		old_cs = osr.SpatialReference()
		old_cs.ImportFromWkt(gd_raster.GetProjection())
		new_cs = osr.SpatialReference()
		new_cs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
		transform = osr.CoordinateTransformation(old_cs, new_cs)
		pixelWidth = geotransform[1]
		pixelHeight = geotransform[-1]

		# 2D indexes for all elements
		idxs_2d = []

		for i, idx_line in enumerate(range(grid_step, height - block_size, grid_step)):
			for j, idx_row in enumerate(range(grid_step, width - block_size, grid_step)):
				#print('\ndr_idx_line, dr_idx_row: (%s, %s)' % (idx_line, idx_row))

				x_2d[i, j] = idx_line
				y_2d[i, j] = idx_row

				# Convert x and y to lon lat
				xx1 = geotransform[0] + idx_line * pixelWidth
				yy1 = geotransform[3] + idx_row * pixelHeight

				try:
					latlon = transform.TransformPoint(float(xx1), float(yy1))
					ilon = latlon[0]
					ilat = latlon[1]
					lon_2d[i, j] = ilon
					lat_2d[i, j] = ilat
				# Non geo data
				except:
					lon_2d[i, j] = 0.
					lat_2d[i, j] = 79.

				Lt0 = np.max([0, idx_line - grid_step])
				Lt1 = np.max([0, idx_row - grid_step])
				Rt0 = np.min([idx_line + grid_step, height])
				Rt1 = np.min([idx_row + grid_step, width])

				# Crop patch from img1
				#im1 = img1[Lt0:Rt0, Lt1:Rt1]

				Li0 = np.max([0, idx_line - grid_step * search_area])
				Li1 = np.max([0, idx_row - grid_step * search_area])
				Ri0 = np.min([idx_line + grid_step * search_area, height])
				Ri1 = np.min([idx_row + grid_step * search_area, width])

				# Crop search area from img2
				#im2 = img2[Li0:Ri0, Li1:Ri1]

				# Save to image pathes
				#f1_name = 'test_images/01_%s_%s.png' % (idx_line, idx_row)
				#f2_name = 'test_images/02_%s_%s.png' % (idx_line, idx_row)

				try:
					#io.imsave(f1_name, im1)
					#io.imsave(f2_name, im2)

					ll_line_0.append(idx_line)
					ll_row_0.append(idx_row)

					# First image bounds
					ll_Lt0.append(Lt0)
					ll_Rt0.append(Rt0)
					ll_Lt1.append(Lt1)
					ll_Rt1.append(Rt1)

					# Second image bounds
					ll_Li0.append(Li0)
					ll_Ri0.append(Ri0)
					ll_Li1.append(Li1)
					ll_Ri1.append(Ri1)

					idxs_2d.append((i,j))

					#ll_im1_name.append(f1_name)
					#ll_im2_name.append(f2_name)
				except:
					pass

		self.geotransform = geotransform
		self.pixelWidth = pixelWidth
		self.pixelHeight = pixelHeight

		self.ll_line_0 = ll_line_0
		self.ll_row_0 = ll_row_0

		self.ll_Lt0 = ll_Lt0
		self.ll_Rt0 = ll_Rt0
		self.ll_Lt1 = ll_Lt1
		self.ll_Rt1 = ll_Rt1

		self.ll_Li0 = ll_Li0
		self.ll_Ri0 = ll_Ri0
		self.ll_Li1 = ll_Li1
		self.ll_Ri1 = ll_Ri1

		self.idxs_2d = idxs_2d
		self.u_2d = u_2d
		self.v_2d = v_2d
		self.u_2d_med = u_2d_med
		self.v_2d_med = v_2d_med

		self.Count = len(ll_line_0)
		self.peak_pref = ['dr'] * self.Count

	def calculate_drift(self, itr):
		iidx_line = self.ll_line_0[itr]
		iidx_row = self.ll_row_0[itr]

		Lt0 = self.ll_Lt0[itr]
		Rt0 = self.ll_Rt0[itr]
		Lt1 = self.ll_Lt1[itr]
		Rt1 = self.ll_Rt1[itr]

		Li0 = self.ll_Li0[itr]
		Ri0 = self.ll_Ri0[itr]
		Li1 = self.ll_Li1[itr]
		Ri1 = self.ll_Ri1[itr]

		pref = self.peak_pref[itr]

		img_median_filtering = self.Conf.img_median_filtering
		img_laplace_filtering = self.Conf.img_laplace_filtering
		img_gradient_filtering = self.Conf.img_gradient_filtering
		img_scharr_filtering = self.Conf.img_scharr_filtering
		plot_correlation_peaks = self.Conf.plot_correlation_peaks
		median_kernel = self.Conf.median_kernel

		if iidx_line is not None:
			# Open two images
			im1 = self.img1[Lt0:Rt0, Lt1:Rt1]
			im2 = self.img2[Li0:Ri0, Li1:Ri1]

			# Filtering
			if img_median_filtering:
				im1 = median(im1, disk(median_kernel))
				im2 = median(im2, disk(median_kernel))

			if img_laplace_filtering:
				im1 = laplace(im1)
				im2 = laplace(im2)

			if img_gradient_filtering:
				im1 = gradient(im1, disk(3))
				im2 = gradient(im2, disk(3))

			if img_scharr_filtering:
				im1 = filters.scharr(im1)
				im2 = filters.scharr(im2)

			# Check for black stripes
			flag1 = self.check_borders(im1)
			flag2 = self.check_borders(im2)

			# No black borders in the first image
			if flag1 == 0: # and flag2 == 0:
				u_direct, v_direct, result = self.matching(im1, im2)
				# Peak maximum CC
				cc_max = np.max(result)

				# Get coordinates with offsets
				lline_2, rrow_2 = v_direct + Li0, u_direct +  Li1
				lline_1, rrow_1 = iidx_line, iidx_row

				#ff_out_txt.write('%s, %s, %s, %s, %s, %s, %s, %s' %
				#                 (lline_1, rrow_1, lline_2, rrow_2, u_direct, Li0, v_direct, Li1))
				print(lline_1, rrow_1, lline_2, rrow_2, u_direct, Li0, v_direct, Li1)

				#print('\nCOORDS: %s %s' % (arr_lines_1[i, j], arr_rows_1[i, j]))
				#print('COORDS: %s %s\n' % (arr_lines_2[i, j], arr_rows_2[i, j]))

				# Peaks plot
				if plot_correlation_peaks:
					plot_peaks(im1, im2, u_direct, v_direct, iidx_line, iidx_row, result, pref,
							lline_1, rrow_1, lline_2, rrow_2, u_direct, Li0, v_direct, Li1)
					#plot_peaks(im1_bm, im2_bm, uu_bm, vv_bm, iidx_line, iidx_row,
					#           result_bm, 'bm')

				# If all elements are equal
				if np.unique(result).size == 1:
					return np.nan, np.nan, np.nan, np.nan, np.nan

				# If second peak close to first
				flat = result.flatten()
				flat.sort()

				#print('#Flat: %s' % flat)

				#if abs(flat[-1]-flat[-2]) < 0.05:
				#    return np.nan, np.nan, np.nan, np.nan, np.nan

				ret = (lline_1, rrow_1, rrow_2-rrow_1, lline_2-lline_1, cc_max)
				#return lline_1, rrow_1, u_direct, v_direct, cc_max
			else:
				#pass
				# ! Testing (return result in any case)
				ret = (np.nan, np.nan, np.nan, np.nan, np.nan)
				'''
				# if crop images have black stripes
				if flag1 == 1:
					print('IMG_1: %s_%s' % (iidx_line, iidx_row))
					io.imsave('ci_%s_1/black_%s_%s.png' % (Conf.out_fname, iidx_line, iidx_row), im1)
				if flag2 == 1:
					print('IMG_2: %s_%s' % (idx_line, idx_row))
					io.imsave('ci_%s_2/black_%s_%s.png' % (Conf.out_fname, iidx_line, iidx_row), im2)
				'''

		print("Processed block: {} from {}".format(itr, self.Count))
		return ret

	# TODO: check
	def check_borders(self, im):
		''' n pixels along line means image has a black border '''
		flag = 0
		ch = 0
		j = 0
		for i in range(im.shape[0] - 1):
			while j < im.shape[1] - 1 and im[i,j] > 0:
				j += 1
			else:
				if j < im.shape[1] - 1 and im[i,j] == 0:
					while im[i,j] == 0 and j < im.shape[1] - 1:
						j += 1
						ch += 1
					if ch >= 55:
						flag = 1
						#print('Black stripe detected!')
						return flag
			j = 0
			ch = 0
		return flag

	# Matching
	def matching(self, templ, im):
		''' Matching '''
		# Direct macthing
		#pool = Pool(processes=3)
		#result = pool.apply(match_template, args=(im, templ, True, 'edge',))
		#pool.close()

		result = match_template(im, templ, True, 'edge',)

		# Drihle statement
		# No need if 'edge' in 'match_template'
		#n = Conf.block_size #/ 2  # 100
		n = int(im.shape[0]/10.)
		# First and last n lines
		result[0:n, :] = 0.
		result[-n:, :] = 0.
		# First and last n rows
		result[:, 0:n] = 0.
		result[:, -n:] = 0.

		ij = np.unravel_index(np.argmax(result), result.shape)
		u_peak, v_peak = ij[::-1]

		#print('u_peak, v_peak: (%s, %s)' % (u_peak, v_peak))

		return u_peak, v_peak, result