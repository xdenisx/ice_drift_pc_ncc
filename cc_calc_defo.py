import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import KDTree
import numpy as np

class CalcDefo(object):
	def __init__(self, Conf, Calc, Filter):
		self.Conf = Conf
		self.Calc = Calc
		self.Filter = Filter

	def calculate_defo(self):
		# Get 2D matrix coordinates (for deformation calculation) of not NaN elements
		idxs_2d = self.Calc.idxs_2d
		idxs_not_nan = self.Filter.idxs_not_nan
		idxs_2d_not_nan = np.array(idxs_2d)[idxs_not_nan][0]

		u_2d = self.Calc.u_2d
		v_2d = self.Calc.v_2d

		u_2d_int = self.Calc.u_2d
		v_2d_int = self.Calc.v_2d

		u_2d_med = self.Calc.u_2d_med
		v_2d_med = self.Calc.v_2d_med

		xxx_f = self.Filter.xxx_f
		yyy_f = self.Filter.yyy_f
		uuu_f = self.Filter.uuu_f
		vvv_f = self.Filter.vvv_f
		ccc_f = self.Filter.ccc_f

		# Median smoothing for deformation
		num_iter_median = 0
		if num_iter_median > 0:
			i = 0
			while i < num_iter_median:
				print('Median iter. number: %d' % i)
				# TODO: Error here! Do the same i times
				x1_med, y1_med, uu_med, vv_med, cc_med = median_filtering(xxx_f, yyy_f, uuu_f, vvv_f, ccc_f,
														radius=radius*2, total_neighbours=7)
				i += 1
			#plot_arrows('%s/01_median_%s_%s.png' % (Conf.res_dir, pref, Conf.out_fname), img1, x1_med, y1_med,
			#			uu_med, vv_med, cc_med,
			#			arrwidth=0.002, headwidth=5.5, flag_color=True)

		# Fill 2D matrix with a data for deformation calculation
		# Raw filtered vectors
		# Invert to calculate actual position of deformation
		for ch, index in enumerate(idxs_2d_not_nan):
			ii, jj = index[0], index[1]
			u_2d[ii, jj] = uuu_f[ch]
			v_2d[ii, jj] = vvv_f[ch]

		# Replace nan vectors with nearest values (within 300 pixels)
		'''
		print('Start interpolation of drift data')
		vector_start_data = np.vstack((self.Calc.row_2d.ravel(), self.Calc.col_2d.ravel())).T
		vector_start_tree = KDTree(vector_start_data)

		for r in range(len(u_2d[:,0])):
			for c in range(len(u_2d[0,:])):
				if np.isnan(u_2d[r,c]):

					# Average neighbors
					req_data = np.array((self.Calc.row_2d[r, c], self.Calc.col_2d[r, c])).reshape(1, -1)

					nn = vector_start_tree.query_radius(req_data, r=200)[0]
					u_2d[r, c] = np.nanmedian(u_2d.ravel()[nn])
					v_2d[r, c] = np.nanmedian(v_2d.ravel()[nn])
		'''

		# Median vectors
		'''
		for ch, index in enumerate(idxs_2d_not_nan):
			ii, jj = index[0], index[1]
			u_2d_med[ii,jj] = uu_med[ch]
			v_2d_med[ii,jj] = vv_med[ch]
		'''

		# Calculate deformation from the 2D arrays

		# get pixel size (in meters)
		gd_raster = gdal.Open(self.Conf.f1_name)
		pixel_size = gd_raster.GetGeoTransform()[1]

		#Calculate time difference between images
		dt1 = datetime.strptime(self.Conf.f1_date, '%Y%m%dT%H%M%S')
		dt2 = datetime.strptime(self.Conf.f2_date, '%Y%m%dT%H%M%S')
		time_dif = (dt2 - dt1).total_seconds()

		# Calcualte deformation for raw vectors
		mag_speed, divergence, curl, shear, total_deform = self.calculate(u_2d, v_2d,
																normalization=True,
																normalization_time=time_dif,
																raster_cell_size=pixel_size,
																defo_cell_size=self.Conf.grid_step,
																invert_meridional=True)

		# Calculate deformation for smoothed vectors
		'''
		mag_speed, divergence, curl, shear, total_deform = self.calculate(u_2d_med, v_2d_med,
																normalization=True,
																normalization_time=86400.*2,
																cell_size=pixel_size,
																invert_meridional=True,
																out_png_name='test_med.png')
		'''

		return mag_speed, divergence, curl, shear, total_deform, u_2d, v_2d

	def calculate(self, dx, dy,
				normalization=True,
				normalization_time=None,
				raster_cell_size=100.,
				defo_cell_size=5000.,
				invert_meridional=True,
				fill_NaN=False):
		'''
		Calculate deformation invariants from U and V ice drift components

		dx, dy - x and y component of motion (pixels)
		normalization - normalize to time (boolean)
		normalization_time - normalization time (in seconds)
		cell_size - ground meters in a pixel
		invert_meridional - invert y component (boolean)
		'''

		if fill_NaN:
			print('\nFilling NaN values with nearest not NaN\n')
			###############################
			# Fill NaNs with NN values
			###############################
			# dX
			mask = np.isnan(dx)
			idx = np.where(~mask, np.arange(mask.shape[1]), 0)
			np.fmax.accumulate(idx, axis=1, out=idx)
			# out = arr[np.arange(idx.shape[0])[:, None], idx]
			dx[mask] = dx[np.nonzero(mask)[0], idx[mask]]

			# dY
			mask = np.isnan(dy)
			idx = np.where(~mask, np.arange(mask.shape[1]), 0)
			np.fmax.accumulate(idx, axis=1, out=idx)
			# out = arr[np.arange(idx.shape[0])[:, None], idx]
			dy[mask] = dy[np.nonzero(mask)[0], idx[mask]]

		# Cell and grid size to cm
		raster_cell_size_cm = raster_cell_size * 100.
		defo_cell_size_cm = defo_cell_size * raster_cell_size_cm

		print('\n### raster_cell_size_cm %s [cm]' % raster_cell_size_cm)
		print('\n### defo_cell_size_cm %s [cm]\n' % defo_cell_size_cm)

		if normalization:
			print('Normalization...')
			# Convert to ground distance (pixels*cell size)
			dx = dx * raster_cell_size_cm # cm
			dy = dy * raster_cell_size_cm # cm

			# Get U/V components of speed (cm/s)
			dx = dx / normalization_time
			dy = dy / normalization_time

		m_div = np.empty((dx.shape[0], dx.shape[1],))
		m_div[:] = np.NAN
		m_curl = np.empty((dx.shape[0], dx.shape[1],))
		m_curl[:] = np.NAN
		m_shear = np.empty((dx.shape[0], dx.shape[1],))
		m_shear[:] = np.NAN
		m_tdef = np.empty((dx.shape[0], dx.shape[1],))
		m_tdef[:] = np.NAN

		# Invert meridional component
		if invert_meridional:
			dy = dy * (-1)

		# Calculate magnitude (speed module) (cm/s)
		mag_speed = np.hypot(dx, dy)

		# Print mean speed in cm/s
		print('Mean speed: %s [cm/s]' % (np.nanmean(mag_speed)))

		#cell_size_factor = 1 / cell_size

		# Test
		#plt.clf()
		#plt.imshow(m_div)

		# Convert to cms per hour
		dx = dx * 3600.
		dy = dy * 3600.

		for i in range(1, dx.shape[0] - 1):
			for j in range(1, dx.shape[1] - 1):
				# div
				if (np.isnan(dx[i, j + 1]) == False and np.isnan(dx[i, j - 1]) == False
						and np.isnan(dy[i - 1, j]) == False and np.isnan(dy[i + 1, j]) == False
						and (np.isnan(dx[i, j]) == False or np.isnan(dy[i, j]) == False)):
					m_div[i, j] = 0.5 * ((dx[i, j + 1] - dx[i, j - 1]) + (dy[i - 1, j] - dy[i + 1, j])) / defo_cell_size_cm
					#print('dx[i,j+1]: %s    div: %s' % (dx[i, j + 1], m_div[i, j]))

				# Curl
				if (np.isnan(dy[i, j + 1]) == False and np.isnan(dy[i, j - 1]) == False and
						np.isnan(dx[i - 1, j]) == False and np.isnan(dx[i + 1, j]) == False
						and (np.isnan(dx[i, j]) == False or np.isnan(dy[i, j]) == False)):
					m_curl[i, j] = 0.5 * (dy[i, j + 1] - dy[i, j - 1] - dx[i - 1, j] + dx[i + 1, j]) / defo_cell_size_cm

				# Shear
				if (np.isnan(dy[i + 1, j]) == False and np.isnan(dy[i - 1, j]) == False and
						np.isnan(dx[i, j - 1]) == False and np.isnan(dx[i, j + 1]) == False and
						np.isnan(dy[i, j - 1]) == False and np.isnan(dy[i, j + 1]) == False and
						np.isnan(dx[i + 1, j]) == False and np.isnan(dx[i - 1, j]) == False and
						(np.isnan(dx[i, j]) == False or np.isnan(dy[i, j]) == False)):
					dc_dc = 0.5 * (dy[i + 1, j] - dy[i - 1, j])
					dr_dr = 0.5 * (dx[i, j - 1] - dx[i, j + 1])
					dc_dr = 0.5 * (dy[i, j - 1] - dy[i, j + 1])
					dr_dc = 0.5 * (dx[i + 1, j] - dx[i - 1, j])

					# !Exclude cell size factor!
					m_shear[i, j] = np.sqrt((dc_dc - dr_dr) * (dc_dc - dr_dr) + (dc_dr - dr_dc) * (dc_dr - dr_dc)) / defo_cell_size_cm

				# Total deformation
				if (np.isnan(m_shear[i, j]) == False and np.isnan(m_div[i, j]) == False):
					m_tdef[i, j] = np.hypot(m_shear[i, j], m_div[i, j])

		# Invert dy back
		if invert_meridional:
			dy = dy * (-1)

		# Plot u and v values inside cells (for testing porposes)
		'''
		font_size = .0000003
		for ii in range(dx.shape[1]):
			for jj in range(dx.shape[0]):
				try:
					if not np.isnan(divergence[ii,jj]):
						if divergence[ii,jj] > 0:
							plt.text(jj, ii,
								'u:%.2f\nv:%.2f\n%s ij:(%s,%s)\n%.6f' %
								(dx[ii,jj], dy[ii,jj], '+', ii, jj, divergence[ii,jj]),
								horizontalalignment='center',
								verticalalignment='center', fontsize=font_size, color='k')
						if divergence[ii,jj] < 0:
							plt.text(jj, ii,
								'u:%.2f\nv:%.2f\n%s ij:(%s,%s)\n%.6f' %
								(dx[ii,jj], dy[ii,jj], '-', ii, jj, divergence[ii,jj]),
								horizontalalignment='center',
								verticalalignment='center', fontsize=font_size, color='k')
						if divergence[ii,jj] == 0:
							plt.text(jj, ii,
								'u:%.2f\nv:%.2f\n%s ij:(%s,%s)\n%.6f' %
								(dx[ii,jj], dy[ii,jj], '0', ii, jj, divergence[ii,jj]),
								horizontalalignment='center',
								verticalalignment='center', fontsize=font_size, color='k')

						if np.isnan(divergence[ii,jj]):
							plt.text(jj, ii,
								'u:%.2f\nv:%.2f\n%s ij:(%s,%s)' % 
								(dx[ii,jj], dy[ii,jj], '-', ii, jj),
								horizontalalignment='center',
								verticalalignment='center', fontsize=font_size, color='k')

					# Plot arrows on top of the deformation
					xxx = range(dx.shape[1])
					yyy = range(dx.shape[0])
				except:
					pass

		'''
		# Plot drift arrows on the top
		#import matplotlib.cm as cm
		#from matplotlib.colors import Normalize
		# Invert meridional component for plotting
		ddy = dy * (-1)

		#norm = Normalize()
		colors = np.hypot(dx, ddy)
		#print(colors)
		#norm.autoscale(colors)
		# we need to normalize our colors array to match it colormap domain
		# which is [0, 1]
		#colormap = cm.inferno

		return mag_speed, m_div, m_curl, m_shear, m_tdef
