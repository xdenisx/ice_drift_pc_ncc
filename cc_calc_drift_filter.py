import numpy as np
from sklearn.neighbors import KDTree

class CalcDriftFilter(object):
	def __init__(self, Conf):
		self.Conf = Conf

	# Filter outliers
	def filter_outliers(self, drift_results):
		'''Filter outliers'''
		# Good ob
		radius = self.Conf.grid_step * 8 #128
		angle_difference = 3
		length_difference = 5
		total_neighbours = 5
		angle_neighbours = 5
		length_neighbours = 5

		# Get list without none and each elements
		ll_data = [x for x in drift_results if x is not None]

		yyy = [i[0] for i in ll_data]
		xxx = [i[1] for i in ll_data]
		uuu = [i[2] for i in ll_data]
		vvv = [i[3] for i in ll_data]
		ccc = [i[4] for i in ll_data]

		# Get indexes of raw data (x-component) where elements are not NaN
		idxs_not_nan = np.argwhere(np.isnan(uuu)==False).transpose()

		# Get elements with not NaN values to filter outliers but preserve their indexes in 'idxs_not_nan'
		xxx_f, yyy_f, uuu_f, vvv_f, ccc_f = np.take(xxx, idxs_not_nan)[0], np.take(yyy, idxs_not_nan)[0], \
											np.take(uuu, idxs_not_nan)[0], np.take(vvv, idxs_not_nan)[0], \
											np.take(ccc, idxs_not_nan)[0]

		if self.Conf.num_iter_outliers > 0:
			i = 0
			while i < self.Conf.num_iter_outliers:
				print('Iter. number: %d' % i)
				xxx_f, yyy_f, uuu_f, vvv_f, ccc_f = self.filter_iterate(xxx_f, yyy_f, uuu_f, vvv_f, ccc_f,
																		radius=radius,
																		angle_difference=angle_difference,
																		length_difference=length_difference,
																		total_neighbours=total_neighbours,
																		angle_neighbours=angle_neighbours,
																		length_neighbours=length_neighbours)
				i = i + 1

		self.idxs_not_nan = idxs_not_nan
		self.xxx_f = xxx_f
		self.yyy_f = yyy_f
		self.uuu_f = uuu_f
		self.vvv_f = vvv_f
		self.ccc_f = ccc_f

		return len(uuu_f), len(uuu_f[~np.isnan(uuu_f)])

	def filter_iterate(self, x1, y1, uu, vv, cc,
							radius=256,
							angle_difference=15,
							length_difference=30,
							total_neighbours=3,
							angle_neighbours=3,
							length_neighbours=3):
		# Get values of vector components
		#uu = x2 - x1
		#vv = y2 - y1

		idx_mask = []
		# Make 2D data of components
		#data = np.vstack((uu, vv)).T

		x1, y1, uu, vv, cc = np.array(x1), np.array(y1),\
							np.array(uu, np.float), np.array(vv, np.float), np.array(cc, np.float)

		#  Radius based filtering
		vector_start_data = np.vstack((x1, y1)).T
		vector_start_tree = KDTree(vector_start_data)

		for i in range(0, len(x1), 1):
			# For list
			# req_data = np.array([x1[i], y1[i]]).reshape(1, -1)
			req_data = np.array((x1[i], y1[i])).reshape(1, -1)
			# Getting number of neighbours
			num_nn = vector_start_tree.query_radius(req_data, r=radius, count_only=True)

			if num_nn[0] < total_neighbours:
				idx_mask.append(i)

			if np.isnan(vv[i]):
				idx_mask.append(i)
			else:
				px = 2

				y1_max = int(min(y1[i] + px, self.Conf.img1.shape[0] - 1))
				y1_min = int(max(y1[i] - px, 0))

				x1_max = int(min(x1[i] + px, self.Conf.img1.shape[1] - 1))
				x1_min = int(max(x1[i] - px, 0))

				y2_max = int(min(y1[i] + vv[i] + px, self.Conf.img1.shape[0] - 1))
				y2_min = int(max(y1[i] + vv[i] - px, 0))

				x2_max = int(min(x1[i] + uu[i] + px, self.Conf.img1.shape[1] - 1))
				x2_min = int(max(x1[i] + uu[i] - px, 0))

				meanI_start_01 = np.nanmean(self.Conf.img1[y1_min:y1_max, x1_min:x1_max])
				meanI_start_02 = np.nanmean(self.Conf.img2[y1_min:y1_max, x1_min:x1_max])
				meanI_end = np.nanmean(self.Conf.img2[y2_min:y2_max, x2_min:x2_max])

				# Keep small vectors
				if np.hypot(uu[i], vv[i]) < 5.:
					pass
				else:
					if (meanI_start_01==0. or meanI_start_01==255. or
							meanI_start_02==0. or meanI_start_02==255.
							or meanI_end==0. or meanI_end==255.):
						idx_mask.append(i)
						continue

					nn = vector_start_tree.query_radius(req_data, r=radius)
					data = np.vstack((uu[nn[0]], vv[nn[0]])).T

					num_of_homo_NN = 0
					num_of_length_homo_NN = 0

					####################################################################
					# Loop through all found ice drift vectors to filter not homo
					####################################################################
					for ii in range(num_nn[0]):

						# Angle between "this" vector and others
						angle_v1_v2 = self.angle_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])

						# Length between "this" vector and others
						diff_v1_v2 = self.length_between([uu[i], vv[i]], [data[:, 0][ii], data[:, 1][ii]])

						if angle_v1_v2 <= angle_difference:
							num_of_homo_NN = num_of_homo_NN + 1

						if diff_v1_v2 < length_difference:
							num_of_length_homo_NN = num_of_length_homo_NN + 1

					if not (num_of_homo_NN >= angle_neighbours and num_of_length_homo_NN >= length_neighbours):
						idx_mask.append(i)

		tt = list(set(idx_mask))
		iidx_mask = np.array(tt)

		# Delete bad data
		'''
		x1_f = np.delete(x1, iidx_mask)
		y1_f = np.delete(y1, iidx_mask)
		uu_f = np.delete(uu, iidx_mask)
		vv_f = np.delete(vv, iidx_mask)
		cc_f = np.delete(cc, iidx_mask)
		'''

		# Mask (=NaN) bad values
		uu = np.array(uu, np.float)
		vv = np.array(vv, np.float)
		uu[iidx_mask] = np.nan
		vv[iidx_mask] = np.nan
		cc[iidx_mask] = 0.

		return x1, y1, uu, vv, cc

	def angle_between(self, v1, v2):
		""" Returns the angle in radians between vectors 'v1' and 'v2'::

			angle_between((1, 0, 0), (0, 1, 0))
			1.5707963267948966
			angle_between((1, 0, 0), (1, 0, 0))
			0.0
			angle_between((1, 0, 0), (-1, 0, 0))
			3.141592653589793
		"""
		v1_u = self.unit_vector(v1)
		v2_u = self.unit_vector(v2)
		angle = np.arccos(np.dot(v1_u, v2_u))
		if np.isnan(angle):
			if (v1_u == v2_u).all():
				return np.degrees(0.0)
			else:
				return np.degrees(np.pi)
		return np.degrees(angle)

	def length_between(self, v1, v2):
		v1_length = np.hypot(v1[0], v1[1])
		v2_length = np.hypot(v2[0], v2[1])
		return abs(v1_length - v2_length)

	def unit_vector(self, vector):
		""" Returns the unit vector of the vector.  """
		return vector / np.linalg.norm(vector)