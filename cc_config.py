import time
import os.path
import re
from skimage import io

class Config(object):
	"""Base configuration class. Constants and parameters here.
	"""

	def init(self, f1_name, f2_name, block_size, search_area, grid_step):
		### Section ###
		### files and folders ###

		self.f1_name = f1_name
		self.f2_name = f2_name

		timestr = time.strftime("%Y%m%d_%H_%M")
		self.res_dir = 'res_' + timestr + '/pyr'
		self.res_peaks_plot_dir = 'res_' + timestr + '/peaks_plot'

		### Section ###
		### calculation params ###

		self.image_intensity_byte_normalization = True

		#64 # 64 / 8 seems be good
		self.block_size = block_size
		#3 # 6, 8 - OK?
		self.search_area = search_area
		# Calculation step
		self.grid_step = grid_step

		self.rescale_apply = False
		self.rescale_factor = 2.

		# Plot peaks
		self.plot_correlation_peaks = False

		self.img_median_filtering = False
		self.median_kernel = 3

		self.img_laplace_filtering = False
		self.img_gradient_filtering = False
		self.img_scharr_filtering = False

		# Number of outliers filtering iterations
		self.num_iter_outliers = 4

		# Backmatching th
		self.bm_th = 10

	def self_prepare(self):
		assert (os.path.isfile(self.f1_name)), 'filename1 not found: "' + self.f1_name + '"'
		assert (os.path.isfile(self.f2_name)), 'filename2 not found: "' + self.f2_name + '"'

		# out file name and dir
		self.f1_date = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', self.f1_name)[0]
		self.f2_date = re.findall(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', self.f2_name)[0]
		self.out_fname = '%s_%s' % (self.f1_date, self.f2_date)

		os.makedirs(self.res_dir, exist_ok=True)

		if self.plot_correlation_peaks:
			os.makedirs(self.res_peaks_plot_dir, exist_ok=True)

		self.img1 = io.imread(self.f1_name, 'L')
		self.img2 = io.imread(self.f2_name, 'L')