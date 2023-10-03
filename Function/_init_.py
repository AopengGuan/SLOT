import multiprocessing
from skimage import img_as_float32, filters

if __name__ == '__main__':
	# enable support for multiprocessing
	multiprocessing.freeze_support()