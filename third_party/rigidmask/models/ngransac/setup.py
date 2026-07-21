from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

#opencv_inc_dir = '/data/tools/opencv/include/' # directory containing OpenCV header files
#opencv_lib_dir = '/usr/local/lib/' # directory containing OpenCV library files

opencv_inc_dir = '' # directory containing OpenCV header files
opencv_lib_dir = '' # directory containing OpenCV library files


def first_existing_path(paths, required_file=None):
	for path in paths:
		if not path:
			continue
		if required_file is None and os.path.isdir(path):
			return path
		if required_file is not None and os.path.exists(os.path.join(path, required_file)):
			return path
	return ''

#if not explicitly provided, we try to locate OpenCV in the current Conda environment
conda_env = os.environ.get('CONDA_PREFIX', '')

if len(conda_env) > 0 and len(opencv_inc_dir) == 0 and len(opencv_lib_dir) == 0:
	print("Detected active conda environment:", conda_env)

	opencv_inc_dir = first_existing_path([
		os.path.join(conda_env, 'include', 'opencv4'),
		os.path.join(conda_env, 'include'),
	], required_file=os.path.join('opencv2', 'opencv.hpp'))

	opencv_lib_dir = first_existing_path([
		os.path.join(conda_env, 'lib'),
	])

if len(opencv_inc_dir) == 0:
	opencv_inc_dir = first_existing_path([
		'/usr/include/opencv4',
		'/usr/local/include/opencv4',
		'/usr/include',
		'/usr/local/include',
	], required_file=os.path.join('opencv2', 'opencv.hpp'))

if len(opencv_lib_dir) == 0:
	opencv_lib_dir = first_existing_path([
		'/usr/lib/x86_64-linux-gnu',
		'/usr/local/lib',
		'/usr/lib64',
		'/usr/lib',
	])

if len(opencv_inc_dir) > 0 and len(opencv_lib_dir) > 0:
	print("Using OpenCV dependencies in:")
	print(opencv_inc_dir)
	print(opencv_lib_dir)

if len(opencv_inc_dir) == 0:
	print("Error: Could not locate an OpenCV include directory containing opencv2/opencv.hpp. Edit this file or install OpenCV development headers.")
	exit()
if len(opencv_lib_dir) == 0:
	print("Error: Could not locate an OpenCV library directory. Edit this file or install OpenCV development libraries.")
	exit()

setup(
	name='ngransac',
	ext_modules=[CppExtension(
		name='ngransac', 
		sources=['ngransac.cpp','thread_rand.cpp'],
		include_dirs=[opencv_inc_dir],
		library_dirs=[opencv_lib_dir],
		libraries=['opencv_core','opencv_calib3d'],
		extra_compile_args=['-fopenmp']
		)],		
	cmdclass={'build_ext': BuildExtension})
