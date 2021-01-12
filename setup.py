from numpy.distutils.core import setup, Extension
import setuptools
from Cython.Build import cythonize
from numpy import get_include
import os

nmpy_inc = get_include()

if 'HEALPIX_CXX_INC_DIR' in os.environ:
	hpx_inc_dir=os.environ['HEALPIX_CXX_INC_DIR']
else:
	print("Set HEALPIX_CXX_INC_DIR environment variable")

if 'SHARP_INC_DIR' in os.environ:
	sharp_inc_dir=os.environ['SHARP_INC_DIR']
else:
	print("Set SHARP_INC_DIR environment variable")

if 'HEALPIX_CXX_LIB_DIR' in os.environ:
	hpx_lib_dir=os.environ['HEALPIX_CXX_LIB_DIR']
else:
	print("Set HEALPIX_CXX_LIB_DIR environment variable")

LIBS=['healpix_cxx','sharp']#,'cxxsupport','sharp','fftpack','c_utils']
OPTIONS=['-O3','-std=c++11','-fopenmp','-fPIC']

setup(name='lensquest',
	version="beta0.1",
	description='Tools to estimate the CMB lensing potential for Python',
	author='Dominic Beck',
	author_email='dbeck@apc.in2p3.fr',
	packages=['lensquest'],
	ext_modules=cythonize([
	Extension('lensquest._lensquest_norm',
		sources=['lensquest/src/_lensquest_norm.pyx'],
		include_dirs=[hpx_inc_dir,sharp_inc_dir,nmpy_inc], 
		library_dirs=[hpx_lib_dir],
		libraries=LIBS, #
		extra_compile_args=OPTIONS,
		extra_link_args=['-fopenmp'],
		language='c++'),
	Extension('lensquest._lensquest_quest',
		sources=['lensquest/src/_lensquest_quest.pyx'],
		include_dirs=[hpx_inc_dir,sharp_inc_dir,nmpy_inc], 
		library_dirs=[hpx_lib_dir],
		libraries=LIBS, #
		extra_compile_args=OPTIONS,
		extra_link_args=['-fopenmp'],
		language='c++')
	]),
	#install_requires=['numpy', 'healpy'],
	license='GPLv2'
)
