from numpy.distutils.core import setup, Extension
import setuptools
from Cython.Build import cythonize
from numpy import get_include
import os

nmpy_inc = get_include()

if 'HEALPIX_CXX_DIR' in os.environ:
	hpx_dir=os.environ['HEALPIX_CXX_DIR']
else:
	print("Set HEALPIX_CXX_DIR environment variable (<path/to/healpix/installation>/src/cxx/generic_gcc/)")

LIBS=['healpix_cxx','cxxsupport','sharp','fftpack','c_utils']
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
		include_dirs=[os.path.join(hpx_dir,"include"),nmpy_inc], 
		library_dirs=[os.path.join(hpx_dir,"lib")],
		libraries=LIBS, #
		extra_compile_args=OPTIONS,
		extra_link_args=['-fopenmp'],
		language='c++'),
	Extension('lensquest._lensquest_quest',
		sources=['lensquest/src/_lensquest_quest.pyx'],
		include_dirs=[os.path.join(hpx_dir,"include"),nmpy_inc], 
		library_dirs=[os.path.join(hpx_dir,"lib")],
		libraries=LIBS, #
		extra_compile_args=OPTIONS,
		extra_link_args=['-fopenmp'],
		language='c++')
	]),
	install_requires=['numpy', 'healpy'],
	license='GPLv2'
)
