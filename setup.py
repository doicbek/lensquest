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

if 'CFITSIO_DIR' in os.environ:
	cfts_dir=os.environ['CFITSIO_DIR']
else:
	print("Set CFITSIO_DIR environment variable (<path/to/cfitsio/installation>)")

if 'GSL_DIR' in os.environ:
	gsl_dir=os.environ['GSL_DIR']
else:
	print("Set GSL_DIR environment variable")

	
LIBS=['healpix_cxx','cxxsupport','sharp','fftpack','c_utils','cfitsio','gsl','gslcblas']
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
		include_dirs=[os.path.join(hpx_dir,"include"),os.path.join(cfts_dir,"include"),gsl_dir,nmpy_inc], 
		library_dirs=[os.path.join(hpx_dir,"lib"),os.path.join(cfts_dir,"lib"),os.path.join(gsl_dir,".libs"),os.path.join(gsl_dir,"cblas/.libs")],
		libraries=LIBS, #
		extra_compile_args=OPTIONS,
		extra_link_args=['-fopenmp'],
		language='c++'),
	Extension('lensquest._lensquest_quest',
		sources=['lensquest/src/_lensquest_quest.pyx'],
		include_dirs=[os.path.join(hpx_dir,"include"),os.path.join(cfts_dir,"include"),gsl_dir,nmpy_inc], 
		library_dirs=[os.path.join(hpx_dir,"lib"),os.path.join(cfts_dir,"lib"),os.path.join(gsl_dir,".libs"),os.path.join(gsl_dir,"cblas/.libs")],
		libraries=LIBS, #
		extra_compile_args=OPTIONS,
		extra_link_args=['-fopenmp'],
		language='c++')
	]),
	install_requires=['matplotlib', 'numpy', 'healpy'],
	license='GPLv2'
)
