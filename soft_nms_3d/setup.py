from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='Soft Nms',
      ext_modules=cythonize("cpu_soft_nms.pyx"),
include_dirs=[numpy.get_include()])

