
try:
    from setuptools import setup
    from setuptools import Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize
from os import environ
import sys
import numpy as np

environ['CC'] = 'gcc'
environ['CXX'] = 'gcc'


if sys.version[0] in ['2','3']:
    name_src = '_GG0' + sys.version[0]
    #name_ext = 'tofu.geom.GG0' + sys.version[0]
    #name_set = 'tofu.geom.GG0' + sys.version[0]
    name_ext = '_GG0' + sys.version[0]
    name_set = '_GG0' + sys.version[0]
else:
    raise Exception("Pb. with python version : "+sys.version)


ext_modules = [Extension(name=name_ext, sources=[name_src+".pyx"])]

setup(name=name_set,
      cmdclass={'build_ext':build_ext},
      include_dirs=[np.get_include()],  
      ext_modules=ext_modules)

# ext_modules=cythonize(extensions)


