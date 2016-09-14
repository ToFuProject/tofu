from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from os import environ
import numpy

environ['CC'] = 'gcc'
environ['CXX'] = 'gcc'

extensions = [
        Extension("_bsplines_cy", ["_bsplines_cy.pyx"],
            include_dirs=[numpy.get_include()])
        ]

setup(
    name="_bsplines_cy",
    ext_modules=cythonize(extensions)
)

