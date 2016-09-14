from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from os import environ
import numpy

environ['CC'] = 'gcc'
environ['CXX'] = 'gcc'

extensions = [
        Extension("General_Geom_cy", ["General_Geom_cy.pyx"],
            include_dirs=[numpy.get_include()])
        ]

setup(
    name="General_Geom_cy",
    ext_modules=cythonize(extensions)
)

