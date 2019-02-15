""" A tomography library for fusion devices

See:
https://github.com/ToFuProject/tofu
"""

import sys
import os
import subprocess
import shutil
from codecs import open
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

# Always prefer setuptools over distutils
try:
    from setuptools import setup, find_packages
    from setuptools import Extension
    stp = True
except:
    from distutils.core import setup
    from distutils.extension import Extension
    stp = False
import _updateversion as up

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'gcc'

here = os.path.abspath(os.path.dirname(__file__))
version_git = up.updateversion(os.path.join(here,'tofu'))

# To compile the relevant version
if sys.version[:3] in ['2.7','3.6','3.7']:
    gg = '_GG0%s' % sys.version[0]
    poly = 'polygon%s' % sys.version[0]
else:
    raise Exception("Pb. with python version in setup.py file: "+sys.version)

print("")
print("Version for setup.py : ", version_git)
print("")


# Getting relevant compilable files
if sys.version[0]=='3':
    #if not '_GG03.pyx' in os.listdir(os.path.join(here,'tofu/geom/')):
    shutil.copy2(os.path.join(here,'tofu/geom/_GG02.pyx'), os.path.join(here,'tofu/geom/_GG03.pyx'))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# Prepare extensions
# Useful if install from setup.py
#if '--use-cython' in sys.argv:
#    USE_CYTHON = True
#    sys.argv.remove('--use-cython')
#else:
#    USE_CYTHON = False
USE_CYTHON = True
if USE_CYTHON:
    print("")
    print("Using Cython !!!!!!!!!")
    print("")
    extensions = [Extension(name="tofu.geom."+gg, sources=["tofu/geom/"+gg+".pyx"])]
    extensions = cythonize(extensions)
else:
    print("")
    print("NOT Using Cython !!!!!!!!!")
    print("")
    extensions = [Extension(name="tofu.geom."+gg, sources=["tofu/geom/"+gg+".cpp"],
                            language='c++', include_dirs=['tofu/cpp/'])]
setup(
    name='tofu',
    #version="1.2.27",
    version="{ver}".format(ver=version_git),
    use_scm_version=False,
    description='A python library for Tomography for Fusion',
    long_description=long_description,
    url='https://github.com/ToFuProject/tofu',
    author='Didier VEZINET',
    author_email='didier.vezinet@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',

        # In which language most of the code is written ?
        'Natural Language :: English',
        ],

 # What does your project relate to?
    keywords='tomography geometry 3D inversion synthetic fusion',
    packages = find_packages(exclude=['doc', '_Old', '_Old_doc','plugins','plugins.*','*.plugins.*','*.plugins','*.tests10_plugins','*.tests10_plugins.*','tests10_plugins.*','tests10_plugins',]),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        poly,
        'cython',
        'pandas',
    ],
    python_requires = '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,!=3.5.*',
    extras_require={
     'dev' : ['check-manifest'],
     'test': ['coverage','nose==1.3.4'],
    },
    package_data={'tofu.tests.tests01_geom.tests03core_data':['*.py','*.txt']},
    include_package_data=True,
    ext_modules = extensions,
    include_dirs=[np.get_include()],
    )
