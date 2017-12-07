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
if sys.version[:3] in ['2.7','3.6']:
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
    # Use scm to get code version from git tags
    # cf. https://pypi.python.org/pypi/setuptools_scm 
    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    # The version is stored only in the setup.py file and read from it (option 1 in https://packaging.python.org/en/latest/single_source_version.html)
    use_scm_version=False,
    #setup_requires=['setuptools_scm'],

    description='A python library for Tomography for Fusion',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/ToFuProject/tofu',

    # Author details
    author='Didier VEZINET',
    author_email='didier.vezinet@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',

        # In which language most of the code is written ?
        'Natural Language :: English',
    ],

    # What does your project relate to?
    keywords='tomography geometry 3D inversion synthetic fusion',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages = find_packages(exclude=['doc', '_Old', '_Old_doc','plugins','plugins.*','*.plugins.*','*.plugins','*.tests10_plugins','*.tests10_plugins.*','tests10_plugins.*','tests10_plugins',]),
    #packages = ['tofu','tofu.geom'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    # py_modules=["my_module"],

    # List run-time dependencies here. These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            poly,
            'cython',
            ],

    python_requires = '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,!=3.5.*',


    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage','nose==1.3.4'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here. If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #package_data={
    #    # If any package contains *.txt, *.rst or *.npz files, include them:
    #    '': ['*.txt', '*.rst', '*.npz'],
    #    # And include any *.csv files found in the 'ITER' package, too:
    #    'ITER': ['*.csv'],
    #},
    package_data={'tofu.tests.tests01_geom':['test_Ves.txt']},
    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    #entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},

    ext_modules = extensions,
    #cmdclass={'build_ext':build_ext},
    include_dirs=[np.get_include()],
)


