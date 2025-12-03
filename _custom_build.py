"""
See:
    https://stackoverflow.com/questions/73800736/pyproject-toml-and-cython-extension-module
    https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#Cython.Build.cythonize
    https://setuptools.pypa.io/en/latest/userguide/extension.html

"""


import os
import sys


from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py
import numpy


# local
_PATH_HERE = os.path.dirname(__file__)
sys.path.insert(0, _PATH_HERE)
import tofu_helpers as tfh
sys.path.pop(0)


# #################################################
# #################################################
#           Prepare openmp
# #################################################


#  Compiling files
openmp_installed, openmp_flag = tfh.openmp_helpers.is_openmp_installed()

_OPTIONS = {
    'extra_compile_args': ["-O3", "-Wall", "-fno-wrapv"] + openmp_flag,
    'extra_link_args': [] + openmp_flag,
    'include_dirs': [numpy.get_include()],
}


# #################################################
# #################################################
#           DEFAULT
# #################################################


_LEXT = [
    Extension(
        name="tofu.geom._GG",
        sources=["tofu/geom/_GG.pyx"],
        **_OPTIONS,
    ),
    Extension(
        name="tofu.geom._basic_geom_tools",
        sources=["tofu/geom/_basic_geom_tools.pyx"],
        **_OPTIONS,
    ),
    Extension(
        name="tofu.geom._distance_tools",
        sources=["tofu/geom/_distance_tools.pyx"],
        **_OPTIONS,
    ),
    Extension(
        name="tofu.geom._sampling_tools",
        sources=["tofu/geom/_sampling_tools.pyx"],
        **_OPTIONS,
    ),
    Extension(
        name="tofu.geom._raytracing_tools",
        sources=["tofu/geom/_raytracing_tools.pyx"],
        **_OPTIONS,
    ),
    Extension(
        name="tofu.geom._vignetting_tools",
        sources=["tofu/geom/_vignetting_tools.pyx"],
        **_OPTIONS,
    ),
    Extension(
        name="tofu.geom._chained_list",
        sources=["tofu/geom/_chained_list.pyx"],
        **_OPTIONS,
    ),
    Extension(
        name="tofu.geom._sorted_set",
        sources=["tofu/geom/_sorted_set.pyx"],
        **_OPTIONS,
    ),
    Extension(
        name="tofu.geom._openmp_tools",
        sources=["tofu/geom/_openmp_tools.pyx"],
        # cython_compile_time_env=dict(TOFU_OPENMP_ENABLED=openmp_installed),
        **_OPTIONS,
    ),
]


# #################################################
# #################################################
#           Main class
# #################################################


class build_py(_build_py):

    # def run(self):
        # self.run_command("build_ext")
        # return super().run()

    def initialize_options(self):
        super().initialize_options()
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []

        self.distribution.ext_modules += _LEXT
