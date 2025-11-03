"""
See:
    https://stackoverflow.com/questions/73800736/pyproject-toml-and-cython-extension-module

"""


from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py


# local
from tofu_helpers.openmp_helpers import is_openmp_installed


# #################################################
# #################################################
#           Prepare openmp
# #################################################


#  Compiling files
openmp_installed, openmp_flag = is_openmp_installed()
extra_compile_args = ["-O3", "-Wall", "-fno-wrapv"] + openmp_flag
extra_link_args = [] + openmp_flag


# #################################################
# #################################################
#           DEFAULT
# #################################################


_LEXT = [
    Extension(
        name="tofu.geom._GG",
        sources=["tofu/geom/_GG.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language_level="3",
    ),
    Extension(
        name="tofu.geom._basic_geom_tools",
        sources=["tofu/geom/_basic_geom_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._distance_tools",
        sources=["tofu/geom/_distance_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._sampling_tools",
        sources=["tofu/geom/_sampling_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._raytracing_tools",
        sources=["tofu/geom/_raytracing_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._vignetting_tools",
        sources=["tofu/geom/_vignetting_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._chained_list",
        sources=["tofu/geom/_chained_list.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._sorted_set",
        sources=["tofu/geom/_sorted_set.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._openmp_tools",
        sources=["tofu/geom/_openmp_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cython_compile_time_env=dict(TOFU_OPENMP_ENABLED=openmp_installed),
    ),
]


# #################################################
# #################################################
#           Main class
# #################################################


class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []

        self.distribution.ext_modules += _LEXT
