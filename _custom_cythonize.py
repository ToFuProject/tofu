import os
import sys


# https://groups.google.com/g/cython-users
from Cython.Build import cythonize as _cythonize


# local
_PATH_HERE = os.path.dirname(__file__)
sys.path.insert(0, _PATH_HERE)
import tofu_helpers as tfh
sys.path.pop(0)


#  Compiling files
openmp_installed, openmp_flag = tfh.openmp_helpers.is_openmp_installed()


# #################################################
# #################################################
#           Prepare openmp
# #################################################


def cythonize(*args, **kwdargs):

    return _cythonize(
        *args,
        compile_time_env=dict(TOFU_OPENMP_ENABLED=openmp_installed),
        compiler_directives={"language_level": 3},
        **kwdargs,
    )
