import os
import sys
import time
import shutil
import datetime
import tempfile
import platform
import subprocess

from distutils.dist import Distribution
from distutils.sysconfig import customize_compiler
from numpy.distutils.ccompiler import new_compiler
from numpy.distutils.command.config_compiler import config_cc


def is_platform_windows():
    """Checks if platfom is windows

    Return
    --------
    bool
        True if on Windows
        False if not

    """
    return platform.system() == "Windows"


omp_source = """
#include <omp.h>
#include <stdio.h>
int main(void) {
  #pragma omp parallel
  printf("nthreads=%d\\n", omp_get_num_threads());
  return 0;
}
"""


def get_compiler():
    """Get a compiler equivalent to the one that will be used to build tofu
    when doing setup.py

    Handles compiler specified as follows:
        - python setup.py build_ext --compiler=<compiler>
        - CC=<compiler> python setup.py build_ext
    """
    dist = Distribution({'script_name': os.path.basename(sys.argv[0]),
                         'script_args': sys.argv[1:],
                         'cmdclass': {'config_cc': config_cc}})
    dist.parse_config_files()
    dist.parse_command_line()

    cmd_opts = dist.command_options.get('build_ext')

    if cmd_opts is not None and 'compiler' in cmd_opts:
        compiler = cmd_opts['compiler'][1]
    else:
        compiler = None

    ccompiler = new_compiler(compiler=compiler)
    customize_compiler(ccompiler)
    return ccompiler


def get_openmp_flag(compiler):
    """Returns list of flags for using OpenMP depending on compiler and
    platform.

    Parameters
    ----------
    compiler : numpy.distutils.compiler
        Compiler used when invoking setup.py build

    """
    if hasattr(compiler, 'compiler'):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ('icc' in compiler or 'icl' in compiler):
        return ['/Qopenmp']
    elif sys.platform == "win32":
        return ['/openmp']
    elif sys.platform in ("darwin", "linux") and "icc" in compiler:
        return ['-qopenmp']
    elif sys.platform == "darwin" and 'openmp' in os.getenv('CPPFLAGS', ''):
        return ['-openmp']
    # Default flag for GCC and clang:
    return ['-fopenmp']


def check_for_openmp():
    """Compiles small code sample to see if OpenMP is installed or not.
    Returns corresponding flags to compile depending on platform and
    compiler.

    Returns
    -------
    int
        0 if no error, else subprocess.call error code
    list
        list of strings indicating openmp flags corresponding or empty
    """

    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = "test.c"
    compiler = get_compiler()
    flag_omp = get_openmp_flag(compiler)

    # getting compiler name
    if hasattr(compiler, 'compiler'):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    try:
        with open(filename, "w") as file:
            file.write(omp_source)
        with open(os.devnull, "w") as fnull:
            result = subprocess.call(
                [compiler] + flag_omp + [filename], stdout=fnull, stderr=fnull,
                shell=is_platform_windows()
            )
    except subprocess.CalledProcessError:
        result = -1

    finally:
        # in any case, go back to previous cwd and clean up
        os.chdir(curdir)
        shutil.rmtree(tmpdir)
        if not result == 0:
            flag_omp = []
    return result, flag_omp


def is_openmp_installed():
    """ Returns ``True`` if OpenMP is installed, else ``False`` and a list of
    flags needed to compile with openmp if installed (or an empty list)
    and generates ``tofu.openmp_enabled.is_openmp_enabled``, which can then be
    used to determine, post build, whether the package was built with or
    without OpenMP support."""

    openmp_support, flag = check_for_openmp()
    openmp_enabled = not openmp_support
    generate_openmp_enabled_py(openmp_enabled)

    return openmp_enabled, flag


_IS_OPENMP_ENABLED_SRC = """
# Autogenerated by Tofu's setup.py on {timestamp!s}
def is_openmp_enabled():
    \"\"\"
    Determine whether this package was built with OpenMP support.
    \"\"\"
    return {return_bool}
"""[1:]


def generate_openmp_enabled_py(openmp_support, srcdir='.'):
    """
    Generate ``tofu.openmp_enabled.is_openmp_enabled``, which can then be used
    to determine, post build, whether the package was built with or without
    OpenMP support.
    """

    epoch = int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))
    timestamp = datetime.datetime.utcfromtimestamp(epoch)

    src = _IS_OPENMP_ENABLED_SRC.format(timestamp=timestamp,
                                        return_bool=openmp_support)

    package_srcdir = os.path.join(srcdir, "tofu", "geom")
    is_openmp_enabled_py = os.path.join(package_srcdir, 'openmp_enabled.py')
    with open(is_openmp_enabled_py, 'w') as f:
        f.write(src)
