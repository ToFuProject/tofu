# ###################################
# ###################################
#        General
# ###################################


# version: run_command(['git', 'describe'], capture:true, check:false).stdout().strip().split('-')[0],
# https://mesonbuild.com/Builtin-options.html
project(
  'tofu',
  'c', 'cpp', 'cython',
  license: 'MIT',
  meson_version: '>=1.8.3',
  version: run_command(['python', 'tofu/version.py'], check:true).stdout().strip(),
  default_options: [
    'buildtype=release',
    'c_std=c11',
    'cpp_std=c++17',
    'pkgconfig.relocatable=true',
  ],
)

# ------------------
# project
# ------------------

# refs:
# https://github.com/scipy/scipy/blob/main/meson.build
# https://github.com/numpy/numpy/blob/main/meson.build
# https://github.com/silx-kit/silx/blob/main/meson.build

# find local python install
py_mod = import('python')
py = py_mod.find_installation(pure: false)
os = import('fs')

# ------------------------
# Min / max numpy versions
# ------------------------

min_numpy_version = '1.26.4'  # keep in sync with pyproject.toml
min_python_version = '3.9'   # keep in sync with pyproject.toml

python_version = py.language_version()
if python_version.version_compare(f'<@min_python_version@')
  error(f'Min Python version is @min_python_version@, found @python_version@')
endif

# ------------------
# check  platform
# ------------------

# Emit a warning for 32-bit Python installs on Windows; users are getting
# unexpected from-source builds there because we no longer provide wheels.
is_windows = host_machine.system() == 'windows'
if is_windows and py.has_variable('EXT_SUFFIX')
  ext_suffix = py.get_variable('EXT_SUFFIX')
  if ext_suffix.contains('win32')
    warning('Impossible to build from source on a 32-bit Windows Python install!')
  endif
endif

# ----------------
# Check backend
# ----------------

if meson.backend() != 'ninja'
  error('Ninja backend required')
endif

# ----------------
# Openmp
# ----------------

omp = dependency('openmp', required: false)

# ----------------
# Compiler
# ----------------

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')
# generator() doesn't accept compilers, only found programs - cast it.
cython = find_program(cy.cmd_array()[0])

# ------------------------
# check compiler versions
# ------------------------

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=9.1')
    error('tofu requires GCC >= 9.1')
  endif
elif cc.get_id() == 'clang' or cc.get_id() == 'clang-cl'
  if not cc.version().version_compare('>=15.0')
    error('tofu requires clang >= 15.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('tofu requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

if not cy.version().version_compare('>=3.0.8')
  error('tofu requires Cython >= 3.0.8')
endif

# ----------------
# numpy dependency
# ----------------

# From
# https://groups.google.com/g/cython-users/c/-AbF6gslN1U
# https://github.com/mesonbuild/meson/issues/9598#issuecomment-1662695303
# https://github.com/scipy/scipy/blob/main/scipy/meson.build#L30-L73

# Uses the `numpy-config` executable (or a user's numpy.pc pkg-config file).
# Will work for numpy>=2.0, hence not required (it'll be a while until 2.0 is
# our minimum supported version). Using this now to be able to detect the
# version easily for >=2.0.
_numpy_dep = dependency('numpy', required: false)
f2py_freethreading_arg = []
if _numpy_dep.found() and _numpy_dep.version().version_compare('>=2.1.0')
  f2py_freethreading_arg = ['--free-threading']
  message('f2py free-threading enabled')
else
  message('f2py free-threading disabled; need numpy >=2.1.0.')
  message('See https://github.com/mesonbuild/meson/issues/14651')
endif

# NumPy include directory - needed in all submodules
# The chdir is needed because within numpy there's an `import signal`
# statement, and we don't want that to pick up scipy's signal module rather
# than the stdlib module. The try-except is needed because when things are
# split across drives on Windows, there is no relative path and an exception
# gets raised. There may be other such cases, so add a catch-all and switch to
# an absolute path. Relative paths are needed when for example a virtualenv is
# placed inside the source tree; Meson rejects absolute paths to places inside
# the source tree.
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#   [properties]
#   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
#
# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

  # We do need an absolute path to feed to `cc.find_library` below
  _incdir_numpy_abs = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)
# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation.
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_nodepr_api)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

npymath_path = _incdir_numpy_abs / '..' / 'lib'
npymath_lib = cc.find_library('npymath', dirs: npymath_path)


# ------------------------
# -lm for C code
# ------------------------

# We need -lm for all C code (assuming it uses math functions, which is safe).
# For C++ it isn't needed, because libstdc++/libc++ is guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# https://mesonbuild.com/Python-module.html
py_dep = py.dependency()


# ----------------
# Directories
# ----------------

# installation dir of tofu, from the py install
tofu_dir = py.get_install_dir() / 'tofu'

# => will look into tofu for another meson.build
# local variables are accessible
# # then resume execution after this line
subdir('tofu')

# install_subdir('examples', install_dir: tofu_dir)
