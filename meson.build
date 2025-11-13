# ------------------
# project
# ------------------

# refs:
# https://github.com/scipy/scipy/blob/main/meson.build
# https://github.com/numpy/numpy/blob/main/meson.build
# https://github.com/silx-kit/silx/blob/main/meson.build

project(
  'tofu',
  'c', 'cpp', 'cython',
  license: 'MIT',
  meson_version: '>=1.8.3',  # version in vendored-meson
  version: run_command(
    ['git', 'describe'],
    # ['tofu/_version.py', '--wheel'],
    check:true,
  ).stdout().strip(),
  default_options: [
    # 'buildtype=plain',
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c11',
    'cpp_std=c++17',
    'pkgconfig.relocatable=true',
  ],
)

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

# omp = dependency('openmp', required: get_option('use_openmp'))
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

tofu_dir = py.get_install_dir() / 'tofu'

# subdir('src/tofu')
# install_subdir('examples', install_dir: tofu_dir)


# --------------------------
# Cython args for extensions
# --------------------------


cython_args = [
  '-Xboundscheck=True',
]
cython_args2 = cython_args
if omp.found()
  cython_args2 += ['--compile-time-env', 'TOFU_OPENMP_ENABLED=True'] 
endif


compile_args = [
  '-O3',
  '-Wall',
  '-fno-wrapv',
  'language_level="3"',
]


# ----------------
# Extensions
# ----------------


py.extension_module(
  'tofu.geom._GG',
  'tofu/geom/_GG.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  # extra_args : compile_args,
  install: true,
  # extra_compile_args=extra_compile_args,
  # extra_link_args=extra_link_args,
)
py.extension_module(
  'tofu.geom._basic_geom_tools',
  'tofu/geom/_basic_geom_tools.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  'tofu.geom._distance_tools',
  'tofu/geom/_distance_tools.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  'tofu.geom._sampling_tools',
  'tofu/geom/_sampling_tools.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  'tofu.geom._raytracing_tools',
  'tofu/geom/_raytracing_tools.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  'tofu.geom._vignetting_tools',
  'tofu/geom/_vignetting_tools.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  'tofu.geom._chained_list',
  'tofu/geom/_chained_list.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  'tofu.geom._sorted_set',
  'tofu/geom/_sorted_set.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  'tofu.geom._openmp_tools',
  'tofu/geom/_openmp_tools.pyx',
  subdir: 'tofu/geom/',
  dependencies : [py_dep, omp],
  cython_args : cython_args2,
  install: true,  # true
)
