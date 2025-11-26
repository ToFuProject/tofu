# ################################
#     Direct py modules
# ################################

tofu_geom_dir = tofu_dir / 'geom'

py.install_sources([
    '__init__.py',
],
subdir: tofu_geom_dir,
)


# ################################
# No extension modules => pure dirs
# ################################

pure_subdirs = [
  'inputs',
]

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: tofu_geom_dir)
endforeach


# #################################
#     Compile / cython args
# #################################


cython_args = [
  '-Xboundscheck=True',
]
cython_args2 = cython_args
if omp.found()
  cython_args2 += ['--compile-time-env', 'TOFU_OPENMP_ENABLED=True'] 
endif


include_dirs = [_incdir_numpy_abs]


compile_args = [
  '-O3',
  '-Wall',
  '-fno-wrapv',
  'language_level="3"',
]


# ################################
# local extensions
# ################################


py.extension_module(
  '_GG',
  '_GG.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  # extra_args : compile_args,
  install: true,
  # extra_compile_args=extra_compile_args,
  # extra_link_args=extra_link_args,
)
py.extension_module(
  '_basic_geom_tools',
  '_basic_geom_tools.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  '_distance_tools',
  '_distance_tools.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  '_sampling_tools',
  '_sampling_tools.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  '_raytracing_tools',
  '_raytracing_tools.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  '_vignetting_tools',
  '_vignetting_tools.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  '_chained_list',
  '_chained_list.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  '_sorted_set',
  '_sorted_set.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args,
  install: true,  # true
)
py.extension_module(
  '_openmp_tools',
  '_openmp_tools.pyx',
  subdir: 'tofu/geom/',
  include_directories: include_dirs,
  dependencies : [py_dep, omp],
  cython_args : cython_args2,
  install: true,  # true
)
