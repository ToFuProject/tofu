# ################################
#     Direct py modules
# ################################

py.install_sources([
    '__init__.py',
    'version.py',
    'py.typed',
    'pathfile.py',
    'utils.py',
    '_plot.py',
    'utils.py',
],
subdir: 'tofu',  # Folder relative to site-packages to install to
)


# ################################
# No extension modules => pure dirs
# ################################

pure_subdirs = [
  #'geom',
  #'data',
  'spectro',
  #'tests',
  # 'benchmarks',    # needs geom
  'physics_tools',
  # 'omas2tofu',     # needs data
  'omas2tofu',
  'imas2tofu',
  'nist2tofu',
  'openadas2tofu',
  'tomotok2tofu',
  #'mag',
]

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: tofu_dir)
endforeach

# ################################
# extension modules => subdir with meson.build
# ################################

# subdir('geom')
