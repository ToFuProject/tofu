"""
ToFu: A python library for Tomography for Fusion research
=========================================================

Provides
  1. Objects to handle 3D geometry of diagnostics
  2. Built-in methods for computing key quantities, data treatment, inversions...
  3. Visualisation tools

How to use the documentation
----------------------------
Documentation is available in two forms:
  1. Docstrings provided with the code
  2. A loose standing reference guide, available from `the ToFu homepage <http://www.tofu.org>`_.


Available subpackages
---------------------
geom
    Geometry-handling, with dedicated objects, methods and functions
mesh
    Mesh and basis functions creation and handling, as well as 2D equilibrium storing
matcomp
    Computation of geometry matrix from outputs of both geom and mesh
treat
    Data-handling objects and methods for pre-treatment (visualisation, treatment...)
inv
    Inversion-regularisation algortihms, using outputs from matcomp and data, plus visualisation

Available modules
-----------------
defaults
    Store most default parameters of tofu
pathfile
    Provide a class for identification of all tofu objects, and functions for path and file handling
helper
    miscellaneous helper functions

Utilities
---------
test
    Run tofu unittests
show_config
    Show tofu build configuration
__version__
    tofu version string


Created on Wed May 18 2016

@author: didiervezinet
@author_email: didier.vezinet@gmail.com
"""
import sys
import warnings
if sys.version[0] == '2':
    from .version import __version__
elif sys.version[0] == '3':
    from .version import __version__

# For tests without display with nosetests
if not 'matplotlib.pyplot' in sys.modules:
    import matplotlib
    matplotlib.use('agg')
    del matplotlib

import tofu.pathfile as pathfile
import tofu.utils as utils

from tofu.utils import save, load, load_from_imas, calc_from_imas
import tofu._plot as _plot
import tofu.geom as geom
import tofu.data as data


# -------------------------------------
#   Try importing optional subpackages
# -------------------------------------

msg = None
dsub = dict.fromkeys(['imas2tofu', 'mag'])
for sub in dsub.keys():
    try:
        exec('import tofu.{0} as {0}'.format(sub))
        dsub[sub] = True
    except Exception as err:
        dsub[sub] = str(err)

# -------------------------------------
# If any error, populate warning and store error message
# -------------------------------------

lsubout = [sub for sub in dsub.keys() if dsub[sub] != True]
if len(lsubout) > 0:
    lsubout = ['tofu.{0}'.format(ss) for ss in lsubout]
    msg = "\nThe following subpackages are not available:"
    msg += "\n    - " + "\n    - ".join(lsubout)
    msg += "\n  => see tofu.dsub[<subpackage>] for details on error messages"
    warnings.warn(msg)

# -------------------------------------
# Add optional subpackages to __all__
# -------------------------------------

__all__ = ['pathfile','utils','_plot','geom','data']
for sub in dsub.keys():
    if dsub[sub] == True:
        __all__.append(sub)

# clean-up the mess
del sys, warnings, lsubout, sub, msg
