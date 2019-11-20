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
  2. Online sphinx-generated documentation, available from
`ToFu's homepage <https://tofuproject.github.io/tofu/>`_.


Available subpackages
---------------------
`geom <tofu.geom.html>`_
    Geometry classes to model the 3D geometry (vacuum vessel, structural
    elements, LOS, etc.)
`data <tofu.data.html>`_
    Data-handling classes (storing, processing, plotting, etc.)
`dumpro <tofu.dumpro.html>`_
    Package for dust movie processing
`dust <tofu.dust.html>`_
    Dust module
imas2tofu
    The imas-compatibility module of tofu (optional)
mag
    Magnetic field lines package (optional)

Available modules
-----------------
_plot
    Module providing a basic routine for plotting a shot overview
defaults
    Store most default parameters of tofu
pathfile
    Provide a class for identification of all tofu objects, and functions for path and file handling
utils
    Miscellaneous helper functions

Utilities
---------
tests
    tofu's unit-tests
__version__
    tofu version string
"""

import sys
import warnings
from .version import __version__

# For tests without display with nosetests
if 'matplotlib.pyplot' not in sys.modules.keys():
    import matplotlib
    matplotlib.use('agg')
    del matplotlib

import tofu.pathfile as pathfile
import tofu.utils as utils

from tofu.utils import save, load, load_from_imas, calc_from_imas
import tofu._plot as _plot
import tofu.geom as geom
import tofu.data as data


try:
    import tofu.imas2tofu as imas2tofu
    okimas2tofu = True
except Exception as err:
    warnings.warn(str(err))
    okimas2tofu = False

try:
    import tofu.mag as mag
    okmag = True
except Exception as err:
    warnings.warn(str(err))
    okmag = False

#import tofu.dust as dust


__all__ = ['pathfile','utils','_plot','geom','data']
if okimas2tofu:
    __all__.append('imas2tofu')
if okmag:
    __all__.append('mag')


del sys, warnings, okimas2tofu, okmag

#__all__.extend(['geom', 'mesh', 'matcomp', 'data', 'inv'])

#__name__ = ""
#__date__ = "$Mar 05, 2014$"
#__copyright__ = ""
#__license__ = ""
#__url__ = ""
#__path__ =
