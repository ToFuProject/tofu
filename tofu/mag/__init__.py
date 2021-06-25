# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    Magnetic field lines
'''

import warnings
import traceback

msg, err = None, None
try:
    import imas
except Exception as err:
    if str(err) == 'imas not available':
        msg = ""
        msg += "\n\nIMAS python API issue\n"
        msg += "imas could not be imported into tofu ('import imas' failed):\n"
        msg += "  - it may not be installed (optional dependency)\n"
        msg += "  - or you not have loaded the good working environment\n\n"
        msg += "    => the optional sub-package tofu.mag is not usable\n"
    else:
        msg = str(traceback.format_exc())
        msg += "\n\n    => the optional sub-package tofu.mag is not usable\n"
    raise Exception(msg)

try:
    import pywed
except Exception:
    msg = "pywed not available => no tofu.mag"
    raise Exception(msg)

try:
    from tofu.mag.magFieldLines import *
except Exception:
    try:
        from .magFieldLines import *
    except Exception:
        msg = "Could not import"
        raise Exception(msg)

del warnings, traceback, pywed, imas
del msg, err

__all__ = ['MagFieldLines']
