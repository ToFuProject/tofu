# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    Magnetic ripple
'''

import warnings
import traceback

try:
    try:
        from tofu.mag.mag_ripple import *
    except Exception:
        from .mag_ripple import *
    del warnings, traceback
except Exception as err:
    msg = str(traceback.format_exc())
    msg += "\n\n    => the optional sub-package is not usable\n"
    warnings.warn(msg)
    del msg, err

__all__ = ['mag_ripple']
