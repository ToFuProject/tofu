# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
The imas-compatibility module of tofu

"""
import warnings
import traceback

try:
    try:
        from tofu.imas2tofu._core import *
    except Exception:
        from ._core import *
    del warnings, traceback
except Exception as err:
    if str(err) == 'imas not available':
        msg = ""
        msg += "\n\nIMAS python API issue\n"
        msg += "imas could not be imported into tofu ('import imas' failed):\n"
        msg += "  - it may not be installed (optional dependency)\n"
        msg += "  - or you not have loaded the good working environment\n\n"
        msg += "    => the optional sub-package tofu.imas2tofu is not usable\n"
    else:
        msg = str(traceback.format_exc())
        msg += "\n\n    => the optional sub-package tofu.imas2tofu is not usable\n"
    warnings.warn(msg)
    del msg, err

__all__ = ['MultiIDSLoader', 'load_Config', 'load_Plasma2D',
           'load_Cam', 'load_Data']
