# -*- coding: utf-8 -*-
"""
The nist-compatibility module of tofu

"""
import traceback

try:
    try:
        from tofu.nist2tofu._requests import *
    except Exception:
        from ._requests import *
except Exception as err:
    msg = str(traceback.format_exc())
    msg += "\n\n    => the optional sub-package tofu.nist2tofu is not usable\n"
    raise Exception(msg)

del traceback
