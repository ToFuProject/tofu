# -*- coding: utf-8 -*-
#! /usr/bin/python


"""
The openadas-compatibility module of tofu

"""


import traceback


try:
    try:
        from tofu.tomotok2tofu._core import *
    except Exception:
        from ._core import *
except Exception as err:
    msg = str(traceback.format_exc())
    msg += "\n\n    => optional sub-package tofu.tomotok2tofu not usable\n"
    raise Exception(msg)

del traceback
