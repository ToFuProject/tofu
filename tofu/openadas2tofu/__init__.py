# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
The openadas-compatibility module of tofu

"""
import traceback

try:
    try:
        from tofu.openadas2tofu._requests import *
        from tofu.openadas2tofu._read_files import *
    except Exception:
        from ._requests import *
        from ._read_files import *
except Exception as err:
    msg = str(traceback.format_exc())
    msg += "\n\n    => the optional sub-package tofu.imas2tofu is not usable\n"
    raise Exception(msg)

del traceback
