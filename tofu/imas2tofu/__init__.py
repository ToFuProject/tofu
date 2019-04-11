# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
The imas-compatibility module of tofu

"""

try:
    import imas
except Exception:
    msg = "IMAS python API issue\n"
    msg += "imas could not be imported into tofu ('import imas' failed):\n"
    msg += "  - it may not be installed (optional dependency)\n"
    msg += "  - or you not have loaded the good working environment\n\n"
    msg += "    => the optional sub-package tofu.imas2tofu is not usable\n"
    raise Exception(msg)

try:
    import tofu.imas2tofu._load as load
except Exception:
    from . import _load as load


__all__ = ['load']
