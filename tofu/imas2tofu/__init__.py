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
    from tofu.imas2tofu._configloader import load_Config
    from tofu.imas2tofu._plasma2Dloader import load_Plasma2D
    from tofu.imas2tofu._diagloader import load_Diag
except Exception:
    from ._configloader import load_Config
    from ._plasma2Dloader import load_Plasma2D
    from ._diagloader import load_Diag


__all__ = ['load_Config','load_Plasma2D','load_Diag']
