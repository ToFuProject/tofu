"""
The intermediate module automatically choosing whether to use the Py2 or Py3
version of the cython-compiled numerical core of tofu.geom
"""

import sys

if sys.version[0] == '2':
    try:
        from tofu.geom._GG02_LM import *
    except Exception:
        from _GG02_LM import *
elif sys.version[0] == '3':
    try:
        from tofu.geom._GG03_LM import *
    except Exception:
        from _GG03_LM import *
