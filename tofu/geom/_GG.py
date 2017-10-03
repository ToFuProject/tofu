
import sys

if sys.version[0]=='2':
    try:
        from tofu.geom._GG02 import *
    except Exception:
        from _GG02 import *
elif sys.version[0]=='3':
    try:
        from tofu.geom._GG03 import *
    except Exception:
        from _GG03 import *
else:
    raise Exception("Pb. with python version : "+sys.version)



