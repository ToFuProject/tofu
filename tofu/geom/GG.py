
import sys

if sys.version[0]=='2':
    from GG02 import *
elif sys.version[0]=='3':
    from GG03 import *
else:
    raise Exception("Pb. with python version : "+sys.version)



