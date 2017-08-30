
import tests01_GGNew as test

Lf = dir(test)

for ff in Lf:
    if 'test' in ff:
        if hasattr(eval("test."+ff),'__call__'):
            print("    "+ff)
            eval("test."+ff+"()")







