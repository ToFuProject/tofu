

import numpy as np

# ToFu-specific
import tofu.geom._GG as _GG




def calc_directproblem(pts=None, r=None,
                       Ves=None, LStruct=None, res=None,
                       demiss=None, ani=False, axisym=True,
                       verb=True):

    # Preformat
    if verb:
        s0 = "    Direct problem"

    nt = pts.shape[1]

    DPhi = 
    DV = [None, None, DPhi]
    pts, dV, ind, dVr = Ves.get_sampleV(res, DV=DV, dVMode='abs', ind=None, Out='(X,Y,Z)'):
    nR = 

    if demiss is None or demiss=={}:
        dout = _GG.calc_SolidAngleView()
    elif not ani and axisym:
        dout = _GG.calc_SolidAngleView_emissIsoAxisym()
    else:
        dout = _GG.calc_SolidAngleView_emiss()
    else:
        raise(Exception, "Not coded yet !")




    for ii in range(0,nt):
        if verb:
            print(s0+": step {0}/{1}".format(ii,nt))
        for jj in range(0,nR):
            pp = 
            vis = 
            sa = 



    lpolyCross, lpolyHor, gridCross, gridHor = out[:4]
    saCross, saHor, volIn, saIn = out[4:8]
    contribCross, contribHor, powIn









