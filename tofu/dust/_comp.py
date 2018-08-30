

import numpy as np

# ToFu-specific
import tofu.geom._GG as _GG



"""
def calc_directproblem(pts=None, r=None,
                       Ves=None, LStruct=None, res=None,
                       demiss=None, ani=False, axisym=True,
                       approx=True, out_coefonly=False,
                       Forbid=True, verb=True, Test=True):

    # Preformat
    if verb:
        s0 = "    Direct problem"
    nt = pts.shape[1]
    if Ves is not None:
        VPoly, VIn, Lim, VType = Ves.Poly, Ves.geom['VIn'], Ves.Lim, Ves.Type
        if LStruct is not None:
            LSPoly = [ss.Poly for ss in LStruct]
            LSVIn = [ss.geom['VIn'] for ss in LStruct]
            LSLim = [ss.Lim for ss in LStruct]
        else:
            LSPoly, LSVIn, LSLim = None, None, None
    else:
        VPoly, VIn, Lim, VType = None, None, None, None

    DV = [None, None, DPhi]
    pts, dV, ind, dVr = Ves.get_sampleV(res, DV=DV, dVMode='abs', ind=None, Out='(X,Y,Z)'):
    nR =

    dout = {''}
    if demiss is None or demiss=={}:
        sang = _GG.Dust_calc_SolidAngle(pos, r, pts,
                                        approx=approx, out_coefonly=out_coefonly,
                                        VType=VType, VPoly=VPoly, VIn=VIn,
                                        VLim=Lim, LSPoly=LSPoly, LSLim=LSLim,
                                        LSVIn=LSVIn, Forbid=Forbid, Test=Test):
        dout['sang':out[0]]
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
"""
