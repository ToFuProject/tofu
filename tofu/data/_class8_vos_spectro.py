# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np


from . import _class8_equivalent_apertures as _equivalent_apertures
# from ..geom import _comp_solidangles


# ###########################################################
# ###########################################################
#               Main
# ###########################################################


def _vos(
    x0=None,
    x1=None,
    ind=None,
    dphi=None,
    deti=None,
    lap=None,
    res=None,
    config=None,
    visibility=None,
    # output
    key_cam=None,
    dvos=None,
    sli=None,
    ii=None,
    bool_cross=None,
    path_hor=None,
    # timing
    timing=None,
    dt1111=None,
    dt2222=None,
    dt3333=None,
    dt4444=None,
):

    # --------------------------
    # prepare points and indices

    ir, iz = ind.nonzero()
    iru = np.unique(ir)
    izru = [iz[ir == i0] for i0 in iru]

    nphi = np.ceil(x0[ir]*(dphi[1] - dphi[0]) / res).astype(int)

    irf = np.repeat(ir, nphi)
    izf = np.repeat(iz, nphi)
    phi = np.concatenate(tuple([
        np.linspace(dphi[0], dphi[1], nn) for nn in nphi
    ]))

    xx = x0[irf] * np.cos(phi)
    yy = x0[irf] * np.sin(phi)
    zz = x1[izf]

    # ---------------
    # prepare lambda

    lamb = None

    # -------------
    # get

    for pp in pts:

        # get equivalent aperture
        equivalent aperture(optics => crystal)

        # get angles
        angles =

        # get lambda fro rocking curve
        lamb =

        # get power ratio
        pow_ratio =

        # get rays
        rays =

        # get image on camera
        x0 =
        x1 =

        # Interpolate per pixel
        vv =

        # Integrate per pixel
        vv * ds * dlamb * dV * dsa



    # ------------
    # get indices

    if timing:
        t0 = dtm.datetime.now()     # DB
        out, dt1, dt2, dt3 = out

    for ii, i0 in enumerate(iru):
        ind0 = irf == i0
        for i1 in izru[ii]:
            ind = ind0 & (izf == i1)
            bool_cross[i0 + 1, i1 + 1] = np.any(out[0, ind] > 0.)

    # timing
    if timing:
        dt4444 += (dtm.datetime.now() - t0).total_seconds()
        dt1111 += dt1
        dt2222 += dt2
        dt3333 += dt3

        return dt1111, dt2222, dt3333, dt4444
    else:
        return
