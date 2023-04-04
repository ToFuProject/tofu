# -*- coding: utf-8 -*-


import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt


from . import _class8_etendue_los as _etendue_los
from ..geom import _comp_solidangles


# ###############################################################
# ###############################################################
#                       Main
# ###############################################################


def compute_vos(
    coll=None,
    key_diag=None,
    key_mesh=None,
    config=None,
    # parameters
    res=None,
    margin_par=None,
    margin_perp=None,
    # options
    add_points=None,
    # spectro-only
    rocking_curve_fw=None,
    rocking_curve_max=None,
    # bool
    convex=None,
    check=None,
    verb=None,
    plot=None,
    store=None,
):

    # ------------
    # check inputs

    (
        key,
        spectro,
        is2d,
        doptics,
        dcompute,
        analytical,
        numerical,
        res,
        margin_par,
        margin_perp,
        check,
        verb,
        plot,
        store,
    ) = _etendue_los._check(
        coll=coll,
        key_diag=key_diag,
        key_mesh=key_mesh,
        analytical=analytical,
        numerical=numerical,
        res=res,
        margin_par=margin_par,
        margin_perp=margin_perp,
        check=check,
        verb=verb,
        plot=plot,
        store=store,
    )

    if verb is True:
        msg = f"\nComputing etendue / los for diag '{key}':"
        print(msg)

    # ------------
    # sample mesh

    dsamp = coll.get_sample_mesh(
        key=key_mesh,
        res=res,
        mode='abs',
        grid=True,
        in_mesh=True,
        # non-used
        x0=None,
        x1=None,
        Dx0=Dx0,
        Dx1=Dx1,
        imshow=False,
        store=False,
        kx0=None,
        kx1=None,
    )

    # --------------
    # prepare optics

    doptics = coll.dobj['diagnostic'][key_diag]['doptics']

    # --------------
    # prepare optics

    dvos = {}
    for key_cam, v0 in dcompute.items():

        # --------------
        # loop on pixels

        # get temporary vos
        kvos = doptics[key_cam]['vos']

        for ii in range():

            # -----------------
            # get volume limits

            kdphi = coll.dobj['vos'][kvos]['dphi']
            if np.isnan():
                continue

            # get cross-section polygon
            kpc0, kpc1 = coll.dobj['vos'][kvos]['poly_cross']
            pcross0 = coll.ddata[kpc0]['data'][:, ii]
            pcross1 = coll.ddata[kpc1]['data'][:, ii]
            pcross = Path(np.array([pcross0, pcross1]).T)

            # get phi interval
            dphi = coll.ddata[kdphi]['data'][:, ii]

            # get ind
            ind = (
                dsamp['ind']['data']
                & pcross.contains_points(np.array([
                    dsamp['x0']['data'],
                    dsamp['x1']['data'],
                ]))
            )

            # ---------------------
            # loop on volume points

            if spectro:
                dvos[key_cam] = _vos_spectro(
                    x0=x0,
                    x1=x1,
                    ind=ind,
                    dphi=dphi,
                )

            else:
                dvos[key_cam] = _vos_broadband()

            # -----------
            # aggregate


    return dvos, dstore


# ###########################################################
# ###########################################################
#               Get domain
# ###########################################################


def _get_DRZPhi(
    coll=None,
    key_diag=None,
    key_cam=None,
    res=None,
):

    ptsx, ptsy, ptsz = coll




    return




# ###########################################################
# ###########################################################
#               Broadband
# ###########################################################


def _vos_broadband(
    x0=None,
    x1=None,
    ind=None,
    Dphi=None,
):

    # -----------------
    # loop on (r, z) points

    sa = np.full(())
    ir, iz = ind.nonzero()
    iru = np.unique(iru)

    for i0 in iru:

        nphi = int(np.ceil(rr*(Dphi[1] - Dphi[0]) / res))
        phi = np.linspace(Dphi[0], Dphi[1], nphi)
        xx = x0[i0] * np.cos(phi)
        yy = x0[i0] * np.sin(phi)
        zz = np.full((nphi,), np.nan)

        for i1 in iz[ir == i0]:

            zz[:] = x1[i1]

            out = calc_solidangle_apertures(
                # observation points
                pts_x=xx,
                pts_y=yy,
                pts_z=zz,
                # polygons
                apertures=None,
                detectors=None,
                # possible obstacles
                config=conf,
                # parameters
                summed=None,
                visibility=True,
                return_vector=False,
                return_flat_pts=None,
                return_flat_det=None,
            )

            import pdb; pdb.set_trace()     # DB
            sa[i0, i1] = out

    # -------------
    # format output

    dout = {
        'solid_angle_int': {
            'data': sa,
            'units': '',
            'dim': '',
            'quant': '',
            'name': '',
            'ref': '',
        },
    }

    return dout
