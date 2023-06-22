# -*- coding: utf-8 -*-


import numpy as np
import datastock as ds


# ##############################################################
# ##############################################################
#           Finding reflection points
# ##############################################################


def _get_ptsvect(
    coll=None,
    key=None,
    asplane=None,
    fast=None,
    isnorm=None,
):
    # ------------
    # check inputs

    fast = ds._generic_check._check_var(
        fast, 'fast',
        types=bool,
        default=False,
    )

    isnorm = ds._generic_check._check_var(
        isnorm, 'isnorm',
        types=bool,
        default=False,
    )

    # ---------
    # key

    key, cls = coll.get_optics_cls(optics=key)
    key, cls = key[0], cls[0]
    dgeom = coll.dobj[cls][key]['dgeom']

    # asplane
    asplane = ds._generic_check._check_var(
        asplane, 'asplane',
        types=bool,
        default=False,
    )

    # -------------------
    #     Planar
    # -------------------

    if dgeom['type'] == 'planar' or asplane is True:

        if dgeom['extenthalf'] is None:
            x0max, x1max = None, None
        else:
            x0max, x1max = dgeom['extenthalf']

        if fast is True:
            ptsvect = _get_ptsvect_plane_x01_fast(
                plane_cent=dgeom['cent'],
                plane_nin=dgeom['nin'],
                plane_e0=dgeom['e0'],
                plane_e1=dgeom['e1'],
            )

        else:
            ptsvect = _get_ptsvect_plane(
                plane_cent=dgeom['cent'],
                plane_nin=dgeom['nin'],
                plane_e0=dgeom['e0'],
                plane_e1=dgeom['e1'],
                # limits
                x0max=x0max,
                x1max=x1max,
                # isnorm
                isnorm=isnorm,
            )

    # ----------------
    #   Cylindrical
    # ----------------

    elif dgeom['type'] == 'cylindrical':

        iplan = np.isinf(dgeom['curve_r']).nonzero()[0][0]
        icurv = 1 - iplan
        eax = ['e0', 'e1'][iplan]
        erot = ['e0', 'e1'][icurv]

        rc = dgeom['curve_r'][icurv]
        rcs = np.sign(rc)
        rca = np.abs(rc)

        minmax = np.maximum if rcs > 0 else np.minimum

        def ptsvect(
            # pts
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # vect
            vect_x=None,
            vect_y=None,
            vect_z=None,
            # surface
            O=dgeom['cent'] + dgeom['nin'] * rc,
            rcs=rcs,
            rca=rca,
            minmax=minmax,
            eax=dgeom[eax],
            erot=dgeom[erot],
            iplan=iplan,
            # limits
            thetamax=dgeom['extenthalf'][icurv],
            xmax=dgeom['extenthalf'][iplan],
            # local coordinates
            nin=dgeom['nin'],
            # isnorm
            isnorm=isnorm,
            # return
            strict=None,
            return_x01=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """
            # normalize vect
            if isnorm is False:
                vn = np.sqrt(vect_x**2 + vect_y**2 + vect_z**2)
                vect_x = vect_x / vn
                vect_y = vect_y / vn
                vect_z = vect_z / vn

            # ------------------------------------------
            # Get local coordinates of reflection points

            # get parameters for k
            OAzx = (pts_y - O[1])*eax[2] - (pts_z - O[2])*eax[1]
            OAzy = (pts_z - O[2])*eax[0] - (pts_x - O[0])*eax[2]
            OAzz = (pts_x - O[0])*eax[1] - (pts_y - O[1])*eax[0]
            OAz2 = OAzx**2 + OAzy**2 + OAzz**2

            eazx = vect_y*eax[2] - vect_z*eax[1]
            eazy = vect_z*eax[0] - vect_x*eax[2]
            eazz = vect_x*eax[1] - vect_y*eax[0]
            ez2 = eazx**2 + eazy**2 + eazz**2

            OAzez = OAzx*eazx + OAzy*eazy + OAzz*eazz
            iok = (
                np.isfinite(ez2)
                & np.isfinite(OAz2)
                & np.isfinite(OAzez)
            )

            # Prepare D, theta, xx
            shape = OAzez.shape
            (
                Dx, Dy, Dz, vrx, vry, vrz,
                angle, theta, xx, kk,
            ) = _common_prepare(shape)

            # get k
            delta = (OAzez**2 - ez2*(OAz2-rca**2))[iok]
            ipos = delta >= 0
            iok[iok] = ipos
            sol0 = (-OAzez[iok] - np.sqrt(delta[ipos])) / ez2[iok]
            sol1 = (-OAzez[iok] + np.sqrt(delta[ipos])) / ez2[iok]
            kk[iok] = minmax(sol0, sol1)

            if np.any(iok):
                Dx = pts_x + kk*vect_x
                Dy = pts_y + kk*vect_y
                Dz = pts_z + kk*vect_z

                ODx = Dx[iok] - O[0]
                ODy = Dy[iok] - O[1]
                ODz = Dz[iok] - O[2]

                xxi = ODx * eax[0] + ODy * eax[1] + ODz * eax[2]

                nox = ODx - xxi * eax[0]
                noy = ODy - xxi * eax[1]
                noz = ODz - xxi * eax[2]
                nn = np.sqrt(nox**2 + noy**2 + noz**2)
                nox = rcs * nox / nn
                noy = rcs * noy / nn
                noz = rcs * noz / nn


                # ODzx = (Dy[iok] - O[1])*eax[2] - (Dz[iok] - O[2])*eax[1]
                # ODzy = (Dz[iok] - O[2])*eax[0] - (Dx[iok] - O[0])*eax[2]
                # ODzz = (Dx[iok] - O[0])*eax[1] - (Dy[iok] - O[1])*eax[0]
                # ODzn = np.sqrt(ODzx**2 + ODzy**2 + ODzz**2)

                # nox = -rcs*(ODzy * eax[2] - ODzz * eax[1]) / ODzn
                # noy = -rcs*(ODzz * eax[0] - ODzx * eax[2]) / ODzn
                # noz = -rcs*(ODzx * eax[1] - ODzy * eax[0]) / ODzn

                # scalar product (for angle + reflection)
                if np.isscalar(vect_x):
                    scavn = -(vect_x*nox + vect_y*noy + vect_z*noz)

                    # get vect_reflect
                    vrx[iok] = vect_x + 2.*scavn * nox
                    vry[iok] = vect_y + 2.*scavn * noy
                    vrz[iok] = vect_z + 2.*scavn * noz

                else:
                    scavn = -(
                        vect_x[iok]*nox + vect_y[iok]*noy + vect_z[iok]*noz
                    )
                    # get vect_reflect
                    vrx[iok] = vect_x[iok] + 2.*scavn * nox
                    vry[iok] = vect_y[iok] + 2.*scavn * noy
                    vrz[iok] = vect_z[iok] + 2.*scavn * noz

                angle[iok] = -np.arcsin(scavn)

                # x0, x1
                if strict is True or return_x01 is True:

                    theta[iok] =  rcs * np.arctan2(
                        nox*erot[0] + noy*erot[1] + noz*erot[2],
                        -nox*nin[0] - noy*nin[1] - noz*nin[2],
                    )

                    xx[iok] = xxi

                    if strict is True:
                        iout = (np.abs(theta) > thetamax) | (np.abs(xx) > xmax)

                        if np.any(iout):
                            Dx[iout] = np.nan
                            Dy[iout] = np.nan
                            Dz[iout] = np.nan
                            vrx[iout] = np.nan
                            vry[iout] = np.nan
                            vrz[iout] = np.nan
                            angle[iout] = np.nan
                            iok[iout] = False

                # enforce normalization
                if isnorm is False:
                    vnorm = np.sqrt(vrx[iok]**2 + vry[iok]**2 + vrz[iok]**2)
                    vrx[iok] = vrx[iok] / vnorm
                    vry[iok] = vry[iok] / vnorm
                    vrz[iok] = vrz[iok] / vnorm

            # return
            if return_x01:
                if iplan == 0:
                    return Dx, Dy, Dz, vrx, vry, vrz, angle, iok, xx, theta
                else:
                    return Dx, Dy, Dz, vrx, vry, vrz, angle, iok, theta, xx
            else:
                return Dx, Dy, Dz, vrx, vry, vrz, angle, iok

    # ----------------
    #   Spherical
    # ----------------

    elif dgeom['type'] == 'spherical':

        rc = dgeom['curve_r'][0]
        rcs = np.sign(rc)
        rca = np.abs(rc)

        def ptsvect(
            # pts
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # vect
            vect_x=None,
            vect_y=None,
            vect_z=None,
            # surface
            O=dgeom['cent'] + dgeom['nin'] * rc,
            rcs=rcs,
            rca=rca,
            # limits
            dthetamax=dgeom['extenthalf'][1],
            phimax=dgeom['extenthalf'][0],
            # local coordinates
            nin=dgeom['nin'],
            e0=dgeom['e0'],
            e1=dgeom['e1'],
            # isnorm
            isnorm=isnorm,
            # return
            strict=None,
            return_x01=None,
            # timing
            dt=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """
            # normalize vect
            if isnorm is False:
                vn = np.sqrt(vect_x**2 + vect_y**2 + vect_z**2)
                vect_x = vect_x / vn
                vect_y = vect_y / vn
                vect_z = vect_z / vn

            # ------------------------------------------
            # Get local coordinates of reflection points

            # get parameters for k
            OAx = pts_x - O[0]
            OAy = pts_y - O[1]
            OAz = pts_z - O[2]
            OA2 = OAx**2 + OAy**2 + OAz**2
            OAe = OAx * vect_x + OAy * vect_y + OAz * vect_z

            iok = np.isfinite(OA2) & np.isfinite(OAe)

            # Prepare D, theta, xx
            shape = OAe.shape
            (
                Dx, Dy, Dz, vrx, vry, vrz,
                angle, phi, dtheta, kk,
            ) = _common_prepare(shape)

            # get k
            delta = (OAe**2 - (OA2-rca**2))[iok]
            ipos = delta >= 0
            iok[iok] = ipos
            sol0 = -OAe[iok] - np.sqrt(delta[ipos])
            sol1 = -OAe[iok] + np.sqrt(delta[ipos])
            kk[iok] = np.maximum(sol0, sol1)

            if np.any(iok):
                Dx = pts_x + kk*vect_x
                Dy = pts_y + kk*vect_y
                Dz = pts_z + kk*vect_z

                ODn = np.sqrt((Dx - O[0])**2 + (Dy - O[1])**2 + (Dz - O[2])**2)
                nox = rcs * (Dx[iok] - O[0]) / ODn[iok]
                noy = rcs * (Dy[iok] - O[1]) / ODn[iok]
                noz = rcs * (Dz[iok] - O[2]) / ODn[iok]

                # scalar product (for angle + reflection)
                if np.isscalar(vect_x):
                    scavn = -(vect_x*nox + vect_y*noy + vect_z*noz)

                    # get vect_reflect
                    vrx[iok] = vect_x + 2.*scavn * nox
                    vry[iok] = vect_y + 2.*scavn * noy
                    vrz[iok] = vect_z + 2.*scavn * noz

                else:
                    scavn = -(
                        vect_x[iok]*nox + vect_y[iok]*noy + vect_z[iok]*noz
                    )
                    # get vect_reflect
                    vrx[iok] = vect_x[iok] + 2.*scavn * nox
                    vry[iok] = vect_y[iok] + 2.*scavn * noy
                    vrz[iok] = vect_z[iok] + 2.*scavn * noz

                angle[iok] = -np.arcsin(scavn)

                # x0, x1
                if strict is True or return_x01 is True:
                    dtheta[iok] = rcs * np.arcsin(
                        nox*e1[0] + noy*e1[1] + noz*e1[2]
                    )
                    phi[iok] = rcs * np.arcsin(
                        (nox*e0[0] + noy*e0[1] + noz*e0[2])
                        / np.cos(dtheta[iok])
                    )

                    if strict is True:
                        iout = (
                            (np.abs(dtheta) > dthetamax)
                            | (np.abs(phi) > phimax)
                        )

                        if np.any(iout):
                            Dx[iout] = np.nan
                            Dy[iout] = np.nan
                            Dz[iout] = np.nan
                            vrx[iout] = np.nan
                            vry[iout] = np.nan
                            vrz[iout] = np.nan
                            angle[iout] = np.nan
                            iok[iout] = False

                # enforce normalization
                if isnorm is False:
                    vnorm = np.sqrt(vrx[iok]**2 + vry[iok]**2 + vrz[iok]**2)
                    vrx[iok] = vrx[iok] / vnorm
                    vry[iok] = vry[iok] / vnorm
                    vrz[iok] = vrz[iok] / vnorm

            # return
            if return_x01:
                return Dx, Dy, Dz, vrx, vry, vrz, angle, iok, phi, dtheta
            else:
                return Dx, Dy, Dz, vrx, vry, vrz, angle, iok

    # ----------------
    #   Toroidal
    # ----------------

    elif dgeom['type'] == 'toroidal':

        raise NotImplementedError()

    return ptsvect


# ################################################################
# ################################################################
#                   preparation routine
# ################################################################


def _get_ptsvect_plane(
    plane_cent=None,
    plane_nin=None,
    plane_e0=None,
    plane_e1=None,
    # limits
    x0max=None,
    x1max=None,
    # isnorm
    isnorm=None,
):

    def ptsvect(
        pts_x=None,
        pts_y=None,
        pts_z=None,
        # pts
        vect_x=None,
        vect_y=None,
        vect_z=None,
        # surface
        cent=plane_cent,
        nin=plane_nin,
        e0=plane_e0,
        e1=plane_e1,
        # limits
        x0max=x0max,
        x1max=x1max,
        #isnorm
        isnorm=isnorm,
        # return
        strict=None,
        return_x01=None,
    ):
        """
        """

        # ------------------------------------------
        # Get local coordinates of reflection points

        # normalize vect
        if isnorm is False:
            vn = np.sqrt(vect_x**2 + vect_y**2 + vect_z**2)
            vect_x = vect_x / vn
            vect_y = vect_y / vn
            vect_z = vect_z / vn

        # get parameters for k
        scavn = vect_x*nin[0] + vect_y*nin[1] + vect_z*nin[2]

        kk = (
            (
                (cent[0] - pts_x)*nin[0]
                + (cent[1] - pts_y)*nin[1]
                + (cent[2] - pts_z)*nin[2]
            )
            / scavn
        )

        # negative => wrong side or both sides
        if np.any(kk < 0):
            Dx, Dy, Dz = None, None, None
            vrx, vry, vrz = None, None, None
            angle, iok = None, None
            if return_x01:
                x0, x1 = None, None
                return Dx, Dy, Dz, vrx, vry, vrz, angle, iok, x0, x1
            else:
                return Dx, Dy, Dz, vrx, vry, vrz, angle, iok

        # get D
        Dx = pts_x + kk * vect_x
        Dy = pts_y + kk * vect_y
        Dz = pts_z + kk * vect_z

        # get vect_reflect
        vrx = vect_x - 2.*scavn * nin[0]
        vry = vect_y - 2.*scavn * nin[1]
        vrz = vect_z - 2.*scavn * nin[2]

        # angles
        angle = -np.arcsin(scavn)

        # x0, x1
        if strict is True or return_x01 is True:
            x0 = (
                (Dx - cent[0])*e0[0]
                + (Dy - cent[1])*e0[1]
                + (Dz - cent[2])*e0[2]
            )
            x1 = (
                (Dx - cent[0])*e1[0]
                + (Dy - cent[1])*e1[1]
                + (Dz - cent[2])*e1[2]
            )

            if strict is True:
                iout = (np.abs(x0) > x0max) | (np.abs(x1) > x1max)

                if np.any(iout):
                    Dx[iout] = np.nan
                    Dy[iout] = np.nan
                    Dz[iout] = np.nan
                    vrx[iout] = np.nan
                    vry[iout] = np.nan
                    vrz[iout] = np.nan
                    angle[iout] = np.nan

        iok = np.isfinite(Dx)
        if return_x01:
            return Dx, Dy, Dz, vrx, vry, vrz, angle, iok, x0, x1
        else:
            return Dx, Dy, Dz, vrx, vry, vrz, angle, iok
    return ptsvect


def _get_ptsvect_plane_x01_fast(
    plane_cent=None,
    plane_nin=None,
    plane_e0=None,
    plane_e1=None,
):

    def ptsvect(
        pts_x=None,
        pts_y=None,
        pts_z=None,
        # pts
        vect_x=None,
        vect_y=None,
        vect_z=None,
        # surface
        cent=plane_cent,
        nin=plane_nin,
        e0=plane_e0,
        e1=plane_e1,
    ):
        """
        Faster version to return only x0, x1
        Used for vos_spectro
        assumed normalized vectors
        """

        # ------------------------------------------
        # Get local coordinates of reflection points

        # get parameters for k
        scavn = vect_x*nin[0] + vect_y*nin[1] + vect_z*nin[2]

        kk = (
            (
                (cent[0] - pts_x)*nin[0]
                + (cent[1] - pts_y)*nin[1]
                + (cent[2] - pts_z)*nin[2]
            )
            / scavn
        )

        # negative => wrong side or both sides
        if np.any(kk < 0):
            return None, None, None, None, None

        # get D
        Dx = pts_x + kk * vect_x
        Dy = pts_y + kk * vect_y
        Dz = pts_z + kk * vect_z

        # x0, x1
        return (
            (
                (Dx - cent[0])*e0[0]
                + (Dy - cent[1])*e0[1]
                + (Dz - cent[2])*e0[2]
            ),
            (
                (Dx - cent[0])*e1[0]
                + (Dy - cent[1])*e1[1]
                + (Dz - cent[2])*e1[2]
            ),
            Dx, Dy, Dz,
        )

    return ptsvect


# ###############################################################
# ###############################################################
#           Common formulas
# ###############################################################


def _common_coefs(rc=None, OAz2=None, OAzez=None, ez2=None):
    return ez2, 2.*OAzez, OAz2 - rc**2


def _common_prepare(shape):
    return (
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
    )


def _common_kE(C0=None, C1=None, C2=None):
    kk = np.roots(np.r_[C0, C1, C2])
    kk = np.real(kk[np.isreal(kk)])

    # outgoing is necessarily the maxium k
    if np.any(kk > 0.):
        return np.max(kk[kk > 0.])
    else:
        return np.nan


# #################################################################
# #################################################################
#           Debug
# #################################################################


def _debug_cylindrical(
    pt_x=None,
    pt_y=None,
    pt_z=None,
    pts_x=None,
    pts_y=None,
    pts_z=None,
    kk=None,
    O=None,
    rc=None,
    ABx=None,
    ABy=None,
    ABz=None,
    nox=None,
    noy=None,
    noz=None,
    nin=None,
    eax=None,
    xx=None,
    theta=None,
    xmax=None,
    thetamax=None,
    check=None,
    **kwdargs,
):

    # get E
    Ex = pt_x + kk * ABx
    Ey = pt_y + kk * ABy
    Ez = pt_z + kk * ABz

    # get D
    Dx = O[0] + xx*eax[0] + rc*nox
    Dy = O[1] + xx*eax[1] + rc*noy
    Dz = O[2] + xx*eax[2] + rc*noz

    # get e1
    e1 = np.cross(nin, eax)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

    ax.plot(
        np.r_[pt_x, pts_x],
        np.r_[pt_y, pts_y],
        np.r_[pt_z, pts_z],
        '.-k',
    )

    for ii in range(len(kk)):
        l0, = ax.plot(
            np.r_[pt_x, Dx[ii], pts_x],
            np.r_[pt_y, Dy[ii], pts_y],
            np.r_[pt_z, Dz[ii], pts_z],
            '.-',
        )
        ax.plot(
            np.r_[Ex[ii], Dx[ii]],
            np.r_[Ey[ii], Dy[ii]],
            np.r_[Ez[ii], Dz[ii]],
            ls='--',
            marker='.',
            c=l0.get_color(),
        )

    # cents
    ax.plot(
        np.r_[O[0], O[0] - rc*nin[0]],
        np.r_[O[1], O[1] - rc*nin[1]],
        np.r_[O[2], O[2] - rc*nin[2]],
        'ob',
    )

    # axe
    ax.plot(
        O[0] + xmax*np.r_[-1, 1] * eax[0],
        O[1] + xmax*np.r_[-1, 1] * eax[1],
        O[2] + xmax*np.r_[-1, 1] * eax[2],
        '--k',
    )

    # circles
    ang = thetamax*np.linspace(-1, 1, 11)
    ax.plot(
        O[0] + rc*(-np.cos(ang)*nin[0] + np.sin(ang)*e1[0]),
        O[1] + rc*(-np.cos(ang)*nin[1] + np.sin(ang)*e1[1]),
        O[2] + rc*(-np.cos(ang)*nin[2] + np.sin(ang)*e1[2]),
        '--k',
    )

    # check angles
    DA = np.r_[pt_x - Dx, pt_y - Dy, pt_z - Dz]
    DB = np.r_[pts_x - Dx, pts_y - Dy, pts_z - Dz]
    DE = np.r_[Ex - Dx, Ey - Dy, Ez - Dz]
    nA = np.linalg.norm(DA)
    nB = np.linalg.norm(DB)
    nE = np.linalg.norm(DE)
    print(np.sum(DA*DE) / (nA*nE))
    print(np.sum(DB*DE) / (nB*nE))

    import pdb; pdb.set_trace()     # DB


def _debug_spherical(
    pt_x=None,
    pt_y=None,
    pt_z=None,
    pts_x=None,
    pts_y=None,
    pts_z=None,
    rr=None,
    O=None,
    rc=None,
    ABx=None,
    ABy=None,
    ABz=None,
    nin=None,
    e0=None,
    e1=None,
    **kwdargs,
):

    # get E
    Ex = pt_x + rr * ABx
    Ey = pt_y + rr * ABy
    Ez = pt_z + rr * ABz

    # get nout(E)
    nout_x = Ex - O[0]
    nout_y = Ey - O[1]
    nout_z = Ez - O[2]
    nout_norm = np.sqrt(nout_x**2 + nout_y**2 + nout_z**2)
    nout_x = nout_x / nout_norm
    nout_y = nout_y / nout_norm
    nout_z = nout_z / nout_norm

    # get D
    Dx = O[0] + rc*nout_x
    Dy = O[1] + rc*nout_y
    Dz = O[2] + rc*nout_z

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

    ax.plot(
        np.r_[pt_x, pts_x],
        np.r_[pt_y, pts_y],
        np.r_[pt_z, pts_z],
        '.-k',
    )

    for ii in range(len(rr)):
        l0, = ax.plot(
            np.r_[pt_x, Dx[ii], pts_x],
            np.r_[pt_y, Dy[ii], pts_y],
            np.r_[pt_z, Dz[ii], pts_z],
            '.-',
        )
        ax.plot(
            np.r_[Ex[ii], Dx[ii]],
            np.r_[Ey[ii], Dy[ii]],
            np.r_[Ez[ii], Dz[ii]],
            ls='--',
            marker='.',
            c=l0.get_color(),
        )

    # circles
    ax.plot(
        np.r_[O[0], O[0] - rc*nin[0]],
        np.r_[O[1], O[1] - rc*nin[1]],
        np.r_[O[2], O[2] - rc*nin[2]],
        'ob',
    )

    # circles
    theta = np.pi*np.linspace(-1, 1, 100)[1:]
    ax.plot(
        O[0] + rc*(np.cos(theta)*nin[0] + np.sin(theta)*e0[0]),
        O[1] + rc*(np.cos(theta)*nin[1] + np.sin(theta)*e0[1]),
        O[2] + rc*(np.cos(theta)*nin[2] + np.sin(theta)*e0[2]),
        '--k',
    )
    ax.plot(
        O[0] + rc*(np.cos(theta)*nin[0] + np.sin(theta)*e1[0]),
        O[1] + rc*(np.cos(theta)*nin[1] + np.sin(theta)*e1[1]),
        O[2] + rc*(np.cos(theta)*nin[2] + np.sin(theta)*e1[2]),
        '--k',
    )
    ax.plot(
        O[0] + rc*(np.cos(theta)*e0[0] + np.sin(theta)*e1[0]),
        O[1] + rc*(np.cos(theta)*e0[1] + np.sin(theta)*e1[1]),
        O[2] + rc*(np.cos(theta)*e0[2] + np.sin(theta)*e1[2]),
        '--k',
    )

    # check angles
    DA = np.r_[pt_x - Dx, pt_y - Dy, pt_z - Dz]
    DB = np.r_[pts_x - Dx, pts_y - Dy, pts_z - Dz]
    DE = np.r_[Ex - Dx, Ey - Dy, Ez - Dz]
    nA = np.linalg.norm(DA)
    nB = np.linalg.norm(DB)
    nE = np.linalg.norm(DE)
    print(np.sum(DA*DE) / (nA*nE))
    print(np.sum(DB*DE) / (nB*nE))

    import pdb; pdb.set_trace()     # DB
