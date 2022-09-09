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
):

    # ---------
    # key

    key, cls = coll.get_diagnostic_optics(optics=key)
    key, cls = key[0], cls[0]
    dgeom = coll.dobj[cls][key]['dgeom']

    # -------------------
    #     Planar
    # -------------------

    if dgeom['type'] == 'planar':

        if dgeom['extenthalf'] is None:
            x0max, x1max = None, None
        else:
            x0max, x1max = dgeom['extenthalf']

        def ptsvect(
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # pts
            vect_x=None,
            vect_y=None,
            vect_z=None,
            # surface
            cent=dgeom['cent'],
            nin=dgeom['nin'],
            e0=dgeom['e0'],
            e1=dgeom['e1'],
            # limits
            x0max=x0max,
            x1max=x1max,
            # return
            strict=None,
            return_x01=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """

            # ------------------------------------------
            # Get local coordinates of reflection points

            # normalize vect
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
                        v_ref_x[iout] = np.nan
                        v_ref_y[iout] = np.nan
                        v_ref_z[iout] = np.nan
                        angle[iout] = np.nan

            if return_x01:
                return Dx, Dy, Dz, vrx, vry, vrz, angle, x0, x1
            else:
                return Dx, Dy, Dz, vrx, vry, vrz, angle

    # ----------------
    #   Cylindrical
    # ----------------

    elif dgeom['type'] == 'cylindrical':

        iplan = np.isinf(dgeom['curve_r']).nonzero()[0][0]
        icurv = 1 - iplan
        eax = ['e0', 'e1'][iplan]
        rc = dgeom['curve_r'][icurv]

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
            rc=rc,
            eax=dgeom[eax],
            # limits
            thetamax=dgeom['extenthalf'][icurv],
            xmax=dgeom['extenthalf'][iplan],
            # local coordinates
            nin=dgeom['nin'],
            # return
            strict=None,
            return_x01=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """
            # normalize vect
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

            C0, C1, C2 = _common_coefs(
                rc=rc,
                OAz2=OAz2,
                OAzez=OAzez,
                ez2=ez2,
            )

            # Prepare D, theta, xx
            ndim = OAzez.ndim
            shape = OAzez.shape
            (
                Dx, Dy, Dz, vrx, vry, vrz,
                angle, theta, xx,
            ) = _common_prepare(shape)

            # get k
            kk = np.full(shape, np.nan)
            if ndim == 2:
                for ii in range(shape[0]):
                    for jj in range(shape[1]):
                        kk[ii, jj] = _common_kE(
                            C0[ii, jj], C1[ii, jj], C2[ii, jj],
                        )
            else:
                for ii in range(shape[0]):
                    kk[ii] = _common_kE(C0[ii], C1[ii], C2)

            iok = np.isfinite(kk)
            if np.any(iok):
                Dx = pts_x + kk*vect_x
                Dy = pts_y + kk*vect_y
                Dz = pts_z + kk*vect_z

                ODzx = (Dx[iok] - O[1])*eax[2] - (Dz[iok] - O[2])*eax[1]
                ODzy = (Dz[iok] - O[2])*eax[0] - (Dx[iok] - O[0])*eax[2]
                ODzz = (Dx[iok] - O[0])*eax[1] - (Dy[iok] - O[1])*eax[0]
                ODzn = np.sqrt(ODzx**2 + ODzy**2 + ODzz**2)

                nox = -(ODzy * eax[2] - ODzz * eax[1]) / ODzn
                noy = -(ODzz * eax[0] - ODzx * eax[2]) / ODzn
                noz = -(ODzx * eax[1] - ODzy * eax[0]) / ODzn

                # scalar product (for angle + reflection)
                if np.isscalar(vect_x):
                    scavn = -(vect_x*nox + vect_y*noy + vect_z*noz)

                    # get vect_reflect
                    vrx = vect_x + 2.*scavn * nox
                    vry = vect_y + 2.*scavn * noy
                    vrz = vect_z + 2.*scavn * noz

                else:
                    scavn = -(vect_x[iok]*nox + vect_y[iok]*noy + vect_z[iok]*noz)
                    # get vect_reflect
                    vrx = vect_x[iok] + 2.*scavn * nox
                    vry = vect_y[iok] + 2.*scavn * noy
                    vrz = vect_z[iok] + 2.*scavn * noz

                angle[iok] = -np.arcsin(scavn)

                # x0, x1
                if strict is True or return_x01 is True:
                    theta[iok] = np.arctan2(
                        -((nin[1]*noz - nin[2]*noy)*eax[0]
                        + (nin[2]*nox - nin[0]*noz)*eax[1]
                        + (nin[0]*noy - nin[1]*nox)*eax[2]),
                        -nox*nin[0] - noy*nin[1] - noz*nin[2],
                    )

                    xx[iok] = (
                        (Dx[iok] - O[0])*eax[0]
                        + (Dy[iok] - O[1])*eax[1]
                        + (Dz[iok] - O[2])*eax[2]
                    )

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

            # enforce normalization
            iok = np.isfinite(vrx)
            vnorm = np.sqrt(vrx[iok]**2 + vry[iok]**2 + vrz[iok]**2)
            vrx[iok] = vrx[iok] / vnorm
            vry[iok] = vry[iok] / vnorm
            vrz[iok] = vrz[iok] / vnorm

            # return
            if return_x01:
                return Dx, Dy, Dz, vrx, vry, vrz, angle, xx, theta
            else:
                return Dx, Dy, Dz, vrx, vry, vrz, angle

    # ----------------
    #   Spherical
    # ----------------

    elif dgeom['type'] == 'spherical':

        def ptsvect(
            pt_x=None,
            pt_y=None,
            pt_z=None,
            # pts
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # surface
            O=dgeom['cent'] + dgeom['curve_r'][0]*dgeom['nin'],
            rc=dgeom['curve_r'][0],
            # limits
            dthetamax=dgeom['extenthalf'][0],
            phimax=dgeom['extenthalf'][1],
            # local coordinates
            nin=dgeom['nin'],
            e0=dgeom['e0'],
            e1=dgeom['e1'],
            # return
            return_x01=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """
            # normalize vect
            vn = np.sqrt(vect_x**2 + vect_y**2 + vect_z**2)
            vect_x = vect_x / vn
            vect_y = vect_y / vn
            vect_z = vect_z / vn

            # ------------------------------------------
            # Get local coordinates of reflection points

            # get parameters for k
            OA = np.r_[pt_x - O[0], pt_y - O[1], pt_z - O[2]]
            OA2 = np.sum(OA**2)

            ABx = pts_x - pt_x
            ABy = pts_y - pt_y
            ABz = pts_z - pt_z
            ll = np.sqrt(ABx**2 + ABy**2 + ABz**2)

            OAe = ((OA[0] * ABx) + (OA[1] * ABy) + (OA[2] * ABz)) / ll

            C0, C1, C2, C3, A = _common_coefs(
                rc=rc,
                ll=ll,
                OA2=OA2,
                OAe=OAe,
                ez2=1.,
            )

            # Prepare D, theta, xx
            Dx, Dy, Dz, dtheta, phi = _common_prepare(nb=pts_x.size)

            # get k
            kk = np.full()
            for ii in range(pts_x.size):
                kk = _common_kE(C0[ii], C1[ii], C2)

                if kk.size == 0:
                    continue

                nox = Ex - O[0]
                noy = Ey - O[1]
                noz = Ez - O[2]
                norm = np.sqrt(nox**2 + noy**2 + noz**2)
                nox /= norm
                noy /= norm
                noz /= norm

                if np.sum(ind) == 0:
                    import pdb; pdb.set_trace()     # DB
                    pass

                # local coordinates
                dthi = np.arcsin(
                    nox*e1[0] + noy*e1[1] + noz*e1[2]
                )
                phii = np.arcsin(
                    (nox*e0[0] + noy*e0[1] + noz*e0[2]) / np.cos(dthi)
                )

                # handle multiple solutions
                if np.sum(ind) > 1:

                    ind = (
                        (np.abs(dthi) <= dthetamax)
                        & (np.abs(phii) <= phimax)
                    )

                    if np.sum(ind) == 0:
                        ind = np.argmin(np.abs(dthi)**2 + np.abs(phii)**2)
                    elif np.sum(ind) > 1:
                        msg = f"No / several solutions found: {ind.sum()}"
                        _debug_spherical(
                            pt_x=pt_x,
                            pt_y=pt_y,
                            pt_z=pt_z,
                            pts_x=pts_x[ii],
                            pts_y=pts_y[ii],
                            pts_z=pts_z[ii],
                            rr=rr,
                            O=O,
                            rc=rc,
                            ABx=ABx[ii],
                            ABy=ABy[ii],
                            ABz=ABz[ii],
                            nin=nin,
                            e0=e0,
                            e1=e1,
                        )
                        raise Exception(msg)

                Dx[ii] = O[0] + rc*nox[ind]
                Dy[ii] = O[1] + rc*noy[ind]
                Dz[ii] = O[2] + rc*noz[ind]
                dtheta[ii] = dthi[ind]
                phi[ii] = phii[ind]

            # return
            if return_x01:
                return Dx, Dy, Dz, dtheta, phi
            else:
                return Dx, Dy, Dz

    # ----------------
    #   Toroidal
    # ----------------

    elif dgeom['type'] == 'toroidal':

        raise NotImplementedError()

    return ptsvect


# ##################################################################
# ##################################################################
#                   preparation routine
# ##################################################################


def _get_project_plane(
    plane_pt=None,
    plane_nin=None,
    plane_e0=None,
    plane_e1=None,
):

    def _project_poly_on_plane_from_pt(
        pt_x=None,
        pt_y=None,
        pt_z=None,
        poly_x=None,
        poly_y=None,
        poly_z=None,
        vx=None,
        vy=None,
        vz=None,
        plane_pt=plane_pt,
        plane_nin=plane_nin,
        plane_e0=plane_e0,
        plane_e1=plane_e1,
    ):

        sca0 = (
            (plane_pt[0] - pt_x)*plane_nin[0]
            + (plane_pt[1] - pt_y)*plane_nin[1]
            + (plane_pt[2] - pt_z)*plane_nin[2]
        )

        if vx is None:
            vx = poly_x - pt_x
            vy = poly_y - pt_y
            vz = poly_z - pt_z

        sca1 = vx*plane_nin[0] + vy*plane_nin[1] + vz*plane_nin[2]

        k = sca0 / sca1

        px = pt_x + k * vx
        py = pt_y + k * vy
        pz = pt_z + k * vz

        p0 = (
            (px - plane_pt[0])*plane_e0[0]
            + (py - plane_pt[1])*plane_e0[1]
            + (pz - plane_pt[2])*plane_e0[2]
        )
        p1 = (
            (px - plane_pt[0])*plane_e1[0]
            + (py - plane_pt[1])*plane_e1[1]
            + (pz - plane_pt[2])*plane_e1[2]
        )

        return p0, p1

    def _back_to_3d(
        x0=None,
        x1=None,
        plane_pt=plane_pt,
        plane_e0=plane_e0,
        plane_e1=plane_e1,
    ):

        return (
            plane_pt[0] + x0*plane_e0[0] + x1*plane_e1[0],
            plane_pt[1] + x0*plane_e0[1] + x1*plane_e1[1],
            plane_pt[2] + x0*plane_e0[2] + x1*plane_e1[2],
        )

    return _project_poly_on_plane_from_pt, _back_to_3d


# #################################################################
# #################################################################
#           Common formulas
# #################################################################


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
    )


def _common_kE(C0=None, C1=None, C2=None):
    kk = np.roots(np.r_[C0, C1, C2])
    kk = np.real(kk[np.isreal(kk)])

    # outgoing is necessarily the maxium k
    if np.any( kk > 0.):
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
    print(np.sum(DA*DE) / (nA*nE) )
    print(np.sum(DB*DE) / (nB*nE) )

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
    print(np.sum(DA*DE) / (nA*nE) )
    print(np.sum(DB*DE) / (nB*nE) )

    import pdb; pdb.set_trace()     # DB
