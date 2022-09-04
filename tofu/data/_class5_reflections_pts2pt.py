# -*- coding: utf-8 -*-


import numpy as np
import datastock as ds


# ##############################################################
# ##############################################################
#           Finding reflection points
# ##############################################################


def _get_pts2pt(
    coll=None,
    key=None,
):

    # ---------
    # key

    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lcryst + lgrat,
    )

    cls = 'crystal' if key in lcryst else 'grating'
    dgeom = coll.dobj[cls][key]['dgeom']

    # -------------------
    #     Planar
    # -------------------

    if dgeom['type'] == 'planar':

        def pts2pt(
            pt_x=None,
            pt_y=None,
            pt_z=None,
            # pts
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # surface
            cent=dgeom['cent'],
            nin=dgeom['nin'],
            e0=dgeom['e0'],
            e1=dgeom['e1'],
            # return
            return_xyz=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """

            # ------------------------------------------
            # Get local coordinates of reflection points

            # get parameters for k
            CA = np.r_[
                pt_x - cent[0], pt_y - cent[1], pt_z - cent[2],
            ]
            CAdotn = np.sum(CA * nin)

            ABx = pts_x - pt_x
            ABy = pts_y - pt_y
            ABz = pts_z - pt_z

            length = np.sqrt(ABx**2 + ABy**2 + ABz**2)

            # get k
            kk = CAdotn / (
                2*CAdotn + (nin[0] * ABx + nin[1]*ABy + nin[2]*ABz)
            )

            # get E
            Ex = pt_x + kk * ABx
            Ey = pt_y + kk * ABy
            Ez = pt_z + kk * ABz

            # x0, x1
            x0 = (
                (Ex - cent[0])*e0[0]
                + (Ey - cent[1])*e0[1]
                + (Ez - cent[2])*e0[2]
            )
            x1 = (
                (Ex - cent[0])*e1[0]
                + (Ey - cent[1])*e1[1]
                + (Ez - cent[2])*e1[2]
            )

            if return_xyz:
                # get D
                Dx = cent[0] + x0 * e0[0] + x1*e1[0]
                Dy = cent[1] + x0 * e0[1] + x1*e1[1]
                Dz = cent[2] + x0 * e0[2] + x1*e1[2]

                return x0, x1, Dx, Dy, Dz
            else:
                return x0, x1

    # ----------------
    #   Cylindrical
    # ----------------

    elif dgeom['type'] == 'cylindrical':

        iplan = np.isinf(dgeom['curve_r']).nonzero()[0][0]
        eax = ['e0', 'e1'][iplan]

        def pts2pt(
            pt_x=None,
            pt_y=None,
            pt_z=None,
            # pts
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # surface
            O=dgeom['cent'] + dgeom['nin'] * dgeom['curve_r'][1 - iplan],
            rc=dgeom['curve_r'][1 - iplan],
            eax=dgeom[eax],
            # limits
            thetamax=dgeom['extenthalf'][1-iplan],
            xmax=dgeom['extenthalf'][iplan],
            # local coordinates
            nin=dgeom['nin'],
            # return
            return_x01=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """

            # ------------------------------------------
            # Get local coordinates of reflection points

            # get parameters for k
            OA = np.r_[pt_x - O[0], pt_y - O[1], pt_z - O[2]]
            OAz = np.cross(OA, eax)
            OAz2 = np.sum(OAz**2)

            ABx = pts_x - pt_x
            ABy = pts_y - pt_y
            ABz = pts_z - pt_z
            ll = np.sqrt(ABx**2 + ABy**2 + ABz**2)
            ezx = (ABy*eax[2] - ABz*eax[1]) / ll
            ezy = (ABz*eax[0] - ABx*eax[2]) / ll
            ezz = (ABx*eax[1] - ABy*eax[0]) / ll
            ez2 = ezx**2 + ezy**2 + ezz**2

            OAzez = OAz[0] * ezx + OAz[1] * ezy + OAz[2] * ezz

            A = (rc**2 - OAz2) * OAz2
            B = 2*rc**2 * ll * OAzez
            C = rc**2 * ll**2 * ez2
            D = 4 * ll * OAz2 * OAzez
            E = 2 * ll**2 * OAz2 * ez2
            F = 4 * ll**2 * OAzez**2
            G = ll**4 * ez2**2
            H = 4 * ll**3 * OAzez * ez2

            # coefficients
            C0 = -16.*G
            C1 = -12*H + 24.*G
            C2 = 17.*H - 9.*G - 9.*F - 8.*E + 4.*C
            C3 = -6.*H + 12.*F + 10.*E - 6.*D - 4.*C + 4.*B
            C4 = -4.*F - 3.*E + 7.*D + C - 4.*B + 4.*A
            C5 = -2.*D + B - 4.*A

            # nout dtheta, phi
            Dx = np.full(pts_x.size, np.nan)
            Dy = np.full(pts_x.size, np.nan)
            Dz = np.full(pts_x.size, np.nan)
            theta = np.full(pts_x.size, np.nan)
            xx = np.full(pts_x.size, np.nan)

            # get k
            for ii in range(pts_x.size):
                kk = np.roots(np.r_[
                    C0[ii], C1[ii], C2[ii], C3[ii], C4[ii], C5[ii], A,
                ])
                kk = np.real(kk[np.isreal(kk)])
                kk = kk[(kk > 0.) & (kk < 1.)]

                Ex = pt_x + kk*ABx[ii]
                Ey = pt_y + kk*ABy[ii]
                Ez = pt_z + kk*ABz[ii]

                OEzx = (Ey - O[1])*eax[2] - (Ez - O[2])*eax[1]
                OEzy = (Ez - O[2])*eax[0] - (Ex - O[0])*eax[2]
                OEzz = (Ex - O[0])*eax[1] - (Ey - O[1])*eax[0]
                OEzn = np.sqrt(OEzx**2 + OEzy**2 + OEzz**2)

                nox = (OEzy * eax[2] - OEzz * eax[1]) / OEzn
                noy = (OEzz * eax[0] - OEzx * eax[2]) / OEzn
                noz = (OEzx * eax[1] - OEzy * eax[0]) / OEzn

                # check
                en = -(
                    ABx[ii] * nox
                    + ABy[ii] * noy
                    + ABz[ii] * noz
                ) / ll[ii]
                check = (2*kk - 1)*(rc - OEzn) + 2*kk*(1-kk)*ll[ii]*en
                ind = np.abs(check) < 1e-10

                if np.sum(ind) == 0:
                    import pdb; pdb.set_trace()     # DB
                    pass

                # local coordinates
                thetai = np.arccos(
                    nox*nin[0] + noy*nin[1] + noz*nin[2]
                )
                xxi = (
                    (Ex - O[0])*eax[0]
                    + (Ey - O[1])*eax[1]
                    + (Ez - O[2])*eax[2]
                )

                # handle multiple solutions
                if np.sum(ind) > 1:

                    ind = (
                        (np.abs(thetai) <= thetamax)
                        & (np.abs(xxi) <= xmax)
                    )

                    if np.sum(ind) == 0:
                        ind = np.argmin(np.abs(thetai)**2 + np.abs(xxi)**2)
                    elif np.sum(ind) > 1:
                        msg = f"No / several solutions found: {ind.sum()}"
                        _debug_cylindrical(
                            pt_x=pt_x,
                            pt_y=pt_y,
                            pt_z=pt_z,
                            pts_x=pts_x[ii],
                            pts_y=pts_y[ii],
                            pts_z=pts_z[ii],
                            kk=kk,
                            O=O,
                            rc=rc,
                            ABx=ABx[ii],
                            ABy=ABy[ii],
                            ABz=ABz[ii],
                            nox=nox,
                            noy=noy,
                            noz=noz,
                            nin=nin,
                            eax=eax,
                            xx=xxi,
                            theta=thetai,
                            xmax=xmax,
                            thetamax=thetamax,
                            check=check,
                        )
                        raise Exception(msg)

                _debug_cylindrical(
                    pt_x=pt_x,
                    pt_y=pt_y,
                    pt_z=pt_z,
                    pts_x=pts_x[ii],
                    pts_y=pts_y[ii],
                    pts_z=pts_z[ii],
                    kk=kk,
                    O=O,
                    rc=rc,
                    ABx=ABx[ii],
                    ABy=ABy[ii],
                    ABz=ABz[ii],
                    nox=nox,
                    noy=noy,
                    noz=noz,
                    nin=nin,
                    eax=eax,
                    xx=xxi,
                    theta=thetai,
                    xmax=xmax,
                    thetamax=thetamax,
                    check=check,
                )


                theta[ii] = thetai[ind]
                xx[ii] = xxi[ind]
                Dx[ii] = O[0] + xx[ii]*eax[0] + rc*nox[ind]
                Dy[ii] = O[1] + xx[ii]*eax[1] + rc*noy[ind]
                Dz[ii] = O[2] + xx[ii]*eax[2] + rc*noz[ind]

            # return
            if return_x01:
                return Dx, Dy, Dz, xx, theta
            else:
                return Dx, Dy, Dz

    # ----------------
    #   Spherical
    # ----------------

    elif dgeom['type'] == 'spherical':

        def pts2pt(
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

            C0 = (rc**2 - OA2) * OA2
            C1 = rc**2 * ll**2
            C2 = 2* ll * rc**2 * OAe
            C3 = 2 * ll * OA2 * (ll + 2*OAe)
            C4 = ll**2 * (ll + 2.*OAe)**2

            A = 4.*C1 - C4
            B = -4.*C1 + 4*C2 - 2*C3
            C = 4.*C0 + C1 - 4.*C2 + C3
            D = -4.*C0 + C2
            E = C0

            # nout dtheta, phi
            Dx = np.full(pts_x.size, np.nan)
            Dy = np.full(pts_x.size, np.nan)
            Dz = np.full(pts_x.size, np.nan)
            dtheta = np.full(pts_x.size, np.nan)
            phi = np.full(pts_x.size, np.nan)

            # get k
            for ii in range(pts_x.size):
                kk = np.roots(np.r_[A[ii], B[ii], C[ii], D[ii], E])
                kk = np.real(kk[np.isreal(kk)])
                kk = kk[(kk > 0.) & (kk < 1.)]

                Ex = pt_x + kk*ABx[ii]
                Ey = pt_y + kk*ABy[ii]
                Ez = pt_z + kk*ABz[ii]
                nox = Ex - O[0]
                noy = Ey - O[1]
                noz = Ez - O[2]
                norm = np.sqrt(nox**2 + noy**2 + noz**2)
                nox /= norm
                noy /= norm
                noz /= norm

                # check the base equation is solved
                en = -(
                    ABx[ii] * nox
                    + ABy[ii] * noy
                    + ABz[ii] * noz
                ) / ll[ii]
                check = (2*kk - 1)*(rc - norm) + 2*kk*(1-kk)*ll[ii]*en
                ind = np.abs(check) < 1e-10

                if np.sum(ind) == 0:
                    import pdb; pdb.set_trace()     # DB
                    pass

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

    return pts2pt


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
