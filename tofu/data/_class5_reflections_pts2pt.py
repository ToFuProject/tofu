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

        def pts2pt(
            pt_x=None,
            pt_y=None,
            pt_z=None,
            # pts
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # surface
            O=None,
            rc=None,
            eax=None,
            # local coordinates
            nin=None,
            e0=None,
            e1=None,
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
            rc2OA = rc**2 - OA2

            A = ll**2 * (4*rc**2 - (ll + 2*OAe)**2)
            B = 2*ll * ( -rc**2*ll + 4*rc**2*OAe - 2*OA2*(ll + 2*OAe) )
            C = ll**2 * (rc**2 + 2*OA2) + 4 * rc2OA * (OA2 - ll*OAe)
            D = -2. * rc2OA * OA2 + 2*ll*rc**2 * OAe
            E = rc2OA * OA2

            # get k
            k = np.full(A.shape, np.nan)
            for ii in range(pts_x.size):
                rr = np.roots(np.r_[A[ii], B[ii], C[ii], D[ii], E[ii]])
                rr = rr[np.isreal(rr)]
                ind = (rr > 0.) & (rr < 1.)
                if np.sum(ind) != 1:
                    msg = f"No / several solutions found: {ind.sum()}"
                    raise Exception(msg)
                k[ii] = rr[ind]

            # get E
            Ex = pt_x + kk * ABx
            Ey = pt_y + kk * ABy
            Ez = pt_z + kk * ABz

            # get nout(E)
            nout_x = Ex - O[0]
            nout_y = Ey - O[1]
            nout_z = Ez - O[2]
            nout_norm = np.sqrt(nout_x**2 + nout_y**2 + nout_z**2)

            # get D
            Dx = O[0] + rc*nout_x
            Dy = O[1] + rc*nout_y
            Dz = O[2] + rc*nout_z

            if return_x01:
                dtheta = np.arcsin(nout_x * e1[0] + nout_y * e1[1] + nout_z * e1[2])
                phi = np.arcsin(
                    (nout_x * e0[0] + nout_y * e0[1] + nout_z * e0[2]) / np.cos(dtheta)
                )

                return Dx, Dy, Dz, dtheta, phi
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
            dthi = np.full((4,), np.nan)
            phii = np.full((4,), np.nan)
            for ii in range(pts_x.size):
                kk = np.roots(np.r_[A[ii], B[ii], C[ii], D[ii], E])
                kk = np.real(kk[np.isreal(kk)])
                ind = (kk > 0.) & (kk < 1.)

                if ind.sum() == 1:
                    Dx[ii] = O[0] + rc*nox[ind]
                    Dy[ii] = O[1] + rc*noy[ind]
                    Dz[ii] = O[2] + rc*noz[ind]
                    dtheta[ii] = dthi[ind]
                    phi[ii] = phii[ind]

                elif ind.sum() == 0:
                    import pdb; pdb.set_trace()     # DB
                    pass

                else:
                    Ex = pt_x + kk[ind]*ABx[ii]
                    Ey = pt_y + kk[ind]*ABy[ii]
                    Ez = pt_z + kk[ind]*ABz[ii]
                    nox = Ex - O[0]
                    noy = Ey - O[1]
                    noz = Ez - O[2]
                    norm = np.sqrt(nox**2 + noy**2 + noz**2)
                    nox /= norm
                    noy /= norm
                    noz /= norm

                    dthi = np.arcsin(
                        nox*e1[0] + noy*e1[1] + noz*e1[2]
                    )
                    phii = np.arcsin(
                        (nox*e0[0] + noy*e0[1] + noz*e0[2]) / np.cos(dthi)
                    )

                    ind = (
                        (np.abs(dthi) <= dthetamax)
                        & (np.abs(phii) <= phimax)
                    )
                    if np.sum(ind) == 1:
                        Dx[ii] = O[0] + rc*nox[ind]
                        Dy[ii] = O[1] + rc*noy[ind]
                        Dz[ii] = O[2] + rc*noz[ind]
                        dtheta[ii] = dthi[ind]
                        phi[ii] = phii[ind]

                    elif np.sum(ind) == 0:
                        i0 = np.argmin(np.abs(dthi)**2 + np.abs(phii)**2)
                        Dx[ii] = O[0] + rc*nox[i0]
                        Dy[ii] = O[1] + rc*noy[i0]
                        Dz[ii] = O[2] + rc*noz[i0]
                        dtheta[ii] = dthi[i0]
                        phi[ii] = phii[i0]

                    elif ind.sum() > 1:
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
