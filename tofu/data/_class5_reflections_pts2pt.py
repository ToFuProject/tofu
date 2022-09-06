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
        erot = ['e0', 'e1'][1-iplan]

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
            erot=dgeom[erot],
            # limits
            thetamax=dgeom['extenthalf'][1-iplan],
            xmax=dgeom['extenthalf'][iplan],
            # local coordinates
            nin=dgeom['nin'],
            # return
            return_x01=None,
            # number of k for interpolation
            nk=1000,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """

            # ------------------------------------------
            # Get local coordinates of reflection points

            costhetamax = np.cos(thetamax)

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

            C0, C1, C2, C3, A = _common_coefs(
                rc=rc,
                ll=ll,
                OA2=OAz2,
                OAe=OAzez,
                ez2=ez2,
            )

            # Prepare D, theta, xx
            Dx, Dy, Dz, theta, xx = _common_prepare(nb=pts_x.size)

            # get k
            AOeax = -np.sum(OA*eax)
            ABeax = ABx * eax[0] + ABy * eax[1] + ABz * eax[2]

            em = costhetamax*(-nin) - np.sin(thetamax)*erot
            eM = costhetamax*(-nin) + np.sin(thetamax)*erot

            kmin, kmax = 0, 1
            leks = [
                (O - xmax*eax, eax, True),
                (O + xmax*eax, eax, False),
                (O, em, True),
                (O, eM, False),
            ]
            for oo, ep, ss in leks:
                sca = ep[0] * ABx + ep[1] * ABy + ep[2] * ABz
                AO = oo - np.r_[pt_x, pt_y, pt_z]
                AOe = np.sum(AO*ep)
                ABe = ABx * ep[0] + ABy * ep[1] + ABz * ep[2]
                kk = AOe / ABe

                if ss is True and np.any(sca > 0):
                    kmin = max(kmin, np.min(kk[sca > 0]))
                elif ss is False and np.any(sca < 0):
                    kmax = min(kmax, np.max(kk[sca < 0]))

            import pdb; pdb.set_trace()     # DB

            k0 = max(0, k0t, k0x)
            k1 = min(1, k1t, k1x)
            kk = np.linspace(k0, k1, nk)
            for ii in range(pts_x.size):
                # ki, Ex, Ey, Ez = _common_kE_bs()


                ki, Ex, Ey, Ez = _common_kE(
                    C0[ii], C1[ii], C2[ii], C3[ii], A,
                    pt_x, pt_y, pt_z,
                    ABx[ii], ABy[ii], ABz[ii],
                )

                OEzx = (Ey - O[1])*eax[2] - (Ez - O[2])*eax[1]
                OEzy = (Ez - O[2])*eax[0] - (Ex - O[0])*eax[2]
                OEzz = (Ex - O[0])*eax[1] - (Ey - O[1])*eax[0]
                OEzn = np.sqrt(OEzx**2 + OEzy**2 + OEzz**2)

                nox = -(OEzy * eax[2] - OEzz * eax[1]) / OEzn
                noy = -(OEzz * eax[0] - OEzx * eax[2]) / OEzn
                noz = -(OEzx * eax[1] - OEzy * eax[0]) / OEzn

                # local coordinates
                thetai = np.arccos(
                    -nox*nin[0] - noy*nin[1] - noz*nin[2]
                )
                xxi = (
                    (Ex - O[0])*eax[0]
                    + (Ey - O[1])*eax[1]
                    + (Ez - O[2])*eax[2]
                )

                iin = (
                    (np.abs(thetai) <= thetamax)
                    & (np.abs(xxi) <= xmax)
                )

                # check
                check = _common_check(
                    ABx[ii], ABy[ii], ABz[ii], ll[ii],
                    nox, noy, noz, OEzn, ki, rc,
                )

                if not np.any(iin & (check < 1e-3)):
                    _common_kE2(
                        Ax=pt_x,
                        Ay=pt_y,
                        Az=pt_z,
                        Bx=pts_x[ii],
                        By=pts_y[ii],
                        Bz=pts_z[ii],
                        kk=kk,
                        rc=rc,
                        O=O,
                        eax=eax,
                        iin=iin,
                        check=check,
                        ki=ki,
                        # polynom
                        C0=C0[ii],
                        C1=C1[ii],
                        C2=C2[ii],
                        C3=C3[ii],
                        A=A,
                        # limit
                        nin=nin,
                        thetai=thetai,
                        xxi=xxi,
                        thetamax=thetamax,
                        xmax=xmax,
                    )

                icheck = check < 1.e-3
                if not np.any(check < 1.e-3):
                    msg = "No satisfactory solution"
                    raise Exception(msg)

                # ind
                ind = iin & icheck

                if np.sum(ind) == 0:
                    dist = np.abs(thetai)**2 + np.abs(xxi)**2
                    import pdb; pdb.set_trace()     # DB
                    ind = icheck & (dist == np.min(dist[icheck]))

                # handle multiple solutions
                if np.sum(ind) > 1:

                    ind = ind & (check == np.min(check[iin]))

                    if np.sum(ind) > 1:

                        msg = f"No / several solutions found: {ind.sum()}"
                        _debug_cylindrical(
                            pt_x=pt_x,
                            pt_y=pt_y,
                            pt_z=pt_z,
                            pts_x=pts_x[ii],
                            pts_y=pts_y[ii],
                            pts_z=pts_z[ii],
                            kk=ki,
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
                            ind=ind,
                            check=check,
                        )
                        raise Exception(msg)

                if thetai[ind].size > 1 or thetai[ind].size == 0:
                    import pdb; pdb.set_trace()     # DB

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
            for ii in range(pts_x.size):

                kk, Ex, Ey, Ez = _common_kE(
                    C0[ii], C1[ii], C2[ii], C3[ii], A,
                    pt_x, pt_y, pt_z,
                    ABx[ii], ABy[ii], ABz[ii],
                )

                nox = Ex - O[0]
                noy = Ey - O[1]
                noz = Ez - O[2]
                norm = np.sqrt(nox**2 + noy**2 + noz**2)
                nox /= norm
                noy /= norm
                noz /= norm

                # check
                ind = _common_check(
                    ABx[ii], ABy[ii], ABz[ii], ll[ii],
                    nox, noy, noz, norm, kk, rc,
                )

                if np.sum(ind) == 0:
                    import pdb; pdb.set_trace()     # DB
                    continue

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

    return pts2pt


# #################################################################
# #################################################################
#           Common formulas
# #################################################################


def _kminmax_plane(
    kk=None,
    sca=None,
    sign=None,
    kmin=None,
    kmax=None,
):

    return kmin, kmax


def _common_coefs(rc=None, ll=None, OA2=None, OAe=None, ez2=None):
    A = (rc**2 - OA2) * OA2
    B = 2* ll * rc**2 * OAe
    C = rc**2 * ll**2 * ez2
    D = ll**2 * (ll*ez2 + 2.*OAe)**2
    E = 2 * ll * OA2 * (ll*ez2 + 2*OAe)

    # coefficients
    C0 = 4.*C - D
    C1 = 4.*B - 4.*C - 2.*E
    C2 = 4.*A - 4.*B + C + E
    C3 = -4.*A + B

    return C0, C1, C2, C3, A


def _common_prepare(nb=None):
    return (
        np.full((nb,), np.nan),
        np.full((nb,), np.nan),
        np.full((nb,), np.nan),
        np.full((nb,), np.nan),
        np.full((nb,), np.nan),
    )


def _common_kE(
    C0=None, C1=None, C2=None, C3=None, A=None,
    pt_x=None, pt_y=None, pt_z=None,
    ABx=None, ABy=None, ABz=None,
):
    kk = np.roots(np.r_[C0, C1, C2, C3, A])

    kk = np.real(kk[np.isreal(kk)])
    kk = kk[(kk > 0.) & (kk < 1.)]

    Ex = pt_x + kk*ABx
    Ey = pt_y + kk*ABy
    Ez = pt_z + kk*ABz
    return kk, Ex, Ey, Ez


def _common_kE2(
    Ax=None,
    Ay=None,
    Az=None,
    Bx=None,
    By=None,
    Bz=None,
    kk=None,
    rc=None,
    O=None,
    eax=None,
    iin=None,
    check=None,
    ki=None,
    C0=None,
    C1=None,
    C2=None,
    C3=None,
    A=None,
    # limits
    nin=None,
    thetai=None,
    xxi=None,
    thetamax=None,
    xmax=None,
):

    Ex = Ax + kk*(Bx - Ax)
    Ey = Ay + kk*(By - Ay)
    Ez = Az + kk*(Bz - Az)

    OEzx = (Ey - O[1])*eax[2] - (Ez - O[2])*eax[1]
    OEzy = (Ez - O[2])*eax[0] - (Ex - O[0])*eax[2]
    OEzz = (Ex - O[0])*eax[1] - (Ey - O[1])*eax[0]
    OEzn = np.sqrt(OEzx**2 + OEzy**2 + OEzz**2)

    nix = (OEzy * eax[2] - OEzz * eax[1]) / OEzn
    niy = (OEzz * eax[0] - OEzx * eax[2]) / OEzn
    niz = (OEzx * eax[1] - OEzy * eax[0]) / OEzn

    xx = (
        (Ex - O[0])*eax[0]
        + (Ey - O[1])*eax[1]
        + (Ez - O[2])*eax[2]
    )
    Dx = O[0] + xx*eax[0] - rc*nix
    Dy = O[1] + xx*eax[1] - rc*niy
    Dz = O[2] + xx*eax[2] - rc*niz

    DAn = (Ax - Dx)*nix + (Ay - Dy)*niy + (Az - Dz)*niz
    DBn = (Bx - Dx)*nix + (By - Dy)*niy + (Bz - Dz)*niz
    DA = np.sqrt((Ax - Dx)**2 + (Ay - Dy)**2 + (Az - Dz)**2)
    DB = np.sqrt((Bx - Dx)**2 + (By - Dy)**2 + (Bz - Dz)**2)

    ex = (Bx - Ax)
    ey = (By - Ay)
    ez = (Bz - Az)
    enorm = np.sqrt(ex**2 + ey**2 + ez**2)
    ex = ex / enorm
    ey = ey / enorm
    ez = ez / enorm

    dE = np.sqrt((Dx - Ex)**2 + (Dy - Ey)**2 + (Dz - Ez)**2)
    ll = np.sqrt((Ax - Bx)**2 + (Ay - By)**2 + (Az - Bz)**2)
    en = ex*nix + ey*niy + ez*niz

    eq2 = (2*kk - 1)*dE + 2.*kk*(1-kk)*ll*en

    eq4 = C0*kk**4 + C1*kk**3 + C2*kk**2 + C3*kk + A

    c0 = 4*ll**2*en**2
    c1 = -8*ll**2*en**2
    c2 = 4*(ll**2*en**2 - dE**2)
    c3 = 4*dE**2
    aa = -dE**2
    eq4bis = c0*kk**4 + c1*kk**3 + c2*kk**2 + c3*kk + aa

    # limits
    theta = np.arctan2(
        (nin[1]*niz - nin[2]*niy)*eax[0]
        + (nin[2]*nix - nin[0]*niz)*eax[1]
        + (nin[0]*niy - nin[1]*nix)*eax[2],
        nix*nin[0] + niy*nin[1] + niz*nin[2],
    )
    xx = (
        (Ex - O[0])*eax[0]
        + (Ey - O[1])*eax[1]
        + (Ez - O[2])*eax[2]
    )

    dist = np.abs(thetai)**2 + np.abs(xxi)**2

    import scipy.interpolate as scpinterp

    bs = scpinterp.InterpolatedUnivariateSpline(
        kk, DAn*DB - DBn*DA, k=3,
    )

    roots = bs.roots()
    Exr = Ax + roots*(Bx - Ax)
    Eyr = Ay + roots*(By - Ay)
    Ezr = Az + roots*(Bz - Az)
    OEzxr = (Eyr - O[1])*eax[2] - (Ezr - O[2])*eax[1]
    OEzyr = (Ezr - O[2])*eax[0] - (Exr - O[0])*eax[2]
    OEzzr = (Exr - O[0])*eax[1] - (Eyr - O[1])*eax[0]
    OEznr = np.sqrt(OEzxr**2 + OEzyr**2 + OEzzr**2)

    nixr = (OEzyr * eax[2] - OEzzr * eax[1]) / OEznr
    niyr = (OEzzr * eax[0] - OEzxr * eax[2]) / OEznr
    nizr = (OEzxr * eax[1] - OEzyr * eax[0]) / OEznr
    thetar = np.arctan2(
        (nin[1]*nizr - nin[2]*niyr)*eax[0]
        + (nin[2]*nixr - nin[0]*nizr)*eax[1]
        + (nin[0]*niyr - nin[1]*nixr)*eax[2],
        nixr*nin[0] + niyr*nin[1] + nizr*nin[2],
    )

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(
        # kk, c0/C0,
        # kk, c1/C1,
        # kk, c2/C2,
        # kk, c3/C3,
        # kk, aa/A,
    # )
    plt.figure()
    plt.plot(
        kk, xx, '-b',
        kk, theta, '-r',
        ki, xxi, 'ob',
        ki, thetai, 'or',
        roots, thetar, 'xr',
    )
    plt.gca().axhspan(-thetamax, thetamax, fc='r', ls='--', alpha=0.2)
    plt.gca().axhspan(-xmax, xmax, fc='b', ls='--', alpha=0.2)

    plt.figure()
    plt.plot(
        kk, DAn*DB, '-r',
        kk, DBn*DA, '-b',
        kk, DAn*DB - DBn*DA, '.-k',
    )
    plt.plot(kk, eq2, '.-m', label='eq2')
    plt.plot(kk, eq2**2, '.-c', label='eq22')
    plt.plot(kk, eq4, '.-g', label='eq4')
    plt.gca().axhline(0, ls='--', c='k')
    for kii in ki:
        plt.gca().axvline(kii, c='k', ls='--')
    for rr in roots:
        plt.gca().axvline(rr, c='r', ls='--')

    print(iin)
    print(check)
    import pdb; pdb.set_trace()     # DB


def _common_check(
    ABx=None,
    ABy=None,
    ABz=None,
    ll=None,
    nox=None,
    noy=None,
    noz=None,
    norm=None,
    kk=None,
    rc=None,
):
    # degree 2
    en = - (ABx * nox + ABy * noy + ABz * noz) / ll
    return np.abs( (2*kk - 1)*(rc - norm) + 2*kk*(1-kk)*ll*en )


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
    ind=None,
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
