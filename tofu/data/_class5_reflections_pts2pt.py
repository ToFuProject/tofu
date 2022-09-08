# -*- coding: utf-8 -*-


import warnings


import numpy as np
import scipy.interpolate as scpinterp
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
            nk=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """

            # ------------------------------------------
            # Get local coordinates of reflection points

            A = np.r_[pt_x, pt_y, pt_z]
            Dx, Dy, Dz, xx, theta = _common_prepare(nb=pts_x.size)

            em = np.cos(thetamax)*(erot) + np.sin(thetamax)*(-nin)
            eM = np.cos(thetamax)*(erot) - np.sin(thetamax)*(-nin)

            lzones = [
                (
                    (O - xmax*eax, eax),
                    (O + xmax*eax, -eax),
                    (O, em),
                    (O, -eM),
                ),
                (
                    (O - xmax*eax, eax),
                    (O + xmax*eax, -eax),
                    (O, -em),
                    (O, eM),
                ),
            ]
            for ii in range(pts_x.size):
                B = np.r_[pts_x[ii], pts_y[ii], pts_z[ii]],
                AB = B - A
                eAB = AB / np.linalg.norm(AB)

                # mindist and associated k
                mdist_k, mdist = _mindist_2lines(
                    A=A,
                    B=B,
                    O=O,
                    eax=eax,
                )

                # compute kmin, kmax
                onaxis, kaxis, kk = _kminmax_plane(
                    A=A,
                    B=B,
                    lzones=lzones,
                    ii=ii,
                    nin=nin,
                    nk=nk,
                    mdist=mdist,
                    mdist_k=mdist_k,
                )

                if onaxis is True:
                    continue
                    # ll = np.linalg.norm(B-A)
                    # en = (1 - 2.*mdist_k)*rc / (2.*mdist_k*(1 - mdist_k) * ll)
                    # if np.abs(en) < 1:
                        # beta = np.arctan2(
                            # -np.sum(erot*eAB),
                            # np.sum(-nin*eAB),
                        # )
                        # if np.abs(np.arccos(-en) + beta) < thetamax:
                            # theta[ii] = np.arccos(-en) + beta
                        # elif np.abs(-np.arccos(-en) + beta) < thetamax:
                            # theta[ii] = -np.arccos(-en) + beta
                        # else:
                            # onaxis = False
                    # else:
                        # onaxis = False

                    # if onaxis is True:
                        # E = np.r_[pt_x, pt_y, pt_z] + mdist_k*AB
                        # xx[ii] = np.sum((E - O) * eax)
                        # no = np.cos(theta[ii])*(-nin) + np.sin(theta[ii])*erot
                        # Dx[ii] = O[0] + xx[ii]*eax[0] + rc*no[0]
                        # Dy[ii] = O[1] + xx[ii]*eax[1] + rc*no[1]
                        # Dz[ii] = O[2] + xx[ii]*eax[2] + rc*no[2]

                if onaxis is False and kk is not None:

                    if kk is None:
                        import pdb; pdb.set_trace()     # DB

                    if np.isscalar(kk):

                        (
                            nix, niy, niz,
                            D0x, D0y, D0z,
                            xxi,
                        ) = _get_Dnin_from_k(
                            O=O, eax=eax,
                            Ax=pt_x, Ay=pt_y, Az=pt_z,
                            Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                            kk=kk,
                            rc=rc,
                            nin=nin,
                        )

                        # derive DAn, DB, DBn, DA
                        eq = _get_DADB(
                            Ax=pt_x, Ay=pt_y, Az=pt_z,
                            Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                            Dx=D0x, Dy=D0y, Dz=D0z,
                            nix=nix, niy=niy, niz=niz,
                        )

                        if np.abs(eq) < 1.e-2:
                            Dx[ii] = D0x
                            Dy[ii] = D0y
                            Dz[ii] = D0z
                            xx[ii] = xxi
                        else:
                            import pdb; pdb.set_trace()     # DB
                            pass

                    else:

                        # nin
                        (
                            nix, niy, niz,
                            Dxi, Dyi, Dzi,
                            # D1x, D1y, D1z,
                            xxi,
                        ) = _get_Dnin_from_k(
                            O=O, eax=eax,
                            Ax=pt_x, Ay=pt_y, Az=pt_z,
                            Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                            kk=kk, rc=rc,
                            nin=nin,
                        )

                        # derive DAn, DB, DBn, DA
                        eq = _get_DADB(
                            Ax=pt_x, Ay=pt_y, Az=pt_z,
                            Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                            Dx=Dxi, Dy=Dyi, Dz=Dzi,
                            nix=nix, niy=niy, niz=niz,
                        )

                        # derive roots
                        roots = scpinterp.InterpolatedUnivariateSpline(
                            kk,
                            eq,
                            k=3,
                        ).roots()

                        if roots.size != 1 or np.abs(theta[ii]) > thetamax:
                            if roots.size > 1:
                                ra, rb = np.polyfit(kk, eq, 1)
                                roots = np.r_[-rb/ra]

                            _debug_cylindrical(**locals())
                            msg = f"{roots.size} solutions for {ii} / {pts_x.size}"
                            print(msg)
                            continue
                            # raise Exception(msg)

                        # nin, xx, D
                        (
                            nix, niy, niz,
                            Dx[ii], Dy[ii], Dz[ii], xx[ii],
                        ) = _get_Dnin_from_k(
                            O=O, eax=eax,
                            Ax=pt_x, Ay=pt_y, Az=pt_z,
                            Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                            kk=roots[0],
                            rc=rc,
                            nin=nin,
                        )

                    # theta, xx
                    theta[ii] = np.arctan2(
                        (nin[1]*niz - nin[2]*niy)*eax[0]
                        + (nin[2]*nix - nin[0]*niz)*eax[1]
                        + (nin[0]*niy - nin[1]*nix)*eax[2],
                        nix*nin[0] + niy*nin[1] + niz*nin[2],
                    )

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


def _mindist_2lines(A=None, B=None, O=None, eax=None):
    OAe = np.cross(A - O, eax)
    ABe = np.cross(B - A, eax)
    OAe2= np.sum(OAe**2)
    ABe2= np.sum(ABe**2)
    OAeABe = np.sum(OAe * ABe)
    kk = -OAeABe/ABe2
    return kk, OAe2 - OAeABe**2/ABe2


def _kminmax_ptinz(pt=None, lp=None):
    return np.all([
        np.sum((pt - oo)*ep) > 0 for oo, ep in lp
    ]) > 0


def _kminmax_plane(
    A=None,
    B=None,
    lzones=None,
    kmin=0,
    kmax=1,
    nin=None,
    ii=None,
    nk=None,
    mdist=None,
    mdist_k=None,
):


    # ---------------
    # check inputs

    if nk is None:
        nk = 100

    AB = B - A

    # ---------------
    # get lkin, lkout

    nz = len(lzones)
    inzone = np.zeros((nz,), dtype=bool)
    kin = np.zeros((nz,))
    kout = np.ones((nz,))
    for ip, lp in enumerate(lzones):

        # check if start / end in zone
        if _kminmax_ptinz(pt=A, lp=lp):
            inzone[ip] = True

        if _kminmax_ptinz(pt=B, lp=lp):
            inzone[ip] = True

        # check transitions
        for jj, (oo, ep) in enumerate(lp):
            AO = oo - A
            AOe = np.sum(AO*ep)
            ABe = np.sum(AB * ep)
            if np.abs(ABe) < 1e-14:
                continue
            kk = AOe / ABe
            if kk < 0 or kk > 1:
                continue

            allinkm = _kminmax_ptinz(pt=A + (kk - 1.e-6)*AB, lp=lp)
            allinkp = _kminmax_ptinz(pt=A + (kk + 1.e-6)*AB, lp=lp)
            if allinkp:
                kin[ip] = max(kin[ip], kk)
                inzone[ip] = True
            elif allinkm:
                kout[ip] = min(kout[ip], kk)

            # safeguard
            if kout[ip] < kin[ip]:
                import pdb; pdb.set_trace()     # DB
                a = 1

    # ---------------
    # derive kmin, kmax

    if mdist < 1.e-14:
        onaxis = True
        kaxis = mdist_k
    else:
        onaxis = False
        kaxis = None
        kk = None

    if np.sum(inzone) == 0:
        kk = None

    else:
        kin= kin[inzone]
        kout= kout[inzone]

        inds = np.argsort(kin)
        kk = []
        for ss in inds:
            if kout[ss] - kin[ss] > 1.e-12:
                kk.append(np.linspace(kin[ss], kout[ss], nk))
            elif onaxis is True:
                pass
            else:
                kk.append([0.5*(kout[ss] + kin[ss])])

        if len(kk) > 0:
            kk = np.concatenate(tuple(kk))
        else:
            kk = None

    return onaxis, kaxis, kk


def _get_Dnin_from_k(
    O=None, eax=None,
    Ax=None, Ay=None, Az=None,
    Bx=None, By=None, Bz=None,
    kk=None, rc=None,
    nin=None,
):
    Ex = Ax + kk*(Bx - Ax)
    Ey = Ay + kk*(By - Ay)
    Ez = Az + kk*(Bz - Az)

    sca = (Ex - O[0])*eax[0] + (Ey - O[1])*eax[1] + (Ez - O[2])*eax[2]
    nix = (Ex - O[0]) - sca*eax[0]
    niy = (Ey - O[1]) - sca*eax[1]
    niz = (Ez - O[2]) - sca*eax[2]
    ninorm = np.sqrt(nix**2 + niy**2 + niz**2)
    nix = -nix / ninorm
    niy = -niy / ninorm
    niz = -niz / ninorm

    xx = (Ex - O[0])*eax[0] + (Ey - O[1])*eax[1] + (Ez - O[2])*eax[2]

    sign = nix*nin[0] + niy*nin[1] + niz*nin[2]

    nix = sign*nix
    niy = sign*niy
    niz = sign*niz

    Dx = O[0] + xx*eax[0] - rc*nix
    Dy = O[1] + xx*eax[1] - rc*niy
    Dz = O[2] + xx*eax[2] - rc*niz
    return nix, niy, niz, Dx, Dy, Dz, xx


def _get_DADB(
    Ax=None, Ay=None, Az=None,
    Bx=None, By=None, Bz=None,
    Dx=None, Dy=None, Dz=None,
    nix=None, niy=None, niz=None,
):

    DA = np.sqrt((Ax - Dx)**2 + (Ay - Dy)**2 + (Az - Dz)**2)
    DB = np.sqrt((Bx - Dx)**2 + (By - Dy)**2 + (Bz - Dz)**2)

    DAn = (Ax - Dx)*nix + (Ay - Dy)*niy + (Az - Dz)*niz
    DBn = (Bx - Dx)*nix + (By - Dy)*niy + (Bz - Dz)*niz
    return DB/DBn - DA/DAn


def _common_prepare(nb=None):
    return (
        np.full((nb,), np.nan),
        np.full((nb,), np.nan),
        np.full((nb,), np.nan),
        np.full((nb,), np.nan),
        np.full((nb,), np.nan),
    )


# #################################################################
# #################################################################
#           Debug
# #################################################################


def _debug_cylindrical(
    nin=None,
    erot=None,
    eax=None,
    nix=None,
    niy=None,
    niz=None,
    xxi=None,
    thetamax=None,
    xmax=None,
    roots=None,
    kk=None,
    eq=None,
    pt_x=None,
    pt_y=None,
    pt_z=None,
    pts_x=None,
    pts_y=None,
    pts_z=None,
    O=None,
    Dxi=None,
    Dyi=None,
    Dzi=None,
    rc=None,
    ii=None,
    **kwdargs,
):

    thetai = np.arctan2(
        (nin[1]*niz - nin[2]*niy)*eax[0]
        + (nin[2]*nix - nin[0]*niz)*eax[1]
        + (nin[0]*niy - nin[1]*nix)*eax[2],
        nix*nin[0] + niy*nin[1] + niz*nin[2],
    )

    import matplotlib.pyplot as plt

    #----------------
    #  plot xx, theta

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"ii = {ii}")

    ax0 = fig.add_subplot(131)
    ax0.plot(
        kk, xxi, '.-b',
        kk, thetai, '.-r',
    )
    ax0.axhspan(-thetamax, thetamax, fc='r', ls='--', alpha=0.2)
    ax0.axhspan(-xmax, xmax, fc='b', ls='--', alpha=0.2)
    for rr in roots:
        ax0.axvline(rr, c='r', ls='--')

    #----------------
    #  plot equation

    ax1 = fig.add_subplot(132)
    ax1.plot(
        kk, eq, '.-k',
        kk, np.polyval(np.polyfit(kk, eq, 1), kk), '-b',
    )
    ax1.axhline(0, ls='--', c='k')
    for rr in roots:
        ax1.axvline(rr, c='r', ls='--')

    #----------------
    #  plot geometry

    ax2 = fig.add_subplot(133, projection='3d')
    ax2.plot(
        np.r_[pt_x, pts_x[ii]],
        np.r_[pt_y, pts_y[ii]],
        np.r_[pt_z, pts_z[ii]],
        '.-k',
    )

    # for rr in roots:
        # l0, = ax.plot(
            # np.r_[pt_x, Dxi, pts_x[ii]],
            # np.r_[pt_y, Dyi, pts_y[ii]],
            # np.r_[pt_z, Dzi, pts_z[ii]],
            # '.-',
        # )
        # ax.plot(
            # np.r_[O[0] + xxi*eax[0], Dxi],
            # np.r_[O[1] + xxi*eax[1], Dyi],
            # np.r_[O[2] + xxi*eax[2], Dzi],
            # ls='--',
            # marker='.',
            # c=l0.get_color(),
        # )

    # cents
    ax2.plot(
        np.r_[O[0], O[0] - rc*nin[0]],
        np.r_[O[1], O[1] - rc*nin[1]],
        np.r_[O[2], O[2] - rc*nin[2]],
        'ob',
    )

    # axe
    ax2.plot(
        O[0] + 5*xmax*np.r_[-1, 1] * eax[0],
        O[1] + 5*xmax*np.r_[-1, 1] * eax[1],
        O[2] + 5*xmax*np.r_[-1, 1] * eax[2],
        '--k',
    )

    # circles
    ang = thetamax*np.linspace(-1, 1, 11)
    ax2.plot(
        O[0] - xmax*eax[0] + rc*(-np.cos(ang)*nin[0] + np.sin(ang)*erot[0]),
        O[1] - xmax*eax[1] + rc*(-np.cos(ang)*nin[1] + np.sin(ang)*erot[1]),
        O[2] - xmax*eax[2] + rc*(-np.cos(ang)*nin[2] + np.sin(ang)*erot[2]),
        '--k',
    )
    ax2.plot(
        O[0] + xmax*eax[0] + rc*(-np.cos(ang)*nin[0] + np.sin(ang)*erot[0]),
        O[1] + xmax*eax[1] + rc*(-np.cos(ang)*nin[1] + np.sin(ang)*erot[1]),
        O[2] + xmax*eax[2] + rc*(-np.cos(ang)*nin[2] + np.sin(ang)*erot[2]),
        '--k',
    )

    Ex = pt_x + roots*(pts_x[ii] - pt_x)
    Ey = pt_y + roots*(pts_y[ii] - pt_y)
    Ez = pt_z + roots*(pts_z[ii] - pt_z)

    OEax = (Ex - O[0])*eax[0] + (Ey - O[1])*eax[1] + (Ez - O[2])*eax[2]
    Er = np.sqrt(
        ((Ex - O[0]) - OEax*eax[0])**2
        + ((Ey - O[1]) - OEax*eax[1])**2
        + ((Ez - O[2]) - OEax*eax[2])**2
    )
    if roots.size == 1:
        warnings.warn(f"Er: {Er}")
    else:
        print(Er)
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
