# -*- coding: utf-8 -*-


import warnings


import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt


import Polygon as plg
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

    key, cls = coll.get_optics_cls(optics=key)
    key, cls = key[0], cls[0]
    dgeom = coll.dobj[cls][key]['dgeom']

    lcls = ['crystal', 'grating']
    if cls not in lcls:
        msg = (
            "Wrong class for reflections:\n"
            f"\t -allowed: {lcls}\n"
            f"\t -cls: {cls}\n"
        )
        raise Exception(msg)

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
            strict=None,
            return_xyz=None,
            return_x01=None,
            debug=None,
            # timing
            dt=None,
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

            if return_xyz and return_x01:
                return Dx, Dy, Dz, x0, x1
            elif return_xyz:
                return Dx, Dy, Dz
            elif return_x01:
                return x0, x1

    # ----------------
    #   Cylindrical
    # ----------------

    elif dgeom['type'] == 'cylindrical':

        iplan = np.isinf(dgeom['curve_r']).nonzero()[0][0]
        eax = ['e0', 'e1'][iplan]
        erot = ['e0', 'e1'][1-iplan]
        
        rc = dgeom['curve_r'][1 - iplan]
        rcs = np.sign(rc)
        rca = np.abs(rc)

        def pts2pt(
            pt_x=None,
            pt_y=None,
            pt_z=None,
            # pts
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # surface
            O=dgeom['cent'] + dgeom['nin'] * rc,
            rcs=rcs,
            rca=rca,
            eax=dgeom[eax],
            erot=dgeom[erot],
            iplan=iplan,
            # limits
            thetamax=dgeom['extenthalf'][1-iplan],
            xmax=dgeom['extenthalf'][iplan],
            # local coordinates
            nin=dgeom['nin'],
            # return
            strict=True,
            return_xyz=None,
            return_x01=None,
            # number of k for interpolation
            nk=None,
            debug=True,
            # timing
            dt=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """

            # ------------------------------------------
            # Get local coordinates of reflection points

            A = np.r_[pt_x, pt_y, pt_z]
            Dx, Dy, Dz, xx, theta = _common_prepare(nb=pts_x.size)

            if strict:
                xmargin = 1.
                thetatot = thetamax
            else:
                xmargin = 1. + 0.01
                thetatot = thetamax*(1. + 0.01)
                 
            # defining zones for kk (point , plane unit vector)
            em = np.cos(thetatot)*(erot) + rcs * np.sin(thetatot)*(-nin)
            eM = np.cos(thetatot)*(erot) - rcs * np.sin(thetatot)*(-nin) 

            lzones = [
                (
                    (O - (xmax*xmargin)*eax, eax),
                    (O + (xmax*xmargin)*eax, -eax),
                    (O, em),
                    (O, -eM),
                ),
                
            ]
            
            if rcs > 0:
                lzones.append((
                    (O - (xmax*xmargin)*eax, eax),
                    (O + (xmax*xmargin)*eax, -eax),
                    (O, -em),
                    (O, eM),
                ))

            # loop on pts 
            for ii in range(pts_x.size):
                B = np.r_[pts_x[ii], pts_y[ii], pts_z[ii]]
                AB = B - A
                eAB = AB / np.linalg.norm(AB)

                # compute kmin, kmax
                kk = _kminmax_plane(
                    A=A,
                    B=B,
                    lzones=lzones,
                    ii=ii,
                    nin=nin,
                    nk=nk,
                    # timing
                    dt=dt,
                )

                if kk is None:
                    continue

                # nin
                (
                    nix, niy, niz,
                    Dxi, Dyi, Dzi,
                    # D1x, D1y, D1z,
                    xxi,
                ) = _get_Dnin_from_k_cyl(
                    O=O, eax=eax,
                    Ax=pt_x, Ay=pt_y, Az=pt_z,
                    Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                    kk=kk,
                    rcs=rcs,
                    rca=rca,
                    nin=nin,
                )

                # derive DAn, DB, DBn, DA
                eq = _get_DADB(
                    Ax=pt_x, Ay=pt_y, Az=pt_z,
                    Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                    Dx=Dxi, Dy=Dyi, Dz=Dzi,
                    nix=nix, niy=niy, niz=niz,
                )
                iok = np.isfinite(eq)
                kk = kk[iok]
                eq = eq[iok]

                # find indices of sign changes
                i0 = (eq[:-1] * eq[1:] < 0).nonzero()[0]
                if len(i0) == 1:
                    roots = np.r_[
                        kk[i0]
                        - eq[i0]*(kk[i0+1] - kk[i0])
                        / (eq[i0+1] - eq[i0])
                    ]

                    # roots = scpinterp.InterpolatedUnivariateSpline(
                    # kk,
                    # eq,
                    # k=3,
                    # ).roots()
                else:
                    continue

                # nin, xx, D
                (
                    nix, niy, niz,
                    Dxi, Dyi, Dzi, xxi,
                ) = _get_Dnin_from_k_cyl(
                    O=O, eax=eax,
                    Ax=pt_x, Ay=pt_y, Az=pt_z,
                    Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                    kk=roots[0],
                    rcs=rcs,
                    rca=rca,
                    nin=nin,
                )

                # theta, xx
                thetai = rcs * np.arctan2(
                    -(nix*erot[0] + niy*erot[1] + niz*erot[2]),
                    nix*nin[0] + niy*nin[1] + niz*nin[2],
                )

                if strict is True:
                    if np.abs(xxi) > xmax or np.abs(thetai) > thetamax:
                        continue

                Dx[ii], Dy[ii], Dz[ii] = Dxi, Dyi, Dzi
                xx[ii], theta[ii] = xxi, thetai

            # safety check
            iok = np.isfinite(Dx)
            Dx = Dx[iok]
            Dy = Dy[iok]
            Dz = Dz[iok]
            theta = theta[iok]
            xx = xx[iok]

            # TODO : find a way to hanbdle complex (self-intersection polygons)
            # Exceptionally a polygon (xx, theta) can self-intersect

            # return
            if return_xyz and return_x01:
                if iplan == 0:
                    return Dx, Dy, Dz, xx, theta
                else:
                    return Dx, Dy, Dz, theta, xx
            elif return_xyz:
                return Dx, Dy, Dz
            elif return_x01:
                if iplan == 0:
                    return xx, theta
                else:
                    return theta, xx

    # ----------------
    #   Spherical
    # ----------------

    elif dgeom['type'] == 'spherical':

        rc = dgeom['curve_r'][0]
        rcs = np.sign(rc)
        rca = np.abs(rc)

        def pts2pt(
            pt_x=None,
            pt_y=None,
            pt_z=None,
            # pts
            pts_x=None,
            pts_y=None,
            pts_z=None,
            # surface
            O=dgeom['cent'] + rc*dgeom['nin'],
            rcs=rcs,
            rca=rca,
            # limits
            dthetamax=dgeom['extenthalf'][0],
            phimax=dgeom['extenthalf'][1],
            # local coordinates
            nin=dgeom['nin'],
            e0=dgeom['e0'],
            e1=dgeom['e1'],
            # return
            strict=None,
            return_xyz=None,
            return_x01=None,
            nk=None,
            debug=None,
            # timing
            dt=None,
        ):
            """

            cf. Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

            """

            # ------------------------------------------
            # Get local coordinates of reflection points

            # get parameters for k
            A = np.r_[pt_x, pt_y, pt_z]
            Dx, Dy, Dz, phi, dtheta = _common_prepare(nb=pts_x.size)

             # defining zones for kk (point , plane unit vector)
            epm = np.cos(phimax)*e0 + rcs * np.sin(phimax)*(-nin)
            epM = np.cos(phimax)*e0 - rcs * np.sin(phimax)*(-nin)

            etm = np.cos(dthetamax)*e1 + rcs * np.sin(dthetamax)*(-nin)
            etM = np.cos(dthetamax)*e1 - rcs * np.sin(dthetamax)*(-nin)

            lzones = [
                (
                    (O, epm),
                    (O, -epM),
                    (O, etm),
                    (O, -etM),
                ),
            ]
            
            if rcs > 0:
                lzones.append((
                    (O, -epm),
                    (O, epM),
                    (O, -etm),
                    (O, etM),
                ))

            # loop on pts
            for ii in range(pts_x.size):
                B = np.r_[pts_x[ii], pts_y[ii], pts_z[ii]]
                AB = B - A
                eAB = AB / np.linalg.norm(AB)

                # compute kmin, kmax
                kk = _kminmax_plane(
                    A=A,
                    B=B,
                    lzones=lzones,
                    ii=ii,
                    nin=nin,
                    nk=nk,
                    # timing
                    dt=dt,
                )

                if kk is None:
                    continue

                # nin, xx, D
                (
                    nix, niy, niz,
                    Dxi, Dyi, Dzi,
                ) = _get_Dnin_from_k_sph(
                    O=O,
                    Ax=pt_x, Ay=pt_y, Az=pt_z,
                    Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                    kk=kk,
                    rcs=rcs,
                    rca=rca,
                    nin=nin,
                )

                # derive DAn, DB, DBn, DA
                eq = _get_DADB(
                    Ax=pt_x, Ay=pt_y, Az=pt_z,
                    Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                    Dx=Dxi, Dy=Dyi, Dz=Dzi,
                    nix=nix, niy=niy, niz=niz,
                )
                iok = np.isfinite(eq)
                kk = kk[iok]
                eq = eq[iok]

                # find indices of sign changes
                i0 = (eq[:-1] * eq[1:] < 0).nonzero()[0]
                if len(i0) == 1:
                    roots = np.r_[
                        kk[i0]
                        - eq[i0]*(kk[i0+1] - kk[i0])
                        / (eq[i0+1] - eq[i0])
                    ]

                    # roots = scpinterp.InterpolatedUnivariateSpline(
                    # kk,
                    # eq,
                    # k=3,
                    # ).roots()
                else:
                    continue

                # nin, xx, D
                (
                    nix, niy, niz,
                    Dxi, Dyi, Dzi,
                ) = _get_Dnin_from_k_sph(
                    O=O,
                    Ax=pt_x, Ay=pt_y, Az=pt_z,
                    Bx=pts_x[ii], By=pts_y[ii], Bz=pts_z[ii],
                    kk=roots[0],
                    rcs=rcs,
                    rca=rca,
                    nin=nin,
                )

                # local coordinates
                dthetai = - rcs * np.arcsin(nix*e1[0] + niy*e1[1] + niz*e1[2])
                phii = - rcs * np.arcsin(
                    (nix*e0[0] + niy*e0[1] + niz*e0[2]) / np.cos(dthetai)
                )

                # ----------

                if strict is True:
                    if np.abs(phii) > phimax or np.abs(dthetai) > dthetamax:
                        continue

                Dx[ii], Dy[ii], Dz[ii] = Dxi, Dyi, Dzi
                phi[ii], dtheta[ii] = phii, dthetai

            # safety check
            iok = np.isfinite(Dx)
            Dx = Dx[iok]
            Dy = Dy[iok]
            Dz = Dz[iok]
            dtheta = dtheta[iok]
            phi = phi[iok]

            # TODO : find a way to hanbdle complex (self-intersection polygons)
            # Exceptionally a polygon (xx, theta) can self-intersect

            # return
            if return_xyz and return_x01:
                return Dx, Dy, Dz, phi, dtheta
            elif return_xyz:
                return Dx, Dy, Dz
            elif return_x01:
                return phi, dtheta

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


# Not used
def _mindist_2lines(A=None, B=None, O=None, eax=None):
    OAe = np.cross(A - O, eax)
    ABe = np.cross(B - A, eax)
    OAe2 = np.sum(OAe**2)
    ABe2 = np.sum(ABe**2)
    OAeABe = np.sum(OAe * ABe)
    kk = -OAeABe/ABe2
    return kk, OAe2 - OAeABe**2/ABe2


def _kminmax_ptinz(pt=None, lp=None):
    return np.all([
        np.sum((pt - oo)*ep) > 0 for oo, ep in lp
    ])


def _kminmax_ptinz2(pts=None, lp=None):
    return np.all(
        [
            np.sum((pts - oo[:, None])*ep[:, None], axis=0) > 0
            for oo, ep in lp
        ],
        axis=0,
    )


def _kminmax_plane(
    A=None,
    B=None,
    lzones=None,
    kmin=0,
    kmax=1,
    nin=None,
    ii=None,
    nk=None,
    # timing
    dt=None,
):

    # ---------------
    # check inputs

    if nk is None:
        nk = 100

    AB = B - A

    # ---------------
    # get lkin, lkout

    # Intrsection with planes (same planes for all zones)
    lk = []
    for jj, (oo, ep) in enumerate(lzones[0]):
        AO = oo - A
        AOe = np.sum(AO*ep)
        ABe = np.sum(AB * ep)

        if np.abs(ABe) < 1e-16:
            continue
        kk = AOe / ABe
        if kk < 0 or kk > 1:
            continue
        lk.append(kk)

    kk = np.r_[0., np.sort(lk) + 1.e-13, 1.]

    kin = np.array([
        _kminmax_ptinz2(
            pts=A[:, None] + kk[None, :]*AB[:, None],
            lp=lp,
        )
        for lp in lzones
    ])

    iok = np.any(kin, axis=1)

    # trivial case 1
    if not np.any(iok):
        return None

    kin = kin[iok, :]

    # trivial case 2
    if np.any(np.all(kin, axis=1)):
        return np.linspace(0, 1, nk)

    # non-trivial cases
    iin = kin.nonzero()[1]
    iout = kin.shape[1] - kin[:, ::-1].nonzero()[1]

    if iin.shape[0] > 1:
        import pdb; pdb.set_trace()     # DB

    idel = [ii for ii in range(iin.size) if iin[ii] in iout]
    if len(idel) > 0:
        import pdb; pdb.set_trace()     # DB
        np.delete(kin, idel)
        np.delete(kin, idel)

    if iin.size == 0:
        return np.linspace(0, 1, nk)
    elif iin.size == 1:
        return np.linspace(kk[iin[0]], kk[iout[0]], nk)
    else:
        return np.concatenate(
            tuple([
                np.linspace(kk[iin[ii]], kk[iout[ii]], nk)
                for ii in range(iin.size)
            ])
        )


def _get_Dnin_from_k_cyl(
    O=None, eax=None,
    Ax=None, Ay=None, Az=None,
    Bx=None, By=None, Bz=None,
    kk=None,
    rcs=None,
    rca=None,
    nin=None,
):
    Ex = Ax + kk*(Bx - Ax)
    Ey = Ay + kk*(By - Ay)
    Ez = Az + kk*(Bz - Az)

    xx = (Ex - O[0])*eax[0] + (Ey - O[1])*eax[1] + (Ez - O[2])*eax[2]
    nix = (Ex - O[0]) - xx*eax[0]
    niy = (Ey - O[1]) - xx*eax[1]
    niz = (Ez - O[2]) - xx*eax[2]
    ninorm = np.sqrt(nix**2 + niy**2 + niz**2)
    nix = - rcs * nix / ninorm
    niy = - rcs * niy / ninorm
    niz = - rcs * niz / ninorm

    # handle 2 zones
    sign = np.sign(nix*nin[0] + niy*nin[1] + niz*nin[2])
    if rcs > 0:   
        nix = sign*nix
        niy = sign*niy
        niz = sign*niz
    else:
        assert np.all(sign > 0.)

    Dx = O[0] + xx*eax[0] - rcs * rca*nix
    Dy = O[1] + xx*eax[1] - rcs * rca*niy
    Dz = O[2] + xx*eax[2] - rcs * rca*niz
    return nix, niy, niz, Dx, Dy, Dz, xx


def _get_Dnin_from_k_sph(
    O=None,
    Ax=None, Ay=None, Az=None,
    Bx=None, By=None, Bz=None,
    kk=None,
    rcs=None,
    rca=None,
    nin=None,
):
    Ex = Ax + kk*(Bx - Ax)
    Ey = Ay + kk*(By - Ay)
    Ez = Az + kk*(Bz - Az)

    ninorm = np.sqrt((Ex - O[0])**2 + (Ey - O[1])**2 + (Ez - O[2])**2)
    nix = - rcs * (Ex - O[0]) / ninorm
    niy = - rcs * (Ey - O[1]) / ninorm
    niz = - rcs * (Ez - O[2]) / ninorm

    sign = np.sign(nix*nin[0] + niy*nin[1] + niz*nin[2])
    if rcs > 0:   
        nix = sign*nix
        niy = sign*niy
        niz = sign*niz
    else:
        assert np.all(sign > 0.)

    Dx = O[0] - rcs * rca*nix
    Dy = O[1] - rcs * rca*niy
    Dz = O[2] - rcs * rca*niz
    return nix, niy, niz, Dx, Dy, Dz


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
    ra=None,
    rb=None,
    **kwdargs,
):

    thetai = np.arctan2(
        (nin[1]*niz - nin[2]*niy)*eax[0]
        + (nin[2]*nix - nin[0]*niz)*eax[1]
        + (nin[0]*niy - nin[1]*nix)*eax[2],
        nix*nin[0] + niy*nin[1] + niz*nin[2],
    )

    # ----------------
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

    # ----------------
    #  plot equation

    ax1 = fig.add_subplot(132)
    ax1.plot(
        kk, eq, '.-k',
        kk, ra*kk + rb, '-b',
    )
    ax1.axhline(0, ls='--', c='k')
    for rr in roots:
        ax1.axvline(rr, c='r', ls='--')

    # ----------------
    #  plot geometry

    ax2 = fig.add_subplot(133, projection='3d')
    ax2.plot(
        np.r_[pt_x, pts_x[ii]],
        np.r_[pt_y, pts_y[ii]],
        np.r_[pt_z, pts_z[ii]],
        '.-k',
    )

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
    print(np.sum(DA*DE) / (nA*nE))
    print(np.sum(DB*DE) / (nB*nE))

    import pdb; pdb.set_trace()     # DB
