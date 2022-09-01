

import numpy as np


# ##############################################################
# ##############################################################
#           Finding reflection points
# ##############################################################


def pts_to_pt_planar(
    pt_x=None,
    pt_y=None,
    pt_z=None,
    # pts
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # plane
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    exth0=None,
    exth1=None,
    return_k=None,
    return_xyz=None,
):
    """

    Based on Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

    """

    # ------------------------------------------
    # Get local coordinates of reflection points

    # get parameters for k
    CA = np.r_[pt_x - cent[0], pt_y - cent[1], pt_z - cent[2]]
    CAdotn = np.sum(CA * nin)

    ABx = pts_x - pt_x
    ABy = pts_y - pt_y
    ABz = pts_z - pt_z

    length = np.sqrt(ABx**2 + ABy**2 + ABz**2)

    # get k
    kk = CAdotn / (2*CAdotn + (nin[0] * ABx + nin[1]*ABy + nin[2]*ABz))

    # get E
    Ex = pt_x + kk * ABx
    Ey = pt_y + kk * ABy
    Ez = pt_z + kk * ABz

    # x0, x1
    x0 = (Ex - cent[0])*e0[0] + (Ey - cent[1])*e0[1] + (Ez - cent[2])*e0[2]
    x1 = (Ex - cent[0])*e1[0] + (Ey - cent[1])*e1[1] + (Ez - cent[2])*e1[2]

    if return_xyz:
        # get D
        Dx = cent[0] + x0 * e0[0] + x1*e1[0]
        Dy = cent[1] + x0 * e0[1] + x1*e1[1]
        Dz = cent[2] + x0 * e0[2] + x1*e1[2]

        return x0, x1, Dx, Dy, Dz
    else:
        return x0, x1


def pts_to_pt_spherical(
    pt_x=None,
    pt_y=None,
    pt_z=None,
    # pts
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # plane
    cent=None,
    rc=None,
    nin=None,
    e0=None,
    e1=None,
    exth0=None,
    exth1=None,
    return_x01=None,
):
    """

    Based on Notes_Upgrades/raytracing_surface3draytracing_surface3d.pdf

    """

    # ------------------------------------------
    # Get local coordinates of reflection points

    # get parameters for k
    O = cent + rc * nin
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
        rr = np.root(np.r_[A[ii], B[ii], C[ii], D[ii], E[ii]])
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
