# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import scipy.sparse as scpsp


_LOPERATORS_INT = [
    'D0',
    'D0N2',
    'D1N2',
    'D2N2',
]


# #############################################################################
# #############################################################################
#                           Utilities
# #############################################################################


def _check_var(var, varname, types=None, default=None, allowed=None):
    if var is None:
        var = default

    if types is not None:
        if not isinstance(var, types):
            msg = (
                f"Arg {varname} must be of type {types}!\n"
                f"Provided: {type(var)}"
            )
            raise Exception(msg)

    if allowed is not None:
        if var not in allowed:
            msg = (
                f"Arg {varname} must be in {allowed}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)
    return var


# #############################################################################
# #############################################################################
#                   Mesh2DRect - bsplines - operators
# #############################################################################


def _get_mesh2dRect_operators_check(
    deg=None,
    operator=None,
    geometry=None,
    sparse_fmt=None,
):

    # deg
    deg = _check_var(
        deg, 'deg',
        types=int,
        allowed=[0, 1, 2, 3],
    )

    # operator
    operator = _check_var(
        operator, 'operator',
        default='D0',
        types=str,
        allowed=_LOPERATORS_INT,
    )

    # geometry
    geometry = _check_var(
        geometry, 'geometry',
        default='toroidal',
        types=str,
        allowed=['toroidal', 'linear'],
    )

    # sparse_fmt
    sparse_fmt = _check_var(
        sparse_fmt, 'sparse_fmt',
        default='csr',
        types=str,
        allowed=['dia', 'csr', 'csc', 'lil'],
    )

    return operator, geometry, sparse_fmt


def get_mesh2dRect_operators(
    operator=None,
    geometry=None,
    deg=None,
    knotsx=None,
    knotsy=None,
    overlap=None,
    sparse_fmt=None,
):

    # ------------
    # check inputs

    operator, geometry, sparse_fmt = _get_mesh2dRect_operators_check(
        deg=deg,
        operator=operator,
        geometry=geometry,
        sparse_fmt=sparse_fmt,
    )

    # ------------
    # prepare

    nx, ny = knotsx.shape[1], knotsy.shape[1]
    kR = np.tile(knotsx, ny)
    kZ = np.repeat(knotsy, nx, axis=1)
    nbs = nx*ny

    if 'N' in operator and deg >= 1:
        # get intersection indices array
        nbtot = np.sum(overlap >= 0)

        # prepare data and indices arrays
        data = np.full((nbtot,), np.nan)
        row = np.full((nbtot,), np.nan)
        column = np.full((nbtot,), np.nan)

    # ------------
    # D0

    if operator == 'D0' and deg == 0 and geometry == 'linear':

        opmat = (kR[1, :] - kR[0, :]) * (kZ[1, :] - kZ[0, :])

    elif operator == 'D0' and deg == 0 and geometry == 'toroidal':

        opmat = 0.5 * (kR[1, :]**2 - kR[0, :]**2) * (kZ[1, :] - kZ[0, :])

    elif operator == 'D0' and deg == 1 and geometry == 'linear':

        opmat = 0.25 * (kR[2, :] - kR[0, :]) * (kZ[2, :] - kZ[0, :])

    elif operator == 'D0' and deg == 1 and geometry == 'toroidal':

        opmat = (
            0.5
            * (kR[2, :]**2 - kR[0, :]**2 + kR[1, :]*(kR[2, :]-kR[0, :]))
            * (kZ[2, :] - kZ[0, :])
        ) / 6.

    elif operator == 'D0' and deg == 2:

        iZ1 = (kZ[1, :] - kZ[0, :])**2 / (3.*(kZ[2, :] - kZ[0, :]))
        iZ21 = (
            (
                kZ[2, :]**2
                - 2*kZ[1, :]**2
                + kZ[1, :]*kZ[2, :]
                + 3.*kZ[0, :]*(kZ[1,:] - kZ[2,:])
            )
            / (6.*(kZ[2, :]-kZ[0, :]))
        )
        iZ22 = (
            (
                -2.*kZ[2, :]**2
                + kZ[1, :]**2
                + kZ[1, :]*kZ[2, :]
                + 3.*kZ[3,:]*(kZ[2, :] - kZ[1,:])
            )
            / (6.*(kZ[3, :] - kZ[1, :]))
        )
        iZ3 = (kZ[3, :] - kZ[2, :])**2 / (3.*(kZ[3, :] - kZ[1, :]))

        if geometry == 'linear':
            iR1 = (kR[1, :] - kR[0, :])**2 / (3.*(kR[2, :] - kR[0, :]))
            iR21 = (
                (
                    kR[2, :]**2
                    - 2. * kR[1, :]**2
                    + kR[1, :] * kR[2, :]
                    + 3. * kR[0, :] * (kR[1,:] - kR[2,:])
                )
                / (6. * (kR[2, :] - kR[0, :]))
            )
            iR22 = (
                (
                    -2. * kR[2, :]**2
                    + kR[1, :]**2
                    + kR[1, :] * kR[2, :]
                    + 3. * kR[3,:] * (kR[2, :] - kR[1,:])
                )
                / (6.*(kR[3, :] - kR[1, :]))
            )
            iR3 = (kR[3, :] - kR[2, :])**2 / (3.*(kR[3, :] - kR[1, :]))

        else:
            iR1 = (
                (
                    3.*kR[1, :]**3
                    + kR[0, :]**3
                    - 5.*kR[0, :] * kR[1, :]**2
                    + kR[0, :]**2 * kR[1, :]
                )
                / (12. * (kR[2, :] - kR[0, :]))
            )
            iR21 = (
                (
                    kR[2, :]**3
                    - 3.*kR[1, :]**3
                    + kR[1, :]**2 * kR[2, :]
                    + kR[1, :] * kR[2, :]**2
                    - 2.*kR[0, :] * kR[2, :]**2
                    - 2.*kR[0, :] * kR[1, :] * kR[2, :]
                    + 4.*kR[0, :] * kR[1, :]**2
                )
                / (12. * (kR[2, :] - kR[0, :]))
            )
            iR22 = (
                (
                    -3.*kR[2, :]**3
                    + kR[1, :]**3
                    + kR[1, :] * kR[2, :]**2
                    + kR[1, :]**2 * kR[2, :]
                    + 4.*kR[2, :]**2 * kR[3, :]
                    - 2.*kR[1, :]*kR[2, :]*kR[3, :]
                    - 2.*kR[1, :]**2 * kR[3, :]
                )
                / (12. * (kR[3, :] - kR[1, :]))
            )
            iR3 = (
                (
                    kR[3, :]**3
                    + 3.*kR[2, :]**3
                    - 5.*kR[2, :]**2 * kR[3, :]
                    + kR[2, :]*kR[3, :]**2
                ) / (12. * (kR[3, :] - kR[1, :]))
            )

        opmat = (iR1 + iR21 + iR22 + iR3) * (iZ1 + iZ21 + iZ22 + iZ3)

    elif operator == 'D0' and deg == 3:

        raise NotImplementedError("Integral D0 not implemented for deg=3 yet!")

    # ------------
    # D0N2

    elif operator == 'D0N2' and deg == 0:

        iZ = kZ[1, :] - kZ[0, :]
        if geometry == 'linear':
            iR = kR[1, :] - kR[0, :]
        else:
            iR = 0.5 * (kR[1, :]**2 - kR[0, :]**2)

        opmat = scpsp.diags(
            [iR*iZ],
            [0],
            shape=None,
            format=sparse_fmt,
            dtype=float,
        )

    elif operator == 'D0N2' and deg == 1:

        d0Z = (kZ[2, :] - kZ[0, :]) / 3.

        if geometry == 'linear':
            d0R = (kR[2, :] - kR[0, :]) / 3.
        else:
            d0R = _D0N2_Deg1_full_toroidal(kR)

        # set diagonal elements
        data[:nbs] = d0Z*d0R
        row[:nbs] = np.arange(nbs)
        column[:nbs] = np.arange(nbs)

        # set non-diagonal elements
        i0 = nbs
        for ii in range(nbs):
            overlapi = overlap[:, ii][overlap[:, ii] >= 0]
            for jj in overlapi:
                if jj == ii:
                    continue

                # get overlapping knots
                overR = np.intersect1d(kR[:, ii], kR[:, jj])
                overZ = np.intersect1d(kZ[:, ii], kZ[:, jj])

                if overR.size == 2 and geometry == 'linear':
                    iRj = (overR[1] - overR[0]) / 6.
                elif overR.size == 2 and geometry == 'toroidal':
                    iRj = (overR[1]**2 - overR[0]**2) / 12.
                elif overR.size == 3:
                    iRj = d0R[ii]

                if overZ.size == 2:
                    iZj = (overZ[1] - overZ[0]) / 6.
                elif overZ.size == 3:
                    iZj = d0Z[ii]

                data[i0] = iRj * iZj
                row[i0] = ii
                column[i0] = jj
                i0 += 1

        opmat = scpsp.csr_matrix((data, (row, column)), shape=(nbs, nbs))

    elif operator == 'D0N2' and deg == 2:

        d0Z = _D0N2_Deg2_full_linear(kZ)

        if geometry == 'linear':
            d0R = _D0N2_Deg2_full_linear(kR)
        else:
            d0R = _D0N2_Deg2_full_toroidal(kR)

        # set diagonal elements
        data[:nbs] = d0Z*d0R
        row[:nbs] = np.arange(nbs)
        column[:nbs] = np.arange(nbs)

        # set non-diagonal elements
        i0 = nbs
        for ii in range(nbs):
            overlapi = overlap[:, ii][overlap[:, ii] >= 0]
            for jj in overlapi:
                if jj == ii:
                    continue

                # get overlapping knots
                overR = np.intersect1d(kR[:, ii], kR[:, jj])
                overZ = np.intersect1d(kZ[:, ii], kZ[:, jj])

                if overR.size == 2 and geometry == 'linear':
                    iRj = _D0N2_Deg2_2_linear(kx)   # x1, x4 ?
                elif overR.size == 2 and geometry == 'toroidal':
                    iRj = _D0N2_Deg2_2_toroidal(kx)   # x1, x4 ?
                elif overR.size == 3 and geometry == 'linear':
                    iRj = None
                elif overR.size == 3 and geometry == 'toroidal':
                    iRj = None
                elif overR.size == 4:
                    iRj = d0R[ii]

                if overZ.size == 2:
                    iZj = _D0N2_Deg2_2_linear(kx)   # x1, x4 ?
                elif overZ.size == 3:
                    iZj = _D0N2_Deg2_3_linear(kx)   # x1, x4 ?
                elif overZ.size == 4:
                    iZj = d0Z[ii]

                data[i0] = iRj * iZj
                row[i0] = ii
                column[i0] = jj
                i0 += 1

        opmat = scpsp.csr_matrix((data, (row, column)), shape=(nbs, nbs))

    elif operator == 'D0N2' and deg == 3:

        raise NotImplementedError("Integral D0N2 not implemented for deg=3!")

    # ------------
    # D1N2

    elif operator == 'D1N2' and deg == 1:

        raise NotImplementedError("Integral D1N2 not implemented for deg=1!")

    elif operator == 'D1N2' and deg == 2:

        raise NotImplementedError("Integral D1N2 not implemented for deg=2!")

    elif operator == 'D1N2' and deg == 3:

        raise NotImplementedError("Integral D1N2 not implemented for deg=3!")

    # ------------
    # D2N2

    elif operator == 'D2N2' and deg == 2:

        raise NotImplementedError("Integral D2N2 not implemented for deg=2!")

    elif operator == 'D2N2' and deg == 3:

        raise NotImplementedError("Integral D2N2 not implemented for deg=3!")

    # ------------
    # D3N2

    elif operator == 'D3N2' and deg == 3:

        raise NotImplementedError("Integral D3N2 not implemented for deg=3!")

    return operator, opmat



# #############################################################################
# #############################################################################
#               Operator sub-routines 
# #############################################################################


def _D0N2_Deg1_full_toroidal(kx):
    """ from 1d knots, return int_0^2 x b**2(x) dx """
    assert kx.shape[0] == 3
    return (
        (3. * kx[1]**3 - 5.*kx[0]*kx[1]**2 + kx[1]*kx[0]**2 + kx[0]**3)
        / (12. * (kx[1] - kx[0]))
        + (3.*kx[1]**3 - 5.*kx[2]*kx[1]**2 + kx[1]*kx[2]**2 + kx[2]**3)
        /(12. * (kx[2] - kx[1]))
    )


def _D0N2_Deg2_full_linear(kx):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    assert kx.shape[0] == 4
    return (
        (kx[1] - kx[0])**3 / (5.*(kx[2] - kx[0])**2)
        + (
            (kx[2] - kx[1])
            * (
                10.*kx[0]**2 + 6.*kx[1]**2 + 3.*kx[1]*kx[2] + kx[2]**2
                - 5.*kx[0]*(3.*kx[1] + kx[2])
            ) / (30.*(kx[2] - kx[0])**2)
        )
        + (
            (kx[2] - kx[1])
            * (
                10.*kx[3]**2 + 6.*kx[2]**2 + 3.*kx[1]*kx[2] + kx[1]**2
                - 5.*kx[3]*(3.*kx[2] + kx[1])
            ) / (30.*(kx[3] - kx[1])**2)

        )
        + (
            (kx[1] - kx[2])
            * (
                -3.*kx[1]**2 - 4.*kx[1]*kx[2] - 3.*kx[2]**2
                + 5.*kx[0]*(kx[1] + kx[2] - 2.*kx[3])
                + 5.*kx[3]*(kx[1] + kx[2])
            ) / (60.*(kx[2] - kx[0])*(kx[3] - kx[1]))
        )
        - (kx[3] - kx[2])**3 / (5.*(kx[3] - kx[1])**2)
    )


def _D0N2_Deg2_full_toroidal(kx):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    assert kx.shape[0] == 4
    return (
        (5.*kx[1] + kx[0])*(kx[1] - kx[0])**3 / (30.*(kx[2] - kx[0])**2)
        + (kx[2] - kx[1]) * (
            (
                10*kx[1]**3 + 6.*kx[1]**2*kx[2] + 3.*kx[1]*kx[2]**2
                + kx[2]**3 + 5.*kx[0]**2*(3.*kx[1] + kx[2])
                - 4.*kx[0]*(6.*kx[1]**2 + 3.*kx[1]*kx[2] + kx[2]**2)
            ) / (60.*(kx[2] - kx[0])**2)
            + (
                10*kx[2]**3 + 6.*kx[2]**2*kx[1] + 3.*kx[2]*kx[1]**2
                + kx[1]**3 + 5.*kx[3]**2*(3.*kx[2] + kx[1])
                - 4.*kx[3]*(6.*kx[2]**2 + 3.*kx[2]*kx[1] + kx[1]**2)
            ) / (60.*(kx[3] - kx[1])**2)
            + (
                -2.*kx[1]**3 - 2.*kx[2]**3
                -3.*kx[1]*kx[2]*(kx[1] + kx[2])
                - 5.*kx[0]*kx[3]*(kx[1] + kx[2])
                + (kx[0] + kx[3])*(3.*kx[2]**2 + 4.*kx[1]*kx[2] + 3.*kx[1]**2)
            ) / (30.*(kx[3] - kx[1])*(kx[2] - kx[0]))
        )
        + (5.*kx[2] + kx[3])*(kx[3] - kx[2])**3 / (30.*(kx[3] - kx[1])**2)
    )


def _D0N2_Deg2_2_linear(kx):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    assert kx.shape[0] == 4
    # TBF
    return (
        (kx[2] - kx[1])**3
        / (30.*(kx[3]-kx[1])*(kx[2]-kx[0]))
    )


def _D0N2_Deg2_2_toroidal(kx):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    assert kx.shape[0] == 4
    # TBF
    return (
        (kx[2] + kx[1])*(kx[2] - kx[1])**3
        / (60.*(kx[3] - kx[1])*(kx[2] - kx[0]))
    )
