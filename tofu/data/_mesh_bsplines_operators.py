# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import scipy.sparse as scpsp


# specific
from . import _generic_check


_LOPERATORS_INT = [
    'D0',
    'D0N2',
    'D1N2',
    'D2N2',
]


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
    deg = _generic_check._check_var(
        deg, 'deg',
        types=int,
        allowed=[0, 1, 2, 3],
    )

    # operator
    operator = _generic_check._check_var(
        operator, 'operator',
        default='D0',
        types=str,
        allowed=_LOPERATORS_INT,
    )

    # geometry
    geometry = _generic_check._check_var(
        geometry, 'geometry',
        default='toroidal',
        types=str,
        allowed=['toroidal', 'linear'],
    )

    # sparse_fmt
    sparse_fmt = _generic_check._check_var(
        sparse_fmt, 'sparse_fmt',
        default='csr',
        types=str,
        allowed=['dia', 'csr', 'csc', 'lil'],
    )

    # dim
    if operator == 'D0':
        if  geometry == 'linear':
            dim = 'origin x m2'
        else:
            dim = 'origin x m3/rad'
    elif operator == 'D0N2':
        if  geometry == 'linear':
            dim = 'origin2 x m2'
        else:
            dim = 'origin2 x m3/rad'
    elif operator == 'D1N2':
        if  geometry == 'linear':
            dim = 'origin2'
        else:
            dim = 'origin2 x m/rad'
    elif operator == 'D2N2':
        if  geometry == 'linear':
            dim = 'origin2 / m2'
        else:
            dim = 'origin2 / (m2.rad)'

    return operator, geometry, sparse_fmt, dim


def get_mesh2dRect_operators(
    operator=None,
    geometry=None,
    deg=None,
    knotsx_mult=None,
    knotsy_mult=None,
    knotsx_per_bs=None,
    knotsy_per_bs=None,
    overlap=None,
    sparse_fmt=None,
):

    # ------------
    # check inputs

    operator, geometry, sparse_fmt, dim = _get_mesh2dRect_operators_check(
        deg=deg,
        operator=operator,
        geometry=geometry,
        sparse_fmt=sparse_fmt,
    )

    # ------------
    # prepare

    nx, ny = knotsx_per_bs.shape[1], knotsy_per_bs.shape[1]
    kR = np.tile(knotsx_per_bs, ny)
    kZ = np.repeat(knotsy_per_bs, nx, axis=1)
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

        assert i0 == nbtot
        opmat = scpsp.csr_matrix((data, (row, column)), shape=(nbs, nbs))

    elif operator == 'D0N2' and deg == 2:

        # pre-compute integrals
        iR = _D0N2_Deg2(knotsx_mult, geometry=geometry)
        iZ = _D0N2_Deg2(knotsy_mult, geometry='linear')

        # set non-diagonal elements
        i0 = 0
        for ir in range(nx):
            for iz in range(ny):

                iflat = ir + iz*nx

                # general case
                overlapi = overlap[:, iflat][overlap[:, iflat] > iflat]

                # diagonal element
                data[i0] = iR[0, ir] * iZ[0, iz]
                row[i0] = iflat
                column[i0] = iflat
                i0 += 1

                # non-diagonal elements (symmetric)
                for jflat in overlapi:

                    jr = jflat % nx
                    jz = jflat // nx

                    # store (i, j) and (j, i) (symmetric matrix)
                    data[i0:i0+2] = iR[jr - ir, ir] * iZ[jz - iz, iz]
                    row[i0:i0+2] = (iflat, jflat)
                    column[i0:i0+2] = (jflat, iflat)
                    i0 += 2

                    if jr == ir - 1 and jz == iz + 1:
                        import pdb; pdb.set_trace()     # DB
                        pass


        assert i0 == nbtot
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

    return opmat, operator, geometry, dim



# #############################################################################
# #############################################################################
#               Operator sub-routines 
# #############################################################################


def _D0N2_Deg1_full_toroidal(k0, k1, k2):
    """ from 1d knots, return int_0^2 x b**2(x) dx """
    return (
        (3. * k1**3 - 5.*k0*k1**2 + k1*k0**2 + k0**3)
        / (12. * (k1 - k0))
        + (3.*k1**3 - 5.*k2*k1**2 + k1*k2**2 + k2**3)
        /(12. * (k2 - k1))
    )


def _D0N2_Deg2(knots, geometry=None):

    if geometry == 'linear':
        integ = np.array([
            # _D0N2_Deg2_2_linear(
                # np.r_[np.nan, np.nan, ky[1:-4]],
                # np.r_[np.nan, np.nan, ky[2:-3]],
                # np.r_[np.nan, np.nan, ky[3:-2]],
                # np.r_[np.nan, np.nan, ky[4:-1]],
            # )
            # _D0N2_Deg2_3_linear(
                # np.r_[np.nan, ky[:-4]],
                # np.r_[np.nan, ky[1:-3]],
                # np.r_[np.nan, ky[2:-2]],
                # np.r_[np.nan, ky[3:-1]],
                # np.r_[np.nan, ky[4:]],
            # )
            _D0N2_Deg2_full_linear(
                knots[:-3],
                knots[1:-2],
                knots[2:-1],
                knots[3:],
            ),
            _D0N2_Deg2_3_linear(
                np.r_[knots[:-4], np.nan],
                np.r_[knots[1:-3], np.nan],
                np.r_[knots[2:-2], np.nan],
                np.r_[knots[3:-1], np.nan],
                np.r_[knots[4:], np.nan],
            ),
            _D0N2_Deg2_2_linear(
                np.r_[knots[1:-4], np.nan, np.nan],
                np.r_[knots[2:-3], np.nan, np.nan],
                np.r_[knots[3:-2], np.nan, np.nan],
                np.r_[knots[4:-1], np.nan, np.nan],
            ),
        ])
    else:
        integ = np.array([
            # _D0N2_Deg2_2_toroidal(
                # np.r_[np.nan, np.nan, ky[1:-4]],
                # np.r_[np.nan, np.nan, ky[2:-3]],
                # np.r_[np.nan, np.nan, ky[3:-2]],
                # np.r_[np.nan, np.nan, ky[4:-1]],
            # )
            # _D0N2_Deg2_3_toroidal(
                # np.r_[np.nan, ky[:-4]],
                # np.r_[np.nan, ky[1:-3]],
                # np.r_[np.nan, ky[2:-2]],
                # np.r_[np.nan, ky[3:-1]],
                # np.r_[np.nan, ky[4:]],
            # )
            _D0N2_Deg2_full_toroidal(
                knots[:-3],
                knots[1:-2],
                knots[2:-1],
                knots[3:],
            ),
            _D0N2_Deg2_3_toroidal(
                np.r_[knots[:-4], np.nan],
                np.r_[knots[1:-3], np.nan],
                np.r_[knots[2:-2], np.nan],
                np.r_[knots[3:-1], np.nan],
                np.r_[knots[4:], np.nan],
            ),
            _D0N2_Deg2_2_toroidal(
                np.r_[knots[1:-4], np.nan, np.nan],
                np.r_[knots[2:-3], np.nan, np.nan],
                np.r_[knots[3:-2], np.nan, np.nan],
                np.r_[knots[4:-1], np.nan, np.nan],
            ),
        ])
    return integ


def _D0N2_Deg2_full_linear(k0, k1, k2, k3):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    intt = np.zeros((k0.size,))
    intt[1:] += (
        (k1 - k0)[1:]**3 / (5.*(k2 - k0)[1:]**2)
        + (k2 - k1)[1:]
        * (
            10.*k0**2 + 6.*k1**2 + 3.*k1*k2 + k2**2 - 5.*k0*(3.*k1 + k2)
        )[1:] / (30.*(k2 - k0)[1:]**2)
    )
    intt[1:-1] += (
        (k1 - k2)[1:-1]
        * (
            -3.*k1**2 - 4.*k1*k2 - 3.*k2**2 + 5.*k0*(k1 + k2 - 2.*k3)
            + 5.*k3*(k1 + k2)
        )[1:-1] / (60.*(k2 - k0)*(k3 - k1))[1:-1]
    )
    intt[:-1] += (
        (k2 - k1)[:-1]
        * (
            10.*k3**2 + 6.*k2**2 + 3.*k1*k2 + k1**2 - 5.*k3*(3.*k2 + k1)
        )[:-1] / (30.*(k3 - k1)[:-1]**2)
        - (k3 - k2)[:-1]**3 / (5.*(k3 - k1)[:-1]**2)
    )
    return intt


def _D0N2_Deg2_full_toroidal(k0, k1, k2, k3):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    return (
        (5.*k1 + k0)*(k1 - k0)**3 / (30.*(k2 - k0)**2)
        + (k2 - k1) * (
            (
                10*k1**3 + 6.*k1**2*k2 + 3.*k1*k2**2
                + k2**3 + 5.*k0**2*(3.*k1 + k2)
                - 4.*k0*(6.*k1**2 + 3.*k1*k2 + k2**2)
            ) / (60.*(k2 - k0)**2)
            + (
                10*k2**3 + 6.*k2**2*k1 + 3.*k2*k1**2
                + k1**3 + 5.*k3**2*(3.*k2 + k1)
                - 4.*k3*(6.*k2**2 + 3.*k2*k1 + k1**2)
            ) / (60.*(k3 - k1)**2)
            + (
                -2.*k1**3 - 2.*k2**3
                -3.*k1*k2*(k1 + k2)
                - 5.*k0*k3*(k1 + k2)
                + (k0 + k3)*(3.*k2**2 + 4.*k1*k2 + 3.*k1**2)
            ) / (30.*(k3 - k1)*(k2 - k0))
        )
        + (5.*k2 + k3)*(k3 - k2)**3 / (30.*(k3 - k1)**2)
    )


def _D0N2_Deg2_3_linear(k0, k1, k2, k3, k4):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    intt = np.zeros((k0.size,))
    intt[1:-1] += (
        (3.*k2 + 2.*k1 - 5.*k0)[1:-1]*(k2 - k1)[1:-1]**2
        / (60.*(k3 - k1)*(k2 - k0))[1:-1]
        + (3.*k2 + 2.*k1 - 5.*k0)[1:-1]*(k2 - k1)[1:-1]**2
        / (60.*(k3 - k1)*(k2 - k0))[1:-1]
    )
    intt[:-1] += (
        + (5.*k3 - 4.*k2 - k1)[:-1]*(k2 - k1)[:-1]**2
        / (20.*(k3 - k1)**2)[:-1]
        + (4.*k2 + k3 - 5.*k1)[:-1]*(k3 - k2)[:-1]**2
        / (20.*(k3 - k1)**2)[:-1]
    )
    return intt


def _D0N2_Deg2_3_toroidal(k0, k1, k2, k3, k4):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    return (
        (k2 - k1)**2
        * (2*k2**2 + 2*k1*k2 + k1**2 - k0*(3.*k2 + 2.*k1))
        / (60.*(k3 - k1)*(k2 - k0))
        + (k2 - k1)**2
        * (-10*k2**2 - 4*k1*k2 - k1**2 + 3*k3*(4*k2 + k1))
        / (60.*(k3 - k1)**2)
        + ((k3 - k2)**2)
        * (k3**2 + 4*k3*k2 + 10*k2**2 - 3*k1*(k3 + 4*k2))
        / (60*(k3 - k1)**2)
        + ((k3 - k2)**2)
        * (-k3**2 - 2*k3*k2 - 2*k2**2 + k4*(2*k3 + 3*k2))
        / (60*(k4 - k2)*(k3 - k1))
    )


def _D0N2_Deg2_2_linear(k0, k1, k2, k3):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    intt = np.zeros((k0.size,))
    intt[1:-1] = (
        (k2 - k1)[1:-1]**3
        / (30.*(k3-k1)*(k2-k0))[1:-1]
    )
    return intt


def _D0N2_Deg2_2_toroidal(k0, k1, k2, k3):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    return (
        (k2 + k1)*(k2 - k1)**3
        / (60.*(k3 - k1)*(k2 - k0))
    )
