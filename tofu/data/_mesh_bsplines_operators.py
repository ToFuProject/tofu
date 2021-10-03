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
        if geometry == 'linear':
            dim = 'origin x m2'
        else:
            dim = 'origin x m3/rad'
    elif operator == 'D0N2':
        if geometry == 'linear':
            dim = 'origin2 x m2'
        else:
            dim = 'origin2 x m3/rad'
    elif operator == 'D1N2':
        if geometry == 'linear':
            dim = 'origin2'
        else:
            dim = 'origin2 x m/rad'
    elif operator == 'D2N2':
        if geometry == 'linear':
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
    cropbs_flat=None,
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

    if cropbs_flat is None:
        cropbs_flat = False
    if cropbs_flat is not False:
        c0 = (
            isinstance(cropbs_flat, np.ndarray)
            and cropbs_flat.shape == (nbs,)
            and cropbs_flat.dtype == np.bool_
        )
        if not c0:
            msg = (
                f"Arg cropbs_flat must be a bool array of shape {(nbs,)}\n"
                f"Provided: {cropbs_flat.shape}"
            )
            raise Exception(msg)
        nbscrop = cropbs_flat.sum()
        shape = (nbscrop, nbscrop)
        indbs = -np.ones((nbs,), dtype=int)
        indbs[cropbs_flat] = np.arange(0, nbscrop)
    else:
        shape = (nbs, nbs)
        indbs = np.arange(0, nbs)

    if 'N' in operator and deg >= 1:
        # get intersection indices array
        if cropbs_flat is False:
            nbtot = np.sum(overlap >= 0)
        else:
            ind = cropbs_flat[None, :] & cropbs_flat[overlap]
            nbtot = np.sum(ind)

        # prepare data and indices arrays
        if operator == 'D0N2':
            data = np.full((nbtot,), np.nan)
            row = np.zeros((nbtot,), dtype=int)
            column = np.zeros((nbtot,), dtype=int)
        elif operator == 'D1N2':
            datadR = np.full((nbtot,), np.nan)
            datadZ = np.full((nbtot,), np.nan)
            row = np.zeros((nbtot,), dtype=int)
            column = np.zeros((nbtot,), dtype=int)
        elif operator == 'D2N2':
            datad2R = np.full((nbtot,), np.nan)
            datad2Z = np.full((nbtot,), np.nan)
            datadRZ = np.full((nbtot,), np.nan)
            row = np.zeros((nbtot,), dtype=int)
            column = np.zeros((nbtot,), dtype=int)

    # ------------
    # D0

    if operator == 'D0':
        if deg == 0 and geometry == 'linear':

            opmat = (kR[1, :] - kR[0, :]) * (kZ[1, :] - kZ[0, :])

        elif deg == 0 and geometry == 'toroidal':

            opmat = 0.5 * (kR[1, :]**2 - kR[0, :]**2) * (kZ[1, :] - kZ[0, :])

        elif deg == 1 and geometry == 'linear':

            opmat = 0.25 * (kR[2, :] - kR[0, :]) * (kZ[2, :] - kZ[0, :])

        elif deg == 1 and geometry == 'toroidal':

            opmat = (
                0.5
                * (kR[2, :]**2 - kR[0, :]**2 + kR[1, :]*(kR[2, :]-kR[0, :]))
                * (kZ[2, :] - kZ[0, :])
            ) / 6.

        elif deg == 2:

            iZ1 = (kZ[1, :] - kZ[0, :])**2 / (3.*(kZ[2, :] - kZ[0, :]))
            iZ21 = (
                (
                    kZ[2, :]**2
                    - 2*kZ[1, :]**2
                    + kZ[1, :]*kZ[2, :]
                    + 3.*kZ[0, :]*(kZ[1, :] - kZ[2, :])
                )
                / (6.*(kZ[2, :]-kZ[0, :]))
            )
            iZ22 = (
                (
                    -2.*kZ[2, :]**2
                    + kZ[1, :]**2
                    + kZ[1, :]*kZ[2, :]
                    + 3.*kZ[3, :]*(kZ[2, :] - kZ[1, :])
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
                        + 3. * kR[0, :] * (kR[1, :] - kR[2, :])
                    )
                    / (6. * (kR[2, :] - kR[0, :]))
                )
                iR22 = (
                    (
                        -2. * kR[2, :]**2
                        + kR[1, :]**2
                        + kR[1, :] * kR[2, :]
                        + 3. * kR[3, :] * (kR[2, :] - kR[1, :])
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

        elif deg == 3:

            msg = "Integral D0 not implemented for deg=3 yet!"
            raise NotImplementedError(msg)

        # crop
        if cropbs_flat is not False:
            opmat = opmat[indbs_flat]

    # ------------
    # D0N2

    elif operator == 'D0N2' and deg == 0:

        iZ = kZ[1, :] - kZ[0, :]
        if geometry == 'linear':
            iR = kR[1, :] - kR[0, :]
        else:
            iR = 0.5 * (kR[1, :]**2 - kR[0, :]**2)

        if cropbs_flat is not False:
            iR = iR[cropbs_flat]
            iZ = iZ[cropbs_flat]

        opmat = scpsp.diags(
            [iR*iZ],
            [0],
            shape=None,
            format=sparse_fmt,
            dtype=float,
        )

    elif operator == 'D0N2':

        # pre-compute integrals
        if deg == 1:
            iR = _D0N2_Deg1(knotsx_mult, geometry=geometry)
            iZ = _D0N2_Deg1(knotsy_mult, geometry='linear')
        elif deg == 2:
            iR = _D0N2_Deg2(knotsx_mult, geometry=geometry)
            iZ = _D0N2_Deg2(knotsy_mult, geometry='linear')
        elif deg == 3:
            msg = "Integral D0N2 not implemented for deg=3!"
            raise NotImplementedError(msg)

        # set non-diagonal elements
        i0 = 0
        for ir in range(nx):
            for iz in range(ny):

                iflat = ir + iz*nx
                if cropbs_flat is not False and not cropbs_flat[iflat]:
                    continue

                # general case
                overlapi = overlap[:, iflat][overlap[:, iflat] > iflat]

                # diagonal element
                data[i0] = iR[0, ir] * iZ[0, iz]
                row[i0] = indbs[iflat]
                column[i0] = indbs[iflat]
                i0 += 1

                # non-diagonal elements (symmetric)
                for jflat in overlapi:

                    if cropbs_flat is not False and not cropbs_flat[jflat]:
                        continue

                    jr = jflat % nx
                    jz = jflat // nx

                    # store (i, j) and (j, i) (symmetric matrix)
                    if jr >= ir:
                        iiR = iR[jr - ir, ir]
                    else:
                        iiR = iR[abs(jr - ir), jr]
                    if jz >= iz:
                        iiZ = iZ[jz - iz, iz]
                    else:
                        iiZ = iZ[abs(jz - iz), jz]
                    data[i0:i0+2] = iiR * iiZ
                    row[i0:i0+2] = (indbs[iflat], indbs[jflat])
                    column[i0:i0+2] = (indbs[jflat], indbs[iflat])
                    i0 += 2

        assert i0 == nbtot
        opmat = scpsp.csr_matrix((data, (row, column)), shape=(nbs, nbs))

    # ------------
    # D1N2

    elif operator == 'D1N2':

        # pre-compute integrals
        if deg == 1:
            idR = _D1N2_Deg1(knotsx_mult, geometry=geometry)
            idZ = _D1N2_Deg1(knotsy_mult, geometry='linear')
            iR = _D0N2_Deg1(knotsx_mult, geometry=geometry)
            iZ = _D0N2_Deg1(knotsy_mult, geometry='linear')
        elif deg == 2:
            idR = _D1N2_Deg2(knotsx_mult, geometry=geometry)
            idZ = _D1N2_Deg2(knotsy_mult, geometry='linear')
            iR = _D0N2_Deg2(knotsx_mult, geometry=geometry)
            iZ = _D0N2_Deg2(knotsy_mult, geometry='linear')
        elif deg == 3:
            msg = "Integral D1N2 not implemented for deg=3!"
            raise NotImplementedError(msg)

        # set non-diagonal elements
        i0 = 0
        for ir in range(nx):
            for iz in range(ny):

                iflat = ir + iz*nx
                if cropbs_flat is not False and not cropbs_flat[iflat]:
                    continue

                # general case
                overlapi = overlap[:, iflat][overlap[:, iflat] > iflat]

                # diagonal element
                datadR[i0] = idR[0, ir] * iZ[0, iz]
                datadZ[i0] = iR[0, ir] * idZ[0, iz]
                row[i0] = indbs[iflat]
                column[i0] = indbs[iflat]
                i0 += 1

                # non-diagonal elements (symmetric)
                for jflat in overlapi:

                    if cropbs_flat is not False and not cropbs_flat[jflat]:
                        continue

                    jr = jflat % nx
                    jz = jflat // nx

                    # store (i, j) and (j, i) (symmetric matrix)
                    if jr >= ir:
                        iidR = idR[jr - ir, ir]
                        iiR = iR[jr - ir, ir]
                    else:
                        iidR = idR[abs(jr - ir), jr]
                        iiR = iR[abs(jr - ir), jr]
                    if jz >= iz:
                        iidZ = idZ[jz - iz, iz]
                        iiZ = iZ[jz - iz, iz]
                    else:
                        iidZ = idZ[abs(jz - iz), jz]
                        iiZ = iZ[abs(jz - iz), jz]
                    datadR[i0:i0+2] = iidR * iiZ
                    datadZ[i0:i0+2] = iiR * iidZ
                    row[i0:i0+2] = (indbs[iflat], indbs[jflat])
                    column[i0:i0+2] = (indbs[jflat], indbs[iflat])
                    i0 += 2

        assert i0 == nbtot
        opmat = (
            scpsp.csr_matrix((datadR, (row, column)), shape=(nbs, nbs)),
            scpsp.csr_matrix((datadZ, (row, column)), shape=(nbs, nbs)),
        )

    # ------------
    # D2N2

    elif operator == 'D2N2':

        # pre-compute integrals
        if deg == 2:
            id2R = _D2N2_Deg2(knotsx_mult, geometry=geometry)
            id2Z = _D2N2_Deg2(knotsy_mult, geometry='linear')
            idR = _D1N2_Deg2(knotsx_mult, geometry=geometry)
            idZ = _D1N2_Deg2(knotsy_mult, geometry='linear')
            iR = _D0N2_Deg2(knotsx_mult, geometry=geometry)
            iZ = _D0N2_Deg2(knotsy_mult, geometry='linear')
        elif deg == 3:
            msg = "Integral D2N2 not implemented for deg=3!"
            raise NotImplementedError(msg)

        # set non-diagonal elements
        i0 = 0
        for ir in range(nx):
            for iz in range(ny):

                iflat = ir + iz*nx
                if cropbs_flat is not False and not cropbs_flat[iflat]:
                    continue

                # general case
                overlapi = overlap[:, iflat][overlap[:, iflat] > iflat]

                # diagonal element
                datad2R[i0] = id2R[0, ir] * iZ[0, iz]
                datad2Z[i0] = iR[0, ir] * id2Z[0, iz]
                datadRZ[i0] = idR[0, ir] * idZ[0, iz]
                row[i0] = indbs[iflat]
                column[i0] = indbs[iflat]
                i0 += 1

                # non-diagonal elements (symmetric)
                for jflat in overlapi:

                    if cropbs_flat is not False and not cropbs_flat[jflat]:
                        continue

                    jr = jflat % nx
                    jz = jflat // nx

                    # store (i, j) and (j, i) (symmetric matrix)
                    if jr >= ir:
                        iid2R = id2R[jr - ir, ir]
                        iidR = idR[jr - ir, ir]
                        iiR = iR[jr - ir, ir]
                    else:
                        iid2R = id2R[abs(jr - ir), jr]
                        iidR = idR[abs(jr - ir), jr]
                        iiR = iR[abs(jr - ir), jr]
                    if jz >= iz:
                        iid2Z = id2Z[jz - iz, iz]
                        iidZ = idZ[jz - iz, iz]
                        iiZ = iZ[jz - iz, iz]
                    else:
                        iid2Z = id2Z[abs(jz - iz), jz]
                        iidZ = idZ[abs(jz - iz), jz]
                        iiZ = iZ[abs(jz - iz), jz]
                    datad2R[i0:i0+2] = iid2R * iiZ
                    datad2Z[i0:i0+2] = iiR * iid2Z
                    datadRZ[i0:i0+2] = iidR * iidZ
                    row[i0:i0+2] = (indbs[iflat], indbs[jflat])
                    column[i0:i0+2] = (indbs[jflat], indbs[iflat])
                    i0 += 2

        assert i0 == nbtot
        opmat = (
            scpsp.csr_matrix((datad2R, (row, column)), shape=shape),
            scpsp.csr_matrix((datad2Z, (row, column)), shape=shape),
            scpsp.csr_matrix((datadRZ, (row, column)), shape=shape),
        )

    # ------------
    # D3N2

    elif operator == 'D3N2' and deg == 3:

        raise NotImplementedError("Integral D3N2 not implemented for deg=3!")

    return opmat, operator, geometry, dim


# #############################################################################
# #############################################################################
#               Operator sub-routines: D0N2
# #############################################################################


def _D0N2_Deg1_full_linear(k0, k2):
    """ from 1d knots, return int_0^2 x b**2(x) dx """
    return (k2 - k0) / 3.


def _D0N2_Deg1_full_toroidal(k0, k1, k2):
    """ from 1d knots, return int_0^2 x b**2(x) dx """
    intt = np.zeros((k0.size,))
    intt[1:] += (
        (3. * k1**3 - 5.*k0*k1**2 + k1*k0**2 + k0**3)[1:]
        / (12. * (k1 - k0))[1:]
    )
    intt[:-1] = (
        + (3.*k1**3 - 5.*k2*k1**2 + k1*k2**2 + k2**3)[:-1]
        / (12. * (k2 - k1))[:-1]
    )
    return intt


def _D0N2_Deg1_2_linear(k1, k2):
    """ from 1d knots, return int_0^2 x b**2(x) dx """
    return (k2 - k1) / 6.


def _D0N2_Deg1_2_toroidal(k1, k2):
    """ from 1d knots, return int_0^2 x b**2(x) dx """
    return (k2**2 - k1**2) / 12.


def _D0N2_Deg1(knots, geometry=None):

    if geometry == 'linear':
        integ = np.array([
            _D0N2_Deg1_full_linear(
                knots[:-2],
                knots[2:],
            ),
            _D0N2_Deg1_2_linear(
                knots[1:-1],
                knots[2:]
            ),
        ])
    else:
        integ = np.array([
            _D0N2_Deg1_full_toroidal(
                knots[:-2],
                knots[1:-1],
                knots[2:],
            ),
            _D0N2_Deg1_2_toroidal(
                knots[1:-1],
                knots[2:]
            ),
        ])
    return integ


def _D0N2_Deg2(knots, geometry=None):

    if geometry == 'linear':
        ffull = _D0N2_Deg2_full_linear
        f3 = _D0N2_Deg2_3_linear
        f2 = _D0N2_Deg2_2_linear
    else:
        ffull = _D0N2_Deg2_full_toroidal
        f3 = _D0N2_Deg2_3_toroidal
        f2 = _D0N2_Deg2_2_toroidal

    integ = np.array([
        ffull(
            knots[:-3],
            knots[1:-2],
            knots[2:-1],
            knots[3:],
        ),
        f3(
            knots[:-3],
            knots[1:-2],
            knots[2:-1],
            knots[3:],
            np.r_[knots[4:], np.nan],
        ),
        f2(
            knots[1:-2],
            knots[2:-1],
            knots[3:],
            np.r_[knots[4:], np.nan],
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
        (k2 - k1)[1:-1]
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
        + (k3 - k2)[:-1]**3 / (5.*(k3 - k1)[:-1]**2)
    )
    return intt


def _D0N2_Deg2_full_toroidal(k0, k1, k2, k3):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    intt = np.zeros((k0.size,))
    intt[1:] += (
        (5.*k1 + k0)[1:]*(k1 - k0)[1:]**3 / (30.*(k2 - k0)[1:]**2)
        + (k2 - k1)[1:]
        * (
            10*k1**3 + 6.*k1**2*k2 + 3.*k1*k2**2
            + k2**3 + 5.*k0**2*(3.*k1 + k2)
            - 4.*k0*(6.*k1**2 + 3.*k1*k2 + k2**2)
        )[1:] / (60.*(k2 - k0)**2)[1:]
    )
    intt[:-1] += (
        (5.*k2 + k3)[:-1]*(k3 - k2)[:-1]**3 / (30.*(k3 - k1)[:-1]**2)
        + (k2 - k1)[1:]
        * (
            10*k2**3 + 6.*k2**2*k1 + 3.*k2*k1**2
            + k1**3 + 5.*k3**2*(3.*k2 + k1)
            - 4.*k3*(6.*k2**2 + 3.*k2*k1 + k1**2)
        )[:-1] / (60.*(k3 - k1)**2)[:-1]
    )
    intt[1:-1] += (
        (k2 - k1)[1:-1]
        * (
            - 2.*k1**3 - 2.*k2**3
            - 3.*k1*k2*(k1 + k2)
            - 5.*k0*k3*(k1 + k2)
            + (k0 + k3)*(3.*k2**2 + 4.*k1*k2 + 3.*k1**2)
        )[1:-1] / (30.*(k3 - k1)*(k2 - k0))[1:-1]
    )

    return intt


def _D0N2_Deg2_3_linear(k0, k1, k2, k3, k4):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    intt = np.zeros((k0.size,))
    intt[1:-1] += (
        (3.*k2 + 2.*k1 - 5.*k0)[1:-1]*(k2 - k1)[1:-1]**2
        / (60.*(k3 - k1)*(k2 - k0))[1:-1]
    )
    intt[:-2] += (
        + (5.*k4 - 2.*k3 - 3.*k2)[:-2]*(k3 - k2)[:-2]**2
        / (60.*(k4 - k2)*(k3 - k1))[:-2]
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
    intt = np.zeros((k0.size,))
    intt[:-1] = (
        (k2 - k1)[:-1]**2
        * (-10*k2**2 - 4*k1*k2 - k1**2 + 3*k3*(4*k2 + k1))[:-1]
        / (60.*(k3 - k1)**2)[:-1]
        + (k3 - k2)[:-1]**2
        * (k3**2 + 4*k3*k2 + 10*k2**2 - 3*k1*(k3 + 4*k2))[:-1]
        / (60*(k3 - k1)**2)[:-1]
    )
    intt[1:-1] = (
        (k2 - k1)[1:-1]**2
        * (2*k2**2 + 2*k1*k2 + k1**2 - k0*(3.*k2 + 2.*k1))[1:-1]
        / (60.*(k3 - k1)*(k2 - k0))[1:-1]
    )
    intt[:-2] = (
        + (k3 - k2)[:-2]**2
        * (-k3**2 - 2*k3*k2 - 2*k2**2 + k4*(2*k3 + 3*k2))[:-2]
        / (60*(k4 - k2)*(k3 - k1))[:-2]
    )
    return intt


def _D0N2_Deg2_2_linear(k1, k2, k3, k4):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    intt = np.zeros((k1.size,))
    intt[:-2] = (
        (k3 - k2)[:-2]**3
        / (30.*(k4 - k2)*(k3 - k1))[:-2]
    )
    return intt


def _D0N2_Deg2_2_toroidal(k1, k2, k3, k4):
    """ from 1d knots, return int_0^3 b**2(x) dx """
    intt = np.zeros((k1.size,))
    intt[:-2] = (
        (k3 + k2)[:-2]*(k3 - k2)[:-2]**3
        / (60.*(k4 - k2)[:-2]*(k3 - k1)[:-2])
    )
    return intt


# #############################################################################
# #############################################################################
#               Operator sub-routines: D1N2
# #############################################################################


def _D1N2_Deg1(knots, geometry=None):

    if geometry == 'linear':
        ffull = _D1N2_Deg1_full_linear
        f2 = _D1N2_Deg1_2_linear
    else:
        ffull = _D1N2_Deg1_full_toroidal
        f2 = _D1N2_Deg1_2_toroidal

    integ = np.array([
        ffull(
            knots[:-2],
            knots[1:-1],
            knots[2:],
        ),
        f2(
            knots[1:-1],
            knots[2:]
        ),
    ])
    return integ


def _D1N2_Deg1_full_linear(k0, k1, k2):
    intt = np.zeros((k0.size,))
    intt[1:] += 1. / (k1 - k0)[1:]
    intt[:-1] += 1. / (k2 - k1)[:-1]
    return intt


def _D1N2_Deg1_full_toroidal(k0, k1, k2):
    intt = np.zeros((k0.size,))
    intt[1:] += (k1 + k0)[1:] / (2.*(k1 - k0))[1:]
    intt[:-1] += (k2 + k1)[:-1] / (2.*(k2 - k1))[:-1]
    return intt


def _D1N2_Deg1_2_linear(k1, k2):
    intt = np.zeros((k1.size,))
    intt[:-1] = -1. / (k2 - k1)[:-1]
    return intt


def _D1N2_Deg1_2_toroidal(k1, k2):
    intt = np.zeros((k1.size,))
    intt[:-1] = - (k2 + k1)[:-1] / (2.*(k2 - k1))[:-1]
    return intt


def _D1N2_Deg2(knots, geometry=None):

    if geometry == 'linear':
        ffull = _D1N2_Deg2_full_linear
        f3 = _D1N2_Deg2_3_linear
        f2 = _D1N2_Deg2_2_linear
    else:
        ffull = _D1N2_Deg2_full_toroidal
        f3 = _D1N2_Deg2_3_toroidal
        f2 = _D1N2_Deg2_2_toroidal

    integ = np.array([
        ffull(
            knots[:-3],
            knots[1:-2],
            knots[2:-1],
            knots[3:],
        ),
        f3(
            knots[:-3],
            knots[1:-2],
            knots[2:-1],
            knots[3:],
            np.r_[knots[4:], np.nan],
        ),
        f2(
            knots[1:-2],
            knots[2:-1],
            knots[3:],
            np.r_[knots[4:], np.nan],
        ),
    ])
    return integ


def _D1N2_Deg2_full_linear(k0, k1, k2, k3):
    intt = np.zeros((k0.size,))
    intt[1:] += 4.*(k1 - k0)[1:] / (3.*(k2 - k0)[1:]**2)
    intt[:-1] += 4.*(k3 - k2)[:-1] / (3.*(k3 - k1)[:-1]**2)
    intt[1:-1] += (
        4.*(k2 - k1)[1:-1]
        * (
            k2**2 + k2*k1 + k1**2 + k3**2 + k0*k3 + k0**2
            - k3*(k2 + 2.*k1) - k0*(2.*k2 + k1)
        )[1:-1]
        / (3.*(k3 - k1)[1:-1]**2*(k2 - k0)[1:-1]**2)
    )
    return intt


def _D1N2_Deg2_full_toroidal(k0, k1, k2, k3):
    intt = np.zeros((k0.size,))
    intt[1:] += (3.*k1 + k0)[1:]*(k1 - k0)[1:] / (3.*(k2 - k0)[1:]**2)
    intt[:-1] += (3.*k2 + k3)[:-1]*(k3 - k2)[:-1] / (3.*(k3 - k1)[:-1]**2)
    intt[1:-1] += (
        (k2 - k1)[1:-1]
        * (
            3.*(k2 + k1)*(k2**2 + k1**2)
            + k3**2*(k2 + 3.*k1)
            + k0**2*(3.*k2 + k1)
            - 2.*k3*(k2**2 + 2.*k2*k1 + 3.*k1**2)
            - 2.*k0*(3.*k2**2 + 2.*k2*k1 + k1**2)
            + 2.*k3*k0*(k2 + k1)
        )[1:-1]
        / (3.*(k3 - k1)[1:-1]**2*(k2 - k0)[1:-1]**2)
    )
    return intt


def _D1N2_Deg2_3_linear(k0, k1, k2, k3, k4):
    intt = np.zeros((k0.size,))
    intt[1:-1] += (
        2.*(k2 - k1)[1:-1]
        * (k3 - 2.*k2 - k1 + 2.*k0)[1:-1]
        / (3.*(k3 - k1)**2*(k2 - k0))[1:-1]
    )
    intt[:-2] += (
        2.*(k3 - k2)[:-2]
        * (-2.*k4 + k3 + 2.*k2 - k1)[:-2]
        / (3.*(k4 - k2)*(k3 - k1)**2)[:-2]
    )
    return intt


def _D1N2_Deg2_3_toroidal(k0, k1, k2, k3, k4):
    intt = np.zeros((k0.size,))
    intt[1:-1] += (
        (k2 - k1)[1:-1]
        * (
            - (3.*k2**2 + 2.*k2*k1 + k1**2)
            + k3*(k2 + k1)
            + k0*(3.*k2 + k1)
        )[1:-1]
        / (3.*(k3 - k1)**2*(k2 - k0))[1:-1]
    )
    intt[:-2] += (
        (k3 - k2)[:-2]
        * (
            k3**2 + 2.*k2*k3 + 3.*k2**2
            - k4*(k3 + 3.*k2)
            - k1*(k3 + k2)
        )[:-2]
        / (3.*(k4 - k2)*(k3 - k1)**2)[:-2]
    )
    return intt


def _D1N2_Deg2_2_linear(k1, k2, k3, k4):
    intt = np.zeros((k1.size,))
    intt[:-2] += -(
        2.*(k3 - k2)[:-2]
        / (3.*(k4 - k2)*(k3 - k1))[:-2]
    )
    return intt


def _D1N2_Deg2_2_toroidal(k1, k2, k3, k4):
    intt = np.zeros((k1.size,))
    intt[:-2] += -(
        (k3 + k2)[:-2]*(k3 - k2)[:-2]
        / (3.*(k4 - k2)*(k3 - k1))[:-2]
    )
    return intt


# #############################################################################
# #############################################################################
#               Operator sub-routines: D2N2
# #############################################################################


def _D2N2_Deg2(knots, geometry=None):

    if geometry == 'linear':
        ffull = _D2N2_Deg2_full_linear
        f3 = _D2N2_Deg2_3_linear
        f2 = _D2N2_Deg2_2_linear
    else:
        ffull = _D2N2_Deg2_full_toroidal
        f3 = _D2N2_Deg2_3_toroidal
        f2 = _D2N2_Deg2_2_toroidal

    integ = np.array([
        ffull(
            knots[:-3],
            knots[1:-2],
            knots[2:-1],
            knots[3:],
        ),
        f3(
            knots[:-3],
            knots[1:-2],
            knots[2:-1],
            knots[3:],
            np.r_[knots[4:], np.nan],
        ),
        f2(
            knots[1:-2],
            knots[2:-1],
            knots[3:],
            np.r_[knots[4:], np.nan],
        ),
    ])
    return integ


def _D2N2_Deg2_full_linear(k0, k1, k2, k3):
    intt = np.zeros((k0.size,))
    intt[2:] += 4. / ((k2 - k0)**2*(k1 - k0))[2:]
    intt[:-2] += 4. / ((k3 - k2)*(k3 - k1)**2)[:-2]
    intt[1:-1] += (
        4.*(k3 + k2 - k1 - k0)[1:-1]**2
        / ((k3 - k1)**2*(k2 - k1)*(k2 - k0)**2)[1:-1]
    )
    return intt


def _D2N2_Deg2_full_toroidal(k0, k1, k2, k3):
    intt = np.zeros((k0.size,))
    intt[2:] += 2.*(k0 + k0)[2:] / ((k2 - k0)**2*(k1 - k0))[2:]
    intt[:-2] += 2.*(k3 + k2)[2:] / ((k3 - k2)*(k3 - k1)**2)[:-2]
    intt[1:-1] += (
        (2.*(k3 + k2 - k1 - k0)**2*(k2 + k1))[1:-1]
        / ((k3 - k1)**2*(k2 - k1)*(k2 - k0)**2)[1:-1]
    )
    return intt


def _D2N2_Deg2_3_linear(k0, k1, k2, k3, k4):
    intt = np.zeros((k0.size,))
    intt[1:-1] += (
        - 4.*(k3 + k2 - k1 - k0)[1:-1]
        / ((k3 - k1)**2*(k2 - k1)*(k2 - k0))[1:-1]
    )
    intt[:-2] += (
        - 4.*(k4 + k3 - k2 - k1)[:-2]
        / ((k4 - k2)*(k3 - k2)*(k3 - k1)**2)[:-2]
    )
    return intt


def _D2N2_Deg2_3_toroidal(k0, k1, k2, k3, k4):
    intt = np.zeros((k0.size,))
    intt[1:-1] += (
        - 2.*((k3 + k2 - k1 - k0)*(k2 + k1))[1:-1]
        / ((k3 - k1)**2*(k2 - k1)*(k2 - k0))[1:-1]
    )
    intt[:-2] += (
        - 2.*((k4 + k3 - k2 - k1)*(k3 + k2))[:-2]
        / ((k4 - k2)*(k3 - k2)*(k3 - k1)**2)[:-2]
    )
    return intt


def _D2N2_Deg2_2_linear(k1, k2, k3, k4):
    intt = np.zeros((k1.size,))
    intt[:-2] += 4. / ((k4 - k2)*(k3 - k2)*(k3 - k1))[:-2]
    return intt


def _D2N2_Deg2_2_toroidal(k1, k2, k3, k4):
    intt = np.zeros((k1.size,))
    intt[:-2] += 2.*(k3 + k2)[:-2] / ((k4 - k2)*(k3 - k2)*(k3 - k1))[:-2]
    return intt
