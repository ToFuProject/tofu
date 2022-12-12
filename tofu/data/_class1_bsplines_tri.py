# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import scipy.interpolate as scpinterp
import scipy.spatial as scpspace
from matplotlib.tri import Triangulation as mplTri
import datastock as ds


# specific
from . import _class1_checks as _checks
from . import _class1_bsplines_operators_tri
from . import _class1_bsplines_rect as _mbr


# #############################################################################
# #############################################################################
#                       BivariateSplineRect - scipy subclass
# #############################################################################


class BivariateSplineTri(scpinterp.BivariateSpline):
    """ Subclass for tofu

    Defined from knots (unique) and deg
    coefs set to 1 by default

    """

    def __init__(
        self,
        knotsR=None,
        knotsZ=None,
        cents=None,
        trifind=None,
        deg=None,
    ):
        """ Class handling triangular bsplines """

        # ------------
        # check inputs

        knots = np.array([knotsR, knotsZ]).T
        cents, knots = _checks._mesh2DTri_conformity(
            knots=knots, cents=cents, key='class',
        )
        cents = _checks._mesh2DTri_clockwise(
            knots=knots, cents=cents, key='class',
        )

        # check trifinder
        if trifind is None:
            mpltri = mplTri(knotsR, knotsZ, cents)
            trifind = mpltri.get_trifinder()

        # deg
        deg = ds._generic_check._check_var(
            deg, 'deg',
            types=int,
            default=2,
            allowed=[0, 1, 2, 3],
        )

        if deg not in [0, 1]:
            msg = "Only deg=0 implemented for triangular meshes so far!"
            raise NotImplementedError(msg)

        self.knotsR = knotsR
        self.knotsZ = knotsZ
        self.nknots = knots.shape[0]
        self.cents = cents
        self.ncents = cents.shape[0]
        self.deg = deg
        self.trifind = trifind

        # ------------
        # get neigh cents per knot

        self.cents_per_knots = self._get_cents_per_knots()

        # ------------
        # get height per cent /  knot

        self.heights = self._get_heights_per_centsknots()

        # ------------
        # nbsplines

        if deg == 0:
            nbs = self.ncents
        elif deg == 1:
            nbs = self.nknots
        elif deg == 2:
            raise NotImplementedError()

        self.nbs = nbs
        self.shapebs = (nbs,)
        self.coefs = np.ones((nbs,), dtype=float)

    def _get_cents_per_knots(self):
        """ Return a (nknots, nmax) array of int indices

        Array contains -1 where there is no cent anymore
        """

        out = [
            np.any(self.cents == ii, axis=1).nonzero()[0]
            for ii in range(self.nknots)
        ]
        nmax = np.array([oo.size for oo in out])

        cents_per_knots = -np.ones((self.nknots, nmax.max()), dtype=int)
        for ii in range(self.nknots):
            cents_per_knots[ii, :nmax[ii]] = out[ii]
        return cents_per_knots

    def _get_heights_per_centsknots(self):
        """ Return the height of each knot in each cent

        Returnad as (ncents, 3) array, like cents
        """

        R = self.knotsR[self.cents]
        Z = self.knotsZ[self.cents]

        heights = np.full(self.cents.shape, np.nan)

        for iref, (i0, i1) in enumerate([(1, 2), (2, 0), (0, 1)]):
            base = np.sqrt(
                (R[:, i1] - R[:, i0])**2 + (Z[:, i1] - Z[:, i0])**2
            )
            heights[:, iref] = np.abs(
                (R[:, i0] - R[:, iref])*(Z[:, i1] - Z[:, iref])
                - (Z[:, i0] - Z[:, iref])*(R[:, i1] - R[:, iref])
            ) / base

        return heights

    def get_heights_per_centsknots_pts(self, x, y):
        """ Return the height of each knot in each cent

        Returnad as (ncents, 3) array, like cents

        OPTIMIZATION POSSIBLE FOR EV_DETAILS BY TAKING INDBS ASINPUT ARG !!!
        """

        if x.shape != y.shape:
            msg = "Arg x and y must hae the same shape!"
            raise Exception(msg)

        R = self.knotsR[self.cents]
        Z = self.knotsZ[self.cents]

        heights = np.full(tuple(np.r_[x.shape, 3]), np.nan)
        ind = self.trifind(x, y)

        for ii in np.unique(ind):
            if ii == -1:
                continue
            indi = ind == ii
            for iref, (i0, i1) in enumerate([(1, 2), (2, 0), (0, 1)]):
                v_base = np.array([
                    R[ii, i1] - R[ii, i0],
                    Z[ii, i1] - Z[ii, i0],
                ])
                v_perp = np.array([v_base[1], -v_base[0]])
                v_base = v_base / np.linalg.norm(v_base)
                v_perp = v_perp / np.linalg.norm(v_perp)

                v0 = np.array([
                    R[ii, i0] - R[ii, iref],
                    Z[ii, i0] - Z[ii, iref],
                ])
                v0_base = v0[0]*v_base[0] + v0[1]*v_base[1]
                v0_perp = v0[0]*v_perp[0] + v0[1]*v_perp[1]

                v_height = (v0 + (-v0_base*v_base + v0_perp*v_perp))/2.
                v_height_norm = np.linalg.norm(v_height)

                dR = x[indi] - R[ii, iref]
                dZ = y[indi] - Z[ii, iref]
                heights[indi, iref] = (
                    dR*v_height[0] + dZ*v_height[1]
                ) / v_height_norm**2

        indok = ~np.isnan(heights)
        assert np.all(heights[indok] >= 0. - 1e-14)
        assert np.all(heights[indok] <= 1. + 1e-14)
        return heights, ind

    # --------
    # bsplines

    def _get_knotscents_per_bs(
        self,
        ind=None,
        return_cents=None,
        return_knots=None,
        returnas=None,
    ):
        """ Return 2 arrays of int indices

        A (nbs, ncents_per_bs) array
        A (nbs, nknots_per_bs) array
        """

        # ------------
        # check inputs

        return_cents = ds._generic_check._check_var(
            return_cents, 'return_cents',
            types=bool,
            default=True,
        )

        return_knots = ds._generic_check._check_var(
            return_knots, 'return_knots',
            types=bool,
            default=True,
        )

        returnas = ds._generic_check._check_var(
            returnas, 'returnas',
            types=str,
            allowed=['ind', 'data'],
            default='ind',
        )

        if ind is None:
            ind = np.ones((self.nbs,), dtype=bool)
        ind_num = ind.nonzero()[0]
        nbs = ind.sum()

        # ------------
        # added for details

        if self.deg == 0:
            if return_cents:
                cents_per_bs = ind_num[:, None]
            if return_knots:
                knots_per_bs = self.cents[ind, :]

        elif self.deg == 1:
            if return_cents or return_knots:
                cents_per_bs = self.cents_per_knots[ind, :]
            if return_knots:
                nmax_cents = np.sum(cents_per_bs >= 0, axis=1)
                nmax = self.cents_per_knots.shape[1] + 3
                knots_per_bs = -np.ones((nbs, nmax), dtype=int)
                knots_per_bs[:, 0] = ind_num
                for ii, i0 in enumerate(ind_num):
                    nu = np.unique(
                        self.cents[cents_per_bs[ii, :nmax_cents[ii]], :]
                    )
                    knots_per_bs[ii, 1:nu.size] = [nn for nn in nu if nn != i0]

        elif self.deg == 2:
            raise NotImplementedError()

        # ------
        # return

        if returnas == 'data':

            # cents
            if return_cents:
                nmax = np.sum(cents_per_bs >= 0, axis=1)
                cents_per_bs_temp = np.full((2, nbs, nmax.max()), np.nan)
                for ii in range(nbs):
                    ind_temp = self.cents[cents_per_bs[ii, :nmax[ii]], :]
                    cents_per_bs_temp[0, ii, :nmax[ii]] = np.mean(
                        self.knotsR[ind_temp],
                        axis=1,
                    )
                    cents_per_bs_temp[1, ii, :nmax[ii]] = np.mean(
                        self.knotsZ[ind_temp],
                        axis=1,
                    )
                cents_per_bs = cents_per_bs_temp

            # knots
            if return_knots:
                nmax = np.sum(knots_per_bs >= 0, axis=1)
                knots_per_bs_temp = np.full((2, nbs, nmax.max()), np.nan)
                for ii in range(nbs):
                    ind_temp = knots_per_bs[ii, :nmax[ii]]
                    knots_per_bs_temp[0, ii, :nmax[ii]] = self.knotsR[ind_temp]
                    knots_per_bs_temp[1, ii, :nmax[ii]] = self.knotsZ[ind_temp]
                knots_per_bs = knots_per_bs_temp

        # return
        if return_cents and return_knots:
            return knots_per_bs, cents_per_bs
        elif return_cents:
            return cents_per_bs
        elif return_knots:
            return knots_per_bs

    def _get_bs_cents(
        self,
        ind=None,
    ):
        """ Return (2, nbs) array of cordinates of centers per bspline

        """

        # ------------
        # check inputs

        if ind is None:
            ind = np.ones((self.nbs,), dtype=bool)

        # ------------
        # added for details

        if self.deg == 0:
            bs_cents = np.array([
                np.mean(self.knotsR[self.cents[ind, :]], axis=1),
                np.mean(self.knotsZ[self.cents[ind, :]], axis=1),
            ])

        elif self.deg == 1:
            bs_cents = np.array([
                self.knotsR[ind],
                self.knotsZ[ind],
            ])

        elif self.deg == 2:
            raise NotImplementedError()

        return bs_cents

    # --------
    # evaluation checks

    def _check_coefs(self, coefs=None):
        """ None for ev_details, (nt, shapebs) for sum """
        if coefs is not None:
            assert coefs.ndim == len(self.shapebs) + 1
            assert coefs.shape[1:] == self.shapebs

    def _ev_generic(
        self,
        x,
        y,
        indbs=None,
    ):
        # -----------
        # check input

        if indbs is None:
            indbs = np.ones((self.nbs,), dtype=bool)
        else:
            indbs = np.atleast_1d(indbs).ravel()

        c0 = (
            isinstance(indbs, np.ndarray)
            and (
                ('bool' in indbs.dtype.name and indbs.size == self.nbs)
                or ('int' in indbs.dtype.name)
            )
        )
        if not c0:
            msg = (
                "Arg indbs must be  a (nbs,) bool or int array!"
                "\nProvided: {indbs}"
            )
            raise Exception(msg)

        if 'bool' in indbs.dtype.name:
            indbs = indbs.nonzero()[0]

        # indbs => indcent : triangles which are ok
        knots_per_bs, cents_per_bs = self._get_knotscents_per_bs(
            return_cents=True,
            return_knots=True,
            returnas='ind',
        )
        indcent = np.unique(cents_per_bs[indbs, :])
        indcent = indcent[indcent >= 0]

        # -----------
        # prepare

        nbs = indbs.size

        return nbs, knots_per_bs, cents_per_bs, indcent

    # --------
    # evaluation

    def ev_details(
        self,
        R=None,
        Z=None,
        # for compatibility (unused)
        indbs_tf=None,
        crop=None,
        cropbs=None,
        coefs=None,
        val_out=None,
    ):

        # -----------
        # generic

        # points
        x, y, crop = _mbr._check_RZ_crop(
            R=R,
            Z=Z,
            crop=crop,
            cropbs=cropbs,
        )

        # parameters
        nbs, knots_per_bs, cents_per_bs, indcent = self._ev_generic(
            x, y, indbs=indbs_tf,
        )

        # -----------
        # compute

        val = np.zeros(tuple(np.r_[x.shape, nbs]))
        heights, ind = self.get_heights_per_centsknots_pts(x, y)

        indu = np.unique(ind[ind >= 0])
        if self.deg == 0:
            if indbs_tf is None:
                for ii in np.intersect1d(indu, indcent):
                    indi = ind == ii
                    val[indi, ii] = 1.
            else:
                for ii in np.intersect1d(indu, indcent):
                    indi = ind == ii
                    ibs = indbs_tf == ii
                    val[indi, ibs] = 1.

        elif self.deg == 1:
            if indbs_tf is None:
                for ii in np.intersect1d(indu, indcent):
                    indi = ind == ii
                    # get bs
                    ibs = np.any(cents_per_bs == ii, axis=1).nonzero()[0]
                    sorter = np.argsort(self.cents[ii, :])
                    inum = sorter[np.searchsorted(
                        self.cents[ii, :],
                        knots_per_bs[ibs, 0],
                        sorter=sorter,
                    )]
                    for jj, jbs in enumerate(ibs):
                        val[indi, jbs] = 1. - heights[indi, inum[jj]]
            else:
                for ii in np.intersect1d(indu, indcent):
                    indi = ind == ii
                    # get bs
                    ibs = np.intersect1d(
                        indbs_tf,
                        np.any(cents_per_bs == ii, axis=1).nonzero()[0],
                    )
                    sorter = np.argsort(self.cents[ii, :])
                    inum = sorter[np.searchsorted(
                        self.cents[ii, :],
                        knots_per_bs[ibs, 0],
                        sorter=sorter,
                    )]
                    for jj, jbs in enumerate(ibs):
                        ij = indbs_tf == jbs
                        val[indi, ij] = 1. - heights[indi, inum[jj]]
        return val

    def ev_sum(
        self,
        R=None,
        Z=None,
        coefs=None,
        val_out=None,
        # for compatibility (unused)
        crop=None,
        cropbs=None,
        indbs_tf=None,
    ):

        # -----------
        # generic

        if val_out is None:
            val_out = np.nan

        # coefs
        self._check_coefs(coefs=coefs)

        # points
        x, y, crop = _mbr._check_RZ_crop(
            R=R,
            Z=Z,
            crop=crop,
            cropbs=cropbs,
        )
        nt = coefs.shape[0]

        # parameters
        nbs, knots_per_bs, cents_per_bs, indcent = self._ev_generic(
            x, y, indbs=None,
        )

        # -----------
        # prepare

        val = np.zeros(np.r_[nt, x.shape])
        heights, ind = self.get_heights_per_centsknots_pts(x, y)
        if not np.isscalar(coefs):
            newshape = np.r_[coefs.shape, 1]
            coefs = coefs.reshape(newshape)

        # -----------
        # compute

        indu = np.unique(ind[ind >= 0])
        if self.deg == 0:
            for ii in np.intersect1d(indu, indcent):
                indi = ind == ii
                val[:, indi] += coefs[:, ii, ...]

        elif self.deg == 1:
            for ii in np.intersect1d(indu, indcent):
                indi = ind == ii
                # get bs
                ibs = np.any(cents_per_bs == ii, axis=1).nonzero()[0]
                sorter = np.argsort(self.cents[ii, :])
                inum = sorter[np.searchsorted(
                    self.cents[ii, :],
                    knots_per_bs[ibs, 0],
                    sorter=sorter,
                )]
                for jj, jbs in enumerate(ibs):
                    val[:, indi] += (
                        1. - heights[indi, inum[jj]]
                    ) * coefs[:, jbs, ...]

        if val_out is not False:
            val[:, ind == -1] = val_out
        return val

    # TBC
    def get_overlap(self):
        raise NotImplementedError()
        return _get_overlap(
            deg=self.degrees[0],
            knotsx=self.knots_per_bs_x,
            knotsy=self.knots_per_bs_y,
            shapebs=self.shapebs,
        )

    # TBD / TBF
    def get_operator(
        self,
        operator=None,
        geometry=None,
        cropbs_flat=None,
        # specific to deg = 0
        cropbs=None,
    ):
        """ Get desired operator """
        raise NotImplementedError()
        return _class1_bsplines_operators_tri.get_mesh2dRect_operators(
            deg=self.degrees[0],
            operator=operator,
            geometry=geometry,
            knotsx_mult=self.tck[0],
            knotsy_mult=self.tck[1],
            knotsx_per_bs=self.knots_per_bs_x,
            knotsy_per_bs=self.knots_per_bs_y,
            overlap=self.get_overlap(),
            cropbs_flat=cropbs_flat,
            cropbs=cropbs,
        )


# #############################################################################
# #############################################################################
#                       Mesh2Dtri - bsplines - overlap
# #############################################################################


def _get_overlap(
    deg=None,
    knotsx=None,
    knotsy=None,
    shapebs=None,
):
    raise NotImplementedError()
    return inter


# #############################################################################
# #############################################################################
#                           Mesh2DRect - bsplines
# #############################################################################


def get_bs2d_func(
    knotsR=None,
    knotsZ=None,
    cents=None,
    trifind=None,
    deg=None,
):

    # -----------------
    # Instanciate class

    clas = BivariateSplineTri(
        knotsR=knotsR,
        knotsZ=knotsZ,
        cents=cents,
        trifind=trifind,
        deg=deg,
    )

    return clas.ev_details, clas.ev_sum, clas
