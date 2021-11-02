# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import scipy.interpolate as scpinterp
import scipy.spatial as scpspace
from matplotlib.tri import Triangulation as mplTri


# specific
from . import _generic_check
from . import _mesh_checks
from . import _mesh_bsplines_operators_tri


# #############################################################################
# #############################################################################
#                       BivariateSplineRect - scipy subclass
# #############################################################################


class BivariateSplineTri(scpinterp.BivariateSpline):
    """ Subclass for tofu

    Defined from knots (unique) and deg
    coefs set to 1 by default

    Used self.set_coefs() to update
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
        cents, knots = _mesh_checks._mesh2DTri_conformity(
            knots=knots, cents=cents, key='class',
        )
        cents = _mesh_checks._mesh2DTri_clockwise(
            knots=knots, cents=cents, key='class',
        )

        # check trifinder
        if trifind is None:
            mpltri = mplTri(knotsR, knotsZ, cents)
            trifind = mpltri.get_trifinder()

        # deg
        deg = _generic_check._check_var(
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
            v0norm = np.sqrt(
                (R[:, i0] - R[:, iref])**2 + (Z[:, i0] - Z[:, iref])**2
            )
            heights[:, iref] = (
                (R[:, i0] - R[:, iref])*(R[:, i1] - R[:, iref])
                 + (Z[:, i0] - Z[:, iref])*(Z[:, i1] - Z[:, iref])
            ) / v0norm

        return heights

    # --------
    # bsplines

    def _get_centsknots_per_bs(
        self,
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

        return_cents = _generic_check._check_var(
            return_cents, 'return_cents',
            types=bool,
            default=True,
        )

        return_knots = _generic_check._check_var(
            return_knots, 'return_knots',
            types=bool,
            default=True,
        )

        returnas = _generic_check._check_var(
            returnas, 'returnas',
            types=str,
            allowed=['ind', 'data'],
            default='ind',
        )

        # ------------
        # added for details

        if self.deg == 0:
            if return_cents:
                cents_per_bs = np.arange(0, self.nbs)[:, None]
            if return_knots:
                knots_per_bs = self.cents
        elif self.deg == 1:
            if return_cents:
                cents_per_bs = self.cents_per_knots
            if return_knots:
                knots0 = np.arange(0, self.nknots)
                knots_per_bs = -np.ones((self.nbs, ), dtype=int)
                knots_per_bs[:, 0] = knots0
                for ii in range(self.nbs):
                    import pdb; pdb.set_trace()     # DB  / TBF
                    nn = np.unique(self.cents[cents_per_bs, :])
                    knots_per_bs[ii, 1:] = None

        elif self.deg == 2:
            raise NotImplementedError()

        # ------
        # return

        if returnas == 'data':
            if return_cents:
                cents_per_bs = np.array([
                    np.mean(self.knotsR[self.cents], axis=1)[cents_per_bs],
                    np.mean(self.knotsZ[self.cents], axis=1)[cents_per_bs],
                ])
            if return_knots:
                knots_per_bs = np.array([
                    self.knotsR[knots_per_bs],
                    self.knotsZ[knots_per_bs],
                ])

        if return_cents and return_knots:
            return cents_per_bs, knots_per_bs
        elif return_cents:
            return cents_per_bs
        elif return_knots:
            return knots_per_bs

    def set_coefs(
        self,
        coefs=None,
    ):

        # ------------
        # check inputs

        # trivial case
        if coefs is None:
            return

        # ------------
        # set coefs

        if np.isscalar(coefs):
            self.coefs[...] = coefs
        else:
            if not isinstance(coefs, np.ndarray):
                coefs = np.asarray(coefs)
            if coefs.shape != (self.nbs,):
                msg = (
                    "Arg coefs has wrong shape!\n"
                    f"\t- expected: {(self.nbs,)}\n"
                    f"\t- provided: {coefs.shape}\n"
                )
                raise Exception(msg)
            self.coefs = coefs

    # --------
    # evaluation

    def __call__(
        self,
        x,
        y,
        coefs=None,
        **kwdargs,
    ):

        # prepare
        self.set_coefs(coefs=coefs)

        # compute
        val = _eval_bsplinestri(x, y, deg=self.deg)
        return val

    def ev_details(
        self,
        x,
        y,
        indbs_tuple_flat=None,
        reshape=None,
    ):
        raise NotImplementedError()

        # -----------
        # check input

        c0 = (
            isinstance(indbs_tuple_flat, tuple)
            and len(indbs_tuple_flat) == 2
            and all([isinstance(ind, np.ndarray) for ind in indbs_tuple_flat])
            and indbs_tuple_flat[0].shape == indbs_tuple_flat[1].shape
        )
        if not c0:
            msg = (
                "Arg indbs_tuple_flat must be a tuple of indices!"
            )
            raise Exception(msg)

        if reshape is None:
            reshape = True
        if not isinstance(reshape, bool):
            msg = f"Arg reshape must be a bool!\nProvided {reshape}"
            raise Exception(msg)

        # -----------
        # prepare

        deg = self.degrees[0]
        nbs = indbs_tuple_flat[0].size
        shape = x.shape
        x = np.ascontiguousarray(x.ravel(), dtype=np.floating)
        y = np.ascontiguousarray(y.ravel(), dtype=np.floating)
        coef = np.zeros((deg + 4, 1), dtype=float)
        coef[deg] = 1.
        outy = np.full((x.size, 1), np.nan)

        # -----------
        # compute

        val = np.zeros(tuple(np.r_[x.size, nbs]))
        indtot = np.arange(0, nbs)

        iz_u = np.unique(indbs_tuple_flat[1])

        for iz in iz_u:

            scpinterp._bspl.evaluate_spline(
                self.knots_per_bs_y_pad[:, iz],
                coef,
                self.degrees[1],
                y,
                0,
                False,
                outy,
            )

            indoky = ~np.isnan(outy)
            if not np.any(indoky):
                continue
            indokx = np.copy(indoky)

            indr = indbs_tuple_flat[1] == iz
            ir = indbs_tuple_flat[0][indr]
            for ii, iir in enumerate(ir):

                if ii > 0:
                    indokx[...] = indoky

                outx = np.full((indoky.sum(), 1), np.nan)

                scpinterp._bspl.evaluate_spline(
                    self.knots_per_bs_x_pad[:, iir],
                    coef,
                    self.degrees[0],
                    x[indoky[:, 0]],
                    0,
                    False,
                    outx,
                )

                ixok = ~np.isnan(outx)
                if not np.any(ixok):
                    continue

                indokx[indoky] = ixok[:, 0]
                val[indokx[:, 0], indtot[indr][ii]] = (outx[ixok]*outy[indokx])

        if reshape:
            val = np.reshape(val, shape)

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

    # TBC
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
        return _mesh_bsplines_operators_tri.get_mesh2dRect_operators(
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
#                       Mesh2Dtri - bsplines - eval
# #############################################################################


def _eval_bsplinestri(x, y, deg=None):

    # ------------
    # check inputs

    c0 = (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and x.shape == y.shape
    )
    if not c0:
        msg = (
            "x and y must be np.ndarrays of the same shape!\n"
        )
        raise Exception(msg)

    # ------------
    # prepare output

    val = np.full(x.shape, np.nan)

    ind = trifind(x, y)
    import pdb; pdb.set_trace()     # DB
    if deg == 0:
        pass

    elif deg == 1:
        pass

    elif deg == 2:
        raise NotImplementedError()

    # ------
    # return

    return val


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
    # nb of overlapping, inc. itself in 1d
    nbsR, nbsZ = shapebs
    indR0 = np.tile(np.arange(0, nbsR), nbsZ)
    indZ0 = np.repeat(np.arange(0, nbsZ), nbsR)

    # complete
    ntot = 2*deg + 1

    addR = np.tile(np.arange(-deg, deg+1), ntot)
    addZ = np.repeat(np.arange(-deg, deg+1), ntot)

    interR = indR0[None, :] + addR[:, None]
    interZ = indZ0[None, :] + addZ[:, None]

    # purge
    inter = interR + interZ*nbsR
    indneg = (
        (interR < 0) | (interR >= nbsR) | (interZ < 0) | (interZ >= nbsZ)
    )
    inter[indneg] = -1

    return inter


# #############################################################################
# #############################################################################
#                           Mesh2DRect - bsplines
# #############################################################################


def _get_bs2d_func_check(
    coefs=None,
    ii=None,
    jj=None,
    indbs=None,
    R=None,
    Z=None,
    shapebs=None,
    crop=None,
    cropbs=None,
):

    # ii, jj
    if ii is not None:
        c0 = (
            isinstance(ii, (int, np.integer))
            and ii >= 0
            and ii < shapebs[0]
        )
        if not c0:
            msg = (
                "Arg ii must be an index in the range [0, {shapebs[0]}[\n"
                f"Provided: {ii}"
            )
            raise Exception(msg)
    if jj is not None:
        c0 = (
            isinstance(jj, (int, np.integer))
            and jj >= 0
            and jj < shapebs[1]
        )
        if not c0:
            msg = (
                "Arg jj must be an index in the range [0, {shapebs[1]}[\n"
                f"Provided: {jj}"
            )
            raise Exception(msg)

    # R, Z
    if not isinstance(R, np.ndarray):
        R = np.atleast_1d(R)
    if not isinstance(Z, np.ndarray):
        Z = np.atleast_1d(Z)
    assert R.shape == Z.shape

    # coefs
    if coefs is None:
        coefs = 1.
    if np.isscalar(coefs):
        pass
    else:
        coefs = np.atleast_1d(coefs)
        if coefs.ndim < len(shapebs) or coefs.ndim > len(shapebs) + 1:
            msg = (
                "coefs has too small / big shape!\n"
                f"\t- coefs.shape: {coefs.shape}\n"
                f"\t- shapebs:     {shapebs}"
            )
            raise Exception(msg)
        if coefs.ndim == len(shapebs):
            coefs = coefs.reshape(tuple(np.r_[1, coefs.shape]))
        if coefs.shape[1:] != shapebs:
            msg = (
                "coefs has wrong shape!\n"
                f"\t- coefs.shape: {coefs.shape}\n"
                f"\t- shapebs:     {shapebs}"
            )
            raise Exception(msg)

    # crop
    if crop is None:
        crop = True
    if not isinstance(crop, bool):
        msg = (
            "Arg crop must be a bool!\n"
            f"Provided: {crop}"
        )
        raise Exception(msg)
    crop = crop and cropbs is not None and cropbs is not False

    return coefs, ii, jj, R, Z, crop


def _get_bs2d_func_knots(knots, deg=None, returnas=None, return_unique=None):

    # ----------
    # check input

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=str,
        default='data',
        allowed=['ind', 'data'],
    )
    return_unique = _generic_check._check_var(
        return_unique, 'return_unique',
        types=bool,
        default=False,
    )

    # ----------
    # compute

    nkpbs = 2 + deg
    size = knots.size
    nbs = size - 1 + deg

    if return_unique:
        if deg == 0:
            knots_per_bs = np.arange(0, size)
        elif deg == 1:
            knots_per_bs = np.r_[0, np.arange(0, size), size-1]
        elif deg == 2:
            knots_per_bs = np.r_[0, 0, np.arange(0, size), size-1, size-1]
        elif deg == 3:
            knots_per_bs = np.r_[
                0, 0, 0, np.arange(0, size), size-1, size-1, size-1,
            ]

    else:
        knots_per_bs = np.zeros((nkpbs, nbs), dtype=int)

        if deg == 0:
            knots_per_bs[:, :] = np.array([
                np.arange(0, size-1),
                np.arange(1, size),
            ])

        elif deg == 1:
            knots_per_bs[:, 1:-1] = np.array([
                np.arange(0, size-2),
                np.arange(1, size-1),
                np.arange(2, size),
            ])
            knots_per_bs[:, 0] = [0, 0, 1]
            knots_per_bs[:, -1] = [-2, -1, -1]

        elif deg == 2:
            knots_per_bs[:, 2:-2] = np.array([
                np.arange(0, size-3),
                np.arange(1, size-2),
                np.arange(2, size-1),
                np.arange(3, size),
            ])
            knots_per_bs[:, 0] = [0, 0, 0, 1]
            knots_per_bs[:, 1] = [0, 0, 1, 2]
            knots_per_bs[:, -2] = [-3, -2, -1, -1]
            knots_per_bs[:, -1] = [-2, -1, -1, -1]

        elif deg == 3:
            knots_per_bs[:, 3:-3] = np.array([
                np.arange(0, size-4),
                np.arange(1, size-3),
                np.arange(2, size-2),
                np.arange(3, size-1),
                np.arange(4, size),
            ])
            knots_per_bs[:, 0] = [0, 0, 0, 0, 1]
            knots_per_bs[:, 1] = [0, 0, 0, 1, 2]
            knots_per_bs[:, 2] = [0, 0, 1, 2, 3]
            knots_per_bs[:, -3] = [-4, -3, -2, -1, -1]
            knots_per_bs[:, -2] = [-3, -2, -1, -1, -1]
            knots_per_bs[:, -1] = [-2, -1, -1, -1, -1]

    # ----------
    # return

    if returnas == 'data':
        knots_per_bs = knots[knots_per_bs]

    if return_unique:
        return knots_per_bs, nbs
    else:
        return knots_per_bs


def _get_bs2d_func_cents(cents, deg=None, returnas=None):

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=str,
        default='data',
        allowed=['ind', 'data'],
    )

    nkpbs = 1 + deg
    size = cents.size
    nbs = size + deg
    cents_per_bs = np.zeros((nkpbs, nbs), dtype=int)

    if deg == 0:
        cents_per_bs[0, :] = np.arange(0, size)

    elif deg == 1:
        cents_per_bs[:, 1:-1] = np.array([
            np.arange(0, size-1),
            np.arange(1, size),
        ])
        cents_per_bs[:, 0] = [0, 0]
        cents_per_bs[:, -1] = [-1, -1]

    elif deg == 2:
        cents_per_bs[:, 2:-2] = np.array([
            np.arange(0, size-2),
            np.arange(1, size-1),
            np.arange(2, size),
        ])
        cents_per_bs[:, 0] = [0, 0, 0]
        cents_per_bs[:, 1] = [0, 0, 1]
        cents_per_bs[:, -2] = [-2, -1, -1]
        cents_per_bs[:, -1] = [-1, -1, -1]

    elif deg == 3:
        cents_per_bs[:, 3:-3] = np.array([
            np.arange(0, size-3),
            np.arange(1, size-2),
            np.arange(2, size-1),
            np.arange(3, size),
        ])
        cents_per_bs[:, 0] = [0, 0, 0, 0]
        cents_per_bs[:, 1] = [0, 0, 0, 1]
        cents_per_bs[:, 2] = [0, 0, 1, 2]
        cents_per_bs[:, -3] = [-3, -2, -1, -1]
        cents_per_bs[:, -2] = [-2, -1, -1, -1]
        cents_per_bs[:, -1] = [-1, -1, -1, -1]

    if returnas == 'data':
        cents_per_bs = cents[cents_per_bs]

    return cents_per_bs


def _get_bs2d_func_max(Rknots=None, Zknots=None, deg=None):

    knots_per_bs_R = _get_bs2d_func_knots(Rknots, deg=deg)
    knots_per_bs_Z = _get_bs2d_func_knots(Zknots, deg=deg)
    nbkbs = knots_per_bs_R.shape[0]

    if nbkbs % 2 == 0:
        ii = int(nbkbs/2)
        Rbs_cent = np.mean(knots_per_bs_R[ii-1:ii+1, :], axis=0)
        Zbs_cent = np.mean(knots_per_bs_Z[ii-1:ii+1, :], axis=0)
        if deg == 2:
            Rbs_cent[:deg] = [Rknots[0], 0.5*(Rknots[0] + Rknots[1])]
            Rbs_cent[-deg:] = [0.5*(Rknots[-2] + Rknots[-1]), Rknots[-1]]
            Zbs_cent[:deg] = [Zknots[0], 0.5*(Zknots[0] + Zknots[1])]
            Zbs_cent[-deg:] = [0.5*(Zknots[-2] + Zknots[-1]), Zknots[-1]]

    else:
        ii = int((nbkbs-1)/2)
        Rbs_cent = knots_per_bs_R[ii, :]
        Zbs_cent = knots_per_bs_Z[ii, :]
        if deg == 1:
            Rbs_cent[:deg] = Rknots[0]
            Rbs_cent[-deg:] = Rknots[-1]
            Zbs_cent[:deg] = Zknots[0]
            Zbs_cent[-deg:] = Zknots[-1]
        elif deg == 3:
            Rbs_cent[:deg] = [Rknots[0], 0.5*(Rknots[0]+Rknots[1]), Rknots[1]]
            Rbs_cent[-deg:] = [
                Rknots[-2], 0.5*(Rknots[-2]+Rknots[-1]), Rknots[-1],
            ]
            Zbs_cent[:deg] = [Zknots[0], 0.5*(Zknots[0]+Zknots[1]), Zknots[1]]
            Zbs_cent[-deg:] = [
                Zknots[-2], 0.5*(Zknots[-2]+Zknots[-1]), Zknots[-1],
            ]

    return Rbs_cent, Zbs_cent


def get_bs2d_RZ(deg=None, Rknots=None, Zknots=None):

    # ----------------
    # get knots per bspline, nb of bsplines...

    knots_per_bs_R = _get_bs2d_func_knots(Rknots, deg=deg, returnas='data')
    knots_per_bs_Z = _get_bs2d_func_knots(Zknots, deg=deg, returnas='data')
    nbkbs = knots_per_bs_R.shape[0]
    shapebs = (knots_per_bs_R.shape[1], knots_per_bs_Z.shape[1])

    # ----------------
    # get centers of bsplines

    Rbs_cent, Zbs_cent = _get_bs2d_func_max(
        Rknots=Rknots, Zknots=Zknots, deg=deg,
    )
    return shapebs, Rbs_cent, Zbs_cent, knots_per_bs_R, knots_per_bs_Z


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

    # TBC
    def ev_details(
        R,
        Z,
        clas=clas,
        indbs_tuple_flat=None,
        crop=None,
        cropbs=None,
        coefs=None,
        reshape=None,
    ):
        """ Return the value for each point summed on all bsplines """

        # check inputs
        _, _, _, r, z, crop = _get_bs2d_func_check(
            R=R,
            Z=Z,
            shapebs=shapebs,
        )

        # compute
        return RectBiv_scipy.ev_details(
            r,
            z,
            indbs_tuple_flat=indbs_tuple_flat,
            reshape=reshape,
        )

    # TBC
    def ev_sum(
        R,
        Z,
        coefs=None,
        clas=clas,
        crop=None,
        cropbs=None,
        indbs_tuple_flat=None,
        reshape=None,
    ):
        """ Return the value for each point summed on all bsplines """

        # check inputs
        coefs, _, _, r, z, crop = _get_bs2d_func_check(
            coefs=coefs,
            R=R,
            Z=Z,
            shapebs=shapebs,
            crop=crop,
            cropbs=cropbs,
        )

        # prepare
        nt = 1 if np.isscalar(coefs) else coefs.shape[0]
        shapepts = r.shape

        shape = tuple(np.r_[nt, shapepts])
        val = np.zeros(shape, dtype=float)
        cropbs_neg_flat = ~cropbs.ravel() if crop else None

        # compute
        if np.isscalar(coefs):
            val[0, ...] = RectBiv_scipy(
                r,
                z,
                grid=False,
                coefs=coefs,
                cropbs_neg_flat=cropbs_neg_flat,
            )

        else:
            for ii in range(coefs.shape[0]):
                val[ii, ...] = RectBiv_scipy(
                    r,
                    z,
                    grid=False,
                    coefs=coefs[ii, ...],
                    cropbs_neg_flat=cropbs_neg_flat,
                )

        return val

    return ev_details, ev_sum, clas
