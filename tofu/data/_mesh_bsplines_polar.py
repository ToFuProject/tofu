

import numpy as np
import scipy.interpolate as scpinterp


from . import _bsplines_utils


# #############################################################################
# #############################################################################
#                   utility
# #############################################################################


def _get_bs2d_func_check(
    coefs=None,
    R=None,
    Z=None,
    radius=None,
    angle=None,
):

    # -------------------------
    # (R, Z) vs (radius, angle)

    c0 = (
    )
    if not c0:
        raise Exception(msg)

    # ------
    # R, Z

    if R is not None:
        if not isinstance(R, np.ndarray):
            R = np.atleast_1d(R)
        if not isinstance(Z, np.ndarray):
            Z = np.atleast_1d(Z)
        assert R.shape == Z.shape

    # -------------
    # radius, angle

    if radius is not None:
        if not isinstance(radius, np.ndarray):
            radius = np.atleast_1d(radius)

    # -------
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

    return coefs, radius, angle


# #############################################################################
# #############################################################################
#                   class
# #############################################################################


class BivariateSplinePolar(scpinterp.BivariateSpline):
    """ Subclass for tofu

    Defined from knots (unique) and deg
    coefs set to 1 by default

    Used self.set_coefs() to update
    """

    def __init__(self, knotsr=None, knotsa=None, deg=None):

        assert deg in [0, 1, 2, 3]

        # get knots pr bs
        self._get_knots_per_bs_for_basis_elements(
            knotsr=knotsr,
            knotsa=knotsa,
            deg=deg,
        )

        # full knots with multiplicity
        knotsR, nbsR = _bsplines_utils._get_bs2d_func_knots(
            knotsR, deg=deg, returnas='data', return_unique=True,
        )
        knotsZ, nbsZ = _bsplines_utils._get_bs2d_func_knots(
            knotsZ, deg=deg, returnas='data', return_unique=True,
        )

        coefs = np.ones((nbsR*nbsZ,), dtype=float)

        self.__nbs = (nbsR, nbsZ)
        self.tck = [knotsR, knotsZ, coefs]
        self.degrees = [deg, deg]

        # shapebs
        self.shapebs = shapebs

    def _get_knots_per_bs_for_basis_elements(
        self,
        knotsR=None,
        knotsZ=None,
        deg=None,
    ):

        # TBF !!!

        # added for details
        knots_per_bs_r = _bsplines_utils._get_bs2d_func_knots(
            knotsR, deg=deg, returnas='data',
        )

        if knotsa is not None:


            knots_per_bs_a = [
                _bsplines_utils._get_bs2d_func_knots(
                    knotsa[ii], deg=deg, returnas='data',
                )
            ]

        self.knots_per_bs_x = knots_per_bs_x
        self.knots_per_bs_y = knots_per_bs_y

        if deg == 0:
            pass
        else:
            knots_per_bs_x = np.concatenate(
                (
                    np.tile(knots_per_bs_x[0, :] - 1, (deg, 1)),
                    knots_per_bs_x,
                    np.tile(knots_per_bs_x[-1, :] + 1, (deg, 1)),
                ),
                axis=0,
            )
            knots_per_bs_y = np.concatenate(
                (
                    np.tile(knots_per_bs_y[0, :] - 1, (deg, 1)),
                    knots_per_bs_y,
                    np.tile(knots_per_bs_y[-1, :] + 1, (deg, 1)),
                ),
                axis=0,
            )

        self.knots_per_bs_x_pad = np.asfortranarray(knots_per_bs_x)
        self.knots_per_bs_y_pad = np.asfortranarray(knots_per_bs_y)

    def set_coefs(
        self,
        coefs=None,
        cropbs_neg_flat=None,
    ):

        # ------------
        # check inputs

        if coefs is None:
            msg = "Please provide coefs!"
            raise Exception(msg)

        shape = (self.__nbs[0]*self.__nbs[1],)
        if np.isscalar(coefs):
            self.tck[2][...] = coefs
        else:
            if not isinstance(coefs, np.ndarray):
                coefs = np.asarray(coefs)
            if coefs.ndim > 1 and coefs.size == shape[0]:
                coefs = coefs.ravel()
            if coefs.shape != shape:
                msg = (
                    "Arg coefs has wrong shape!\n"
                    f"\t- expected: {shape}\n"
                    f"\t- provided: {coefs.shape}\n"
                )
                raise Exception(msg)
            self.tck[2][...] = coefs

        # ------------
        # crop and set

        if cropbs_neg_flat is not None:
            self.tck[2][cropbs_neg_flat] = 0.

    def __call__(
        self,
        x,
        y,
        coefs=None,
        cropbs_neg_flat=None,
        **kwdargs,
    ):

        # prepare
        self.set_coefs(
            coefs=coefs,
            cropbs_neg_flat=cropbs_neg_flat,
        )

        # compute
        val = super().__call__(x, y, **kwdargs)

        # clean
        indout = (
            (x < self.tck[0][0]) | (x > self.tck[0][-1])
            | (y < self.tck[1][0]) | (y > self.tck[1][-1])
        )
        val[indout] = np.nan
        return val

    def ev_details(
        self,
        x,
        y,
        indbs_tuple_flat=None,
        reshape=None,
    ):

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

        reshape = _generic_check._check_var(
            reshape, 'reshape',
            default=True,
            types=bool,
        )

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
            val = np.reshape(val, tuple(np.r_[shape, -1]))

        return val

    def get_overlap(self):
        return _get_overlap(
            deg=self.degrees[0],
            knotsx=self.knots_per_bs_x,
            knotsy=self.knots_per_bs_y,
            shapebs=self.shapebs,
        )

    def get_operator(
        self,
        operator=None,
        geometry=None,
        cropbs_flat=None,
        # specific to deg = 0
        cropbs=None,
        centered=None,
        # to return gradR, gradZ, for D1N2 deg 0, for tomotok
        returnas_element=None,
    ):
        """ Get desired operator """
        return _mesh_bsplines_operators_rect.get_mesh2dRect_operators(
            deg=self.degrees[0],
            operator=operator,
            geometry=geometry,
            knotsx_mult=self.tck[0],
            knotsy_mult=self.tck[1],
            knotsx_per_bs=self.knots_per_bs_x,
            knotsy_per_bs=self.knots_per_bs_y,
            overlap=self.get_overlap(),
            cropbs_flat=cropbs_flat,
            # specific to deg = 0
            cropbs=cropbs,
            centered=centered,
            # to return gradR, gradZ, for D1N2 deg 0, for tomotok
            returnas_element=returnas_element,
        )


# #############################################################################
# #############################################################################
#                       Mesh2DPolar - bsplines - overlap
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
#                   Main
# #############################################################################


def get_bs2d_func(
    deg=None,
    knotsr=None,
    angle=None,
    knots_per_bs_r=None,
    coll=None,
):

    # ----------------
    # check angle

    nbsr = knots_per_bs_r.shape[1]
    if isinstance(angle, np.ndarray) and angle.ndim == 1:
        angle = []

    c0 = (
        isinstance(angle, list)
        and len(angle) == nbsr
        and all([
            aa is None
            or (
                isinstance(aa, np.ndarray)
                and aa.ndim == 1
                and np.allclose(
                    aa,
                    np.unique(np.arctan2(np.sin(aa), np.cos(aa))),
                    equal_nan=False,
                )
            )
        ])
    )
    if not c0:
        msg = (
            f"Arg angle must a list of {nbsr} elements, where each can be:\n"
            "\t- None: no poloidal discretization\n"
            "\t- array: 1d array of sorted angles in radians in [-pi, pi]\n"
            f"Provided: {angles}"
        )
        raise Exception(msg)

    # ----------------
    # Pre-compute bsplines basis elements

    knots_per_bs_r = _bsplines_utils._get_bs2d_func_knots(
        knotsr, deg=deg, returnas='data',
    )

    lbr = [
        scpinterp.BSpline.basis_element(
            knots_per_bs_r[:, ii],
            extrapolate=False,
        )
        for ii in range(nbsr)
    ]

    # angle bsplines
    if angle is not None:
        lba = []
        nbsa = np.zeros((nbsr,), dtype=int)
        for ii in range(nbsr):

            if angle[ii] is None:
                lba.append(None)
                continue

            knots_per_bsai = _bsplines_utils._get_bs2d_func_knots(
                angle[ii], deg=deg, returnas='data',
            )

            nbsa[ii] = knots_per_bsai.shape[1]
            lba.append([
                scpinterp.BSpline.basis_element(
                    knots_per_bsai[:, jj],
                    extrapolate=False,
                )
                for jj in range(nbsa[ii])
            ])

    # ----------------
    # Define functions

    PolarBiv_scipy = BivariateSplinePolar(
        knotsr=knotsr,
        knotsa=angle,
        deg=deg,
    )

    def func_details(
        # coordinates
        R=None,
        Z=None,
        radius=None,
        angle=None,
        # parameters
        coll=None,
        coefs=None,
    ):
        """ Return the value for each point summed on all bsplines """

        # check inputs
        coefs, radius, angle = _get_bs2d_func_check(
            R=R,
            Z=Z,
            radius=radius,
            angle=angle,
            coefs=coefs,
        )

        # compute
        return RectBiv_scipy.ev_details(
            r,
            z,
            indbs_tuple_flat=indbs_tuple_flat,
            reshape=reshape,
        )

    def func_sum(
        R,
        Z,
        coefs=None,
        RectBiv=RectBiv,
        crop=None,
        cropbs=None,
        RectBiv_scipy=RectBiv_scipy,
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


    return func_details, func_sum, clas
