

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


class BivariateSplinePolar():
    """ Subclass for tofu

    Defined from knots (unique) and deg
    coefs set to 1 by default

    Used self.set_coefs() to update
    """

    def __init__(self, knotsr=None, knotsa=None, deg=None):

        assert deg in [0, 1, 2, 3], deg

        # get knots pr bs
        self._get_knots_per_bs_for_basis_elements(
            knotsr=knotsr,
            knotsa=knotsa,
            deg=deg,
        )

        # get nbs
        self.nbs = np.sum(self.nbs_a_per_r)

        if self.knotsa is None:
            self.shapebs = (self.nbs,)
        elif np.unique(self.nbs_a_per_r).size == 1:
            self.shapebs = (self.nbs_r, self.nbs_a_per_r[0])
        else:
            self.shapebs = (self.nbs,)

        # self.tck = [knotsR, knotsZ, coefs]
        # self.degrees = [deg, deg]

    def _get_knots_per_bs_for_basis_elements(
        self,
        knotsr=None,
        knotsa=None,
        deg=None,
    ):

        # ------------------------------------
        # get knots per bs in radius direction

        knots_per_bs_r = _bsplines_utils._get_bs2d_func_knots(
            knotsr, deg=deg, returnas='data',
        )

        nbsr = knots_per_bs_r.shape[1]

        # ----------------
        # check angle

        if isinstance(knotsa, np.ndarray) and knotsa.ndim == 1:
            knotsa = [knotsa for ii in range(nbsr)]

        c0 = (
            isinstance(knotsa, list)
            and len(knotsa) == nbsr
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
                for aa in knotsa
            ])
        )
        if knotsa is not None and not c0:
            msg = (
                f"Arg angle must a list of {nbsr} elements, where each can be:\n"
                "\t- None: no poloidal discretization\n"
                "\t- array: 1d array of sorted angles in radians in [-pi, pi[\n"
                f"Provided: {knotsa}"
            )
            raise Exception(msg)

        # make sure to remove double angles
        # particularly at edges (-pi vs pi)
        if knotsa is not None:
            for ii, aa in enumerate(knotsa):
                if aa is not None:
                    aa = np.unique(np.arctan2(np.sin(aa), np.cos(aa)))
                    damin = np.min(np.diff(aa))
                    if np.abs(aa[-1] - (aa[0] + 2.*np.pi)) < damin/100.:
                        aa = aa[:-1]
                    knotsa[ii] = aa

        # ----------------
        # Pre-compute bsplines basis elements

        lbr = [
            scpinterp.BSpline.basis_element(
                knots_per_bs_r[:, ii],
                extrapolate=False,
            )
            for ii in range(nbsr)
        ]

        # angle bsplines
        nbsa = np.ones((nbsr,), dtype=int)
        if knotsa is not None:
            lba = []
            for ii in range(nbsr):

                if knotsa[ii] is None:
                    lba.append(None)
                    continue

                knots_per_bsai = _bsplines_utils._get_bs_func_knots_poloidal(
                    knotsa[ii], deg=deg, returnas='data',
                )

                nbsa[ii] = knots_per_bsai.shape[1]
                lba.append([
                    scpinterp.BSpline.basis_element(
                        knots_per_bsai[:, jj],
                        extrapolate=False,
                    )
                    for jj in range(nbsa[ii])
                ])
        else:
            lba = None

        # ------
        # store

        self.knotsr = knotsr
        self.knotsa = knotsa
        self.knots_per_bs_r = knots_per_bs_r
        self.knots_per_bs_a = lba
        self.nbs_r = nbsr
        self.nbs_a_per_r = nbsa
        self.lbr = lbr
        self.lba = lba

        # -----------------
        # compute knots per bs for flattened 

        # if deg == 0:
            # pass
        # else:
            # knots_per_bs_r = np.concatenate(
                # (
                    # np.tile(knots_per_bs_r[0, :] - 1, (deg, 1)),
                    # knots_per_bs_r,
                    # np.tile(knots_per_bs_r[-1, :] + 1, (deg, 1)),
                # ),
                # axis=0,
            # )

            # for ii in range(nbsr):
                # if knotsa[ii] is not None:
                    # knots_per_bs_a[ii] = np.concatenate(
                        # (
                            # np.tile(knots_per_bs_a[ii][0, :] - 1, (deg, 1)),
                            # knots_per_bs_a[ii],
                            # np.tile(knots_per_bs_a[ii][-1, :] + 1, (deg, 1)),
                        # ),
                        # axis=0,
                    # )

        # self.knots_per_bs_r_pad = np.asfortranarray(knots_per_bs_r)
        # self.knots_per_bs_a_pad = np.asfortranarray(knots_per_bs_a)

    def set_coefs(
        self,
        coefs=None,
    ):

        # ------------
        # check inputs

        if coefs is None:
            msg = "Please provide coefs!"
            raise Exception(msg)

        shape = self.shapebs
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

    def _check_radiusangle_input(self, radius=None, angle=None):

        if not isinstance(radius, np.ndarray):
            radius = np.atleast_1d(radius)

        if self.knotsa is not None:
            err = False
            if angle is not None:
                if not isinstance(angle, np.ndarray):
                    angle = np.atleast_1d(angle)
                if angle.shape != radius.shape:
                    err = True
            else:
                err = True
            if err:
                msg = (
                    "Arg angle must be a np.ndarray same shape as radius!\n"
                    f"\t- radius.shape = {radius.shape}\n"
                    f"\t- angle = {angle}\n"
                    "It should be an array of poloidal angles in radians"
                )
                raise Exception(msg)

        return radius, angle

    def __call__(
        self,
        radius,
        angle,
        coefs=None,
        **kwdargs,
    ):

        # ------------
        # check inputs

        radius, angle = self._check_radiusangle_input(
            radius=radius,
            angle=angle,
        )

        # ------------
        # prepare

        self.set_coefs(coefs=coefs)
        val = np.full(radius.shape, np.nan)

        # ------------
        # compute

        # TBF
        if self.knotsa is None:
            val = scpinterp.BSpline(
                self.knotsr,
                self.coefs,
                self.deg,
                extrapolate=False,
            )(radius)

        else:
            for ii, nbsa in enumerate(self.nbs_a_per_r):
                valr = self.lbr[ii]()
                if nbsa == 1:
                    val += valr
                else:
                    for jj in range(nbsa):
                        vala = self.lba[ii][jj]()
                        val += valr*vala

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
        crop=None,
        cropbs=None,
        PolarBiv_scipy=PolarBiv_scipy,
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
            val[0, ...] = PolarBiv_scipy(
                r,
                z,
                grid=False,
                coefs=coefs,
                cropbs_neg_flat=cropbs_neg_flat,
            )

        else:
            for ii in range(coefs.shape[0]):
                val[ii, ...] = PolarBiv_scipy(
                    r,
                    z,
                    grid=False,
                    coefs=coefs[ii, ...],
                    cropbs_neg_flat=cropbs_neg_flat,
                )

        return val


    return func_details, func_sum, PolarBiv_scipy
