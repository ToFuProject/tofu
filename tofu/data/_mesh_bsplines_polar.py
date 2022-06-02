

import numpy as np
import scipy.interpolate as scpinterp


from . import _bsplines_utils


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

        # func to get coef from (ii, jj)
        if self.knotsa is not None:
            self.func_coef_ind = self._get_func_coef_ind()

        # deg
        self.deg = deg

    def _get_knots_per_bs_for_basis_elements(
        self,
        knotsr=None,
        knotsa=None,
        deg=None,
    ):

        # ------------------------------------
        # get knots per bs in radius direction

        knots_per_bs_r = _bsplines_utils._get_knots_per_bs(
            knotsr, deg=deg, returnas='data',
        )
        knotsr_with_mult, nbsr = _bsplines_utils._get_knots_per_bs(
            knotsr, deg=deg, returnas='data', return_unique=True,
        )

        if nbsr != knots_per_bs_r.shape[1]:
            msg = "Inconsistent nb. of splines in r direction"
            raise Exception(msg)

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
            lba = [[] for ii in range(nbsr)]
            knots_per_bs_a = []
            for ii in range(nbsr):

                if knotsa[ii] is None:
                    lba.append(None)
                    knots_per_bs_a.append(None)
                    continue

                knots_per_bsai = _bsplines_utils._get_knots_per_bs(
                    knotsa[ii],
                    deg=deg,
                    returnas='data',
                    poloidal=True,
                )

                nbsa[ii] = knots_per_bsai.shape[1]
                knots_per_bs_a.append(knots_per_bsai)

                for jj in range(nbsa[ii]):
                    kj = knots_per_bsai[:, jj]
                    if kj[0] > kj[-1]:
                        i2pi = np.r_[False, np.diff(kj) < 0]
                        kj = np.copy(kj)
                        kj[i2pi] += 2.*np.pi
                    lba[ii].append(scpinterp.BSpline.basis_element(
                        kj,
                        extrapolate=False,
                    ))
        else:
            lba = None
            knots_per_bs_a = None

        # ----------------
        # bsplines centers

        cents_per_bs_r = _bsplines_utils._get_cents_per_bs(
            0.5*(knotsr[1:] + knotsr[:-1]),
            deg=deg,
            returnas='data',
        )
        if knotsa is None:
            cents_per_bs_a = None
        else:
            cents_per_bs_a = [None for ii in range(nbsr)]
            for ii in range(nbsr):
                if knotsa[ii] is None:
                    pass
                else:
                    cents_per_bs_a[ii] = _bsplines_utils._get_cents_per_bs(
                        0.5*(knotsa[ii][1:] + knotsa[ii][:-1]),
                        deg=deg,
                        returnas='data',
                        poloidal=True,
                    )

        # ----------------
        # bsplines apex

        apex_per_bs_r = _bsplines_utils._get_apex_per_bs(
            knots=knotsr,
            knots_per_bs=knots_per_bs_r,
            deg=deg,
        )
        if knotsa is None:
            apex_per_bs_a = None
        else:
            apex_per_bs_a = [None for ii in range(nbsr)]
            for ii in range(nbsr):
                if knotsa[ii] is None:
                    pass
                else:
                    apex_per_bs_a[ii] = _bsplines_utils._get_apex_per_bs(
            knots=knotsa[ii],
            knots_per_bs=knots_per_bs_a[ii],
            deg=deg,
            poloidal=True,
        )

        # ------
        # store

        self.knotsr = knotsr
        self.knotsa = knotsa
        self.knotsr_with_mult = knotsr_with_mult
        self.knotsa_with_mult = knotsa
        self.knots_per_bs_r = knots_per_bs_r
        self.knots_per_bs_a = knots_per_bs_a
        self.cents_per_bs_r = cents_per_bs_r
        self.cents_per_bs_a = cents_per_bs_a
        self.apex_per_bs_r = apex_per_bs_r
        self.apex_per_bs_a = apex_per_bs_a
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

    def _get_func_coef_ind(self):
        def func(ii, jj):
            return np.r_[0, np.cumsum(self.nbs_a_per_r)][ii] + jj
        return func

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
        # coordiantes
        radius=None,
        angle=None,
        # coefs
        coefs=None,
        # options
        radius_vs_time=None,
        shapebs=None,
    ):
        """ Assumes

        coefs.shape = (nt, shapebs)

        """

        # ------------
        # check inputs

        radius, angle = self._check_radiusangle_input(
            radius=radius,
            angle=angle,
        )
        nt = coefs.shape[0]

        if radius_vs_time:
            val = np.zeros(radius.shape)
        else:
            val = np.zeros(tuple(np.r_[nt, radius.shape]))

        # ------------
        # compute

        if self.knotsa is None:
            if radius_vs_time:
                for it in range(nt):
                    val[it, ...] = scpinterp.BSpline(
                        self.knotsr_with_mult,
                        coefs[it, :],
                        self.deg,
                        extrapolate=False,
                    )(radius[it, ...])
            else:
                for it in range(nt):
                    val[it, ...] = scpinterp.BSpline(
                        self.knotsr_with_mult,
                        coefs[it, :],
                        self.deg,
                        extrapolate=False,
                    )(radius)

        elif radius_vs_time:

            for it in range(nt):
                for ii, nbsa in enumerate(self.nbs_a_per_r):
                    iok = (
                        (radius[it, ...] >= self.knots_per_bs_r[0, ii])
                        & ((radius[it, ...] < self.knots_per_bs_r[-1, ii]))
                    )
                    valr = self.lbr[ii](radius[it, iok])
                    if nbsa == 1:
                        ind = self.func_coef_ind(ii, 0)
                        val[it, iok] += coefs[it, ind] * valr
                    else:
                        for jj in range(nbsa):
                            kj = self.knots_per_bs_a[ii][:, jj]
                            if kj[0] > kj[-1]:
                                atemp = np.copy(angle[it, iok])
                                atemp[atemp < kj[0]] += 2.*np.pi
                                vala = self.lba[ii][jj](atemp)
                            else:
                                vala = self.lba[ii][jj](angle[it, iok])

                            iokj = ~np.isnan(vala)
                            if np.any(iokj):
                                iok2 = np.copy(iok)
                                iok2[iok2] = iokj
                            else:
                                continue

                            if len(self.shapebs) == 1:
                                ind = self.func_coef_ind(ii, jj)
                                val[it, iok2] += (
                                    coefs[it, ind] * (valr[iokj]*vala[iokj])
                                )
                            else:
                                val[it, iok2] += (
                                    coefs[it, ii, jj] * (valr[iokj]*vala[iokj])
                                )

            # pts out 
            indout = (
                (radius < self.knotsr.min())
                | (radius > self.knotsr.max())
            )
            val[indout] = np.nan


        else:
            for ii, nbsa in enumerate(self.nbs_a_per_r):
                iok = (
                    (radius >= self.knots_per_bs_r[0, ii])
                    & ((radius < self.knots_per_bs_r[-1, ii]))
                )
                valr = self.lbr[ii](radius[iok])
                if nbsa == 1:
                    ind = self.func_coef_ind(ii, 0)
                    val[:, iok] += coefs[:, ind] * valr[None, ...]
                else:
                    for jj in range(nbsa):
                        kj = self.knots_per_bs_a[ii][:, jj]
                        if kj[0] > kj[-1]:
                            atemp = np.copy(angle[iok])
                            atemp[atemp < kj[0]] += 2.*np.pi
                            vala = self.lba[ii][jj](atemp)
                        else:
                            vala = self.lba[ii][jj](angle[iok])

                        iokj = ~np.isnan(vala)
                        if np.any(iokj):
                            iok2 = np.copy(iok)
                            iok2[iok2] = iokj

                        if len(self.shapebs) == 1:
                            ind = self.func_coef_ind(ii, jj)
                            val[:, iok2] += (
                                coefs[:, ind]
                                * (valr[iokj]*vala[iokj])[None, ...]
                            )
                        else:
                            val[:, iok2] += (
                                coefs[:, ii, jj]
                                * (valr[iokj]*vala[iokj])[None, ...]
                            )

            # pts out 
            indout = (
                (radius < self.knotsr.min())
                | (radius > self.knotsr.max())
            )
            val[:, indout] = np.nan

        return val

    def ev_details(
        self,
        # coordiantes
        radius=None,
        angle=None,
        # coefs
        coefs=None,
        # options
        radius_vs_time=None,
        indbs=None,
        shapebs=None,
    ):
        """ Assumes

        coefs.shape = (nt, shapebs)

        """

        # ------------
        # check inputs

        radius, angle = self._check_radiusangle_input(
            radius=radius,
            angle=angle,
        )
        nt = coefs.shape[0]

        # --------
        # prepare

        if radius_vs_time:
            shape = tuple(np.r_[radius.shape, self.nbs])
        else:
            shape = tuple(np.r_[nt, radius.shape, self.nbs])
        val = np.zeros(shape)

        # ------------
        # compute

        if self.knotsa is None:
            if radius_vs_time:
                for ii in range(self.nbs_r):
                    for it in range(nt):
                        iok = (
                            (radius[it, ...] >= self.knots_per_bs_r[0, ii])
                            & ((radius[it, ...] < self.knots_per_bs_r[-1, ii]))
                        )
                        valr = self.lbr[ii](radius[it, iok])
                        val[it, iok, ii] = coefs[it, ii] * valr
            else:
                cresh = tuple([nt] + [1 for ii in radius.shape])
                for ii in range(self.nbs_r):
                    iok = (
                        (radius >= self.knots_per_bs_r[0, ii])
                        & ((radius < self.knots_per_bs_r[-1, ii]))
                    )
                    valr = self.lbr[ii](radius[iok])
                    val[iok, ii] = coefs[:, ii].reshape(cresh) * valr[None, ...]

        elif radius_vs_time:

            for it in range(nt):
                for ii, nbsa in enumerate(self.nbs_a_per_r):
                    iok = (
                        (radius[it, ...] >= self.knots_per_bs_r[0, ii])
                        & ((radius[it, ...] < self.knots_per_bs_r[-1, ii]))
                    )
                    valr = self.lbr[ii](radius[it, iok])
                    if nbsa == 1:
                        ind = self.func_coef_ind(ii, 0)
                        val[it, iok, ind] = coefs[it, ind] * valr
                    else:
                        for jj in range(nbsa):
                            kj = self.knots_per_bs_a[ii][:, jj]
                            if kj[0] > kj[-1]:
                                atemp = np.copy(angle[it, iok])
                                atemp[atemp < kj[0]] += 2.*np.pi
                                vala = self.lba[ii][jj](atemp)
                            else:
                                vala = self.lba[ii][jj](angle[it, iok])

                            iokj = ~np.isnan(vala)
                            if np.any(iokj):
                                iok2 = np.copy(iok)
                                iok2[iok2] = iokj
                            else:
                                continue

                            ind = self.func_coef_ind(ii, jj)
                            if len(self.shapebs) == 1:
                                val[it, iok2, ind] = (
                                    coefs[it, ind] * (valr[iokj]*vala[iokj])
                                )
                            else:
                                val[it, iok2, ind] = (
                                    coefs[it, ii, jj] * (valr[iokj]*vala[iokj])
                                )

            # # pts out 
            # indout = (
                # (radius < self.knotsr.min())
                # | (radius > self.knotsr.max())
            # )
            # val[indout] = np.nan


        else:
            for ii, nbsa in enumerate(self.nbs_a_per_r):
                iok = (
                    (radius >= self.knots_per_bs_r[0, ii])
                    & ((radius < self.knots_per_bs_r[-1, ii]))
                )
                valr = self.lbr[ii](radius[iok])
                if nbsa == 1:
                    ind = self.func_coef_ind(ii, 0)
                    val[:, iok, ind] = coefs[:, ind] * valr[None, ...]
                else:
                    for jj in range(nbsa):
                        kj = self.knots_per_bs_a[ii][:, jj]
                        if kj[0] > kj[-1]:
                            atemp = np.copy(angle[iok])
                            atemp[atemp < kj[0]] += 2.*np.pi
                            vala = self.lba[ii][jj](atemp)
                        else:
                            vala = self.lba[ii][jj](angle[iok])

                        iokj = ~np.isnan(vala)
                        if np.any(iokj):
                            iok2 = np.copy(iok)
                            iok2[iok2] = iokj

                        ind = self.func_coef_ind(ii, jj)
                        if len(self.shapebs) == 1:
                            val[:, iok2, ind] = (
                                coefs[:, ind]
                                * (valr[iokj]*vala[iokj])[None, ...]
                            )
                        else:
                            val[:, iok2, ind] = (
                                coefs[:, ii, jj]
                                * (valr[iokj]*vala[iokj])[None, ...]
                            )

            # # pts out 
            # indout = (
                # (radius < self.knotsr.min())
                # | (radius > self.knotsr.max())
            # )
            # val[:, indout] = np.nan

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

        msg = (
            "Operator not implemented yet for polar bsplines!"
        )
        raise NotImplementedError(msg)


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

    return PolarBiv_scipy.ev_details, PolarBiv_scipy.__call__, PolarBiv_scipy
