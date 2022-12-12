

import numpy as np
import scipy.interpolate as scpinterp


from . import _utils_bsplines


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

        knots_per_bs_r = _utils_bsplines._get_knots_per_bs(
            knotsr, deg=deg, returnas='data',
        )
        knotsr_with_mult, nbsr = _utils_bsplines._get_knots_per_bs(
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

                knots_per_bsai = _utils_bsplines._get_knots_per_bs(
                    knotsa[ii],
                    deg=deg,
                    returnas='data',
                    poloidal=True,
                )

                nbsa[ii] = knots_per_bsai.shape[1]
                knots_per_bs_a.append(knots_per_bsai)

                for jj in range(nbsa[ii]):
                    kj = knots_per_bsai[:, jj]
                    lba[ii].append(scpinterp.BSpline.basis_element(
                        kj,
                        extrapolate=False,
                    ))
                    if kj[0] >= kj[-1]:
                        i2pi = np.r_[False, np.diff(kj) < 0].nonzero()[0][0]
                        lba[ii][-1].t[deg + i2pi:] += 2.*np.pi
        else:
            lba = None
            knots_per_bs_a = None

        # ----------------
        # bsplines centers

        cents_per_bs_r = _utils_bsplines._get_cents_per_bs(
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
                    cents_per_bs_a[ii] = _utils_bsplines._get_cents_per_bs(
                        0.5*(knotsa[ii][1:] + knotsa[ii][:-1]),
                        deg=deg,
                        returnas='data',
                        poloidal=True,
                    )

        # ----------------
        # bsplines apex

        apex_per_bs_r = _utils_bsplines._get_apex_per_bs(
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
                    apex_per_bs_a[ii] = _utils_bsplines._get_apex_per_bs(
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
                if len(self.shapebs) == 2:
                    err = True
                else:
                    shape = np.r_[np.ones((radius.ndim,), dtype=int), -1]
                    kpbrmin = self.knots_per_bs_r[0, :].reshape(shape)
                    kpbrmax = self.knots_per_bs_r[-1, :].reshape(shape)
                    indr = (
                        (radius[..., None] >= kpbrmin)
                        & (radius[..., None] <= kpbrmax)
                    )
                    for ii in range(radius.ndim):
                        indr = np.any(indr, axis=0)

                    if np.any((self.nbs_a_per_r > 1) & indr):
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

    def _check_coefs(self, coefs=None):
        """ None for ev_details, (nt, shapebs) for sum """
        if coefs is not None:
            assert coefs.ndim == len(self.shapebs) + 1
            assert coefs.shape[1:] == self.shapebs

    def __call__(
        self,
        # coordiantes
        radius=None,
        angle=None,
        # coefs
        coefs=None,
        # options
        radius_vs_time=None,
        val_out=None,
        # for purely radial only
        deriv=None,
        # for compatibility (unused)
        indbs_tf=None,
    ):
        """ Assumes

        coefs.shape = (nt, shapebs)

        """

        # ------------
        # check inputs

        if deriv is None:
            deriv = 0
        if self.knotsa is not None and deriv != 0:
            msg = "Derivatives not-implemented for non-purely radial bsplines!"
            raise Exception(msg)

        # coefs
        self._check_coefs(coefs=coefs)

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
                    )(radius[it, ...], nu=deriv)
            else:
                for it in range(nt):
                    val[it, ...] = scpinterp.BSpline(
                        self.knotsr_with_mult,
                        coefs[it, :],
                        self.deg,
                        extrapolate=False,
                    )(radius, nu=deriv)

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

        else:
            for ii, nbsa in enumerate(self.nbs_a_per_r):
                iok = (
                    (radius >= self.knots_per_bs_r[0, ii])
                    & ((radius < self.knots_per_bs_r[-1, ii]))
                )
                valr = self.lbr[ii](radius[iok])
                if nbsa == 1:
                    ind = self.func_coef_ind(ii, 0)
                    val[:, iok] += coefs[:, ind][:, None] * valr[None, ...]
                else:
                    for jj in range(nbsa):
                        kj = self.knots_per_bs_a[ii][:, jj]
                        if kj[0] > kj[-1]:
                            atemp = np.copy(angle[iok])
                            atemp[atemp < kj[0]] += 2.*np.pi
                            vala = self.lba[ii][jj](atemp)
                        else:
                            vala = self.lba[ii][jj](angle[iok])

                        iokj = np.isfinite(vala)
                        if np.any(iokj):
                            iok2 = np.copy(iok)
                            iok2[iok2] = iokj
                        else:
                            continue

                        if len(self.shapebs) == 1:
                            ind = self.func_coef_ind(ii, jj)
                            val[:, iok2] += (
                                coefs[:, ind][:, None]
                                * (valr[iokj]*vala[iokj])[None, ...]
                            )
                        else:
                            val[:, iok2] += (
                                coefs[:, ii, jj][:, None]
                                * (valr[iokj]*vala[iokj])[None, ...]
                            )

        # clean out-of-mesh
        if val_out is not False:
            # pts out 
            indout = (
                (radius < self.knotsr.min())
                | (radius > self.knotsr.max())
            )
            if radius_vs_time:
                val[indout] = val_out
            else:
                val[:, indout] = val_out

        return val

    def ev_details(
        self,
        # coordiantes
        radius=None,
        angle=None,
        # options
        indbs_tf=None,
        deriv=None,
        # for compatibility (unused)
        coefs=None,
        radius_vs_time=None,
        val_out=None,
    ):
        """ Assumes

        coefs.shape = (nt, shapebs)
        indbs_tf = flat array of int indices

        """

        # ------------
        # check inputs

        radius, angle = self._check_radiusangle_input(
            radius=radius,
            angle=angle,
        )

        if indbs_tf is None:
            nbs = self.nbs
        else:
            indbs_tf = np.atleast_1d(indbs_tf).ravel()
            assert 'int' in indbs_tf.dtype.name, indbs_tf
            assert np.unique(indbs_tf).size == indbs_tf.size
            nbs = indbs_tf.size

        if deriv is None:
            deriv = 0

        # --------
        # prepare

        shape = tuple(np.r_[radius.shape, nbs])
        val = np.zeros(shape)

        # ------------
        # compute

        if self.knotsa is None:

            # radius only
            ni = 0
            for ii in range(self.nbs_r):

                if indbs_tf is not None and ii not in indbs_tf:
                    continue

                iok = (
                    (radius >= self.knots_per_bs_r[0, ii])
                    & ((radius < self.knots_per_bs_r[-1, ii]))
                )
                if np.any(iok):
                    val[iok, ni] = self.lbr[ii](radius[iok], nu=deriv)
                ni += 1

        else:

            # if np.abs(np.nanmin(radius) - 0.1800309517775584) < 0.0001:
                # import pdb.set_trace()      # DB  # chan = 27

            # radius + angle
            ni = 0
            for ii, nbsa in enumerate(self.nbs_a_per_r):

                c0 = (
                    nbsa == 1
                    and indbs_tf is not None
                    and self.func_coef_ind(ii, 0) not in indbs_tf
                )
                if c0:
                    continue

                # compute valr
                iok = (
                    (radius >= self.knots_per_bs_r[0, ii])
                    & ((radius < self.knots_per_bs_r[-1, ii]))
                )

                if not np.any(iok):
                    ni += nbsa
                    continue

                valr = self.lbr[ii](radius[iok], nu=deriv)

                # compute vala
                if nbsa == 1:
                    val[iok, ni] = valr
                    ni += 1
                else:
                    if deriv > 0:
                        msg = "Derivative > 0 not implemented for 2d splines"
                        raise Exception(msg)

                    for jj in range(nbsa):
                        ind = self.func_coef_ind(ii, jj)

                        if indbs_tf is not None and ind not in indbs_tf:
                            continue

                        kj = self.knots_per_bs_a[ii][:, jj]
                        if kj[0] >= kj[-1]:
                            atemp = np.copy(angle[iok])
                            atemp[atemp < kj[0]] += 2.*np.pi
                            vala = self.lba[ii][jj](atemp)
                        else:
                            vala = self.lba[ii][jj](angle[iok])

                        iokj = np.isfinite(vala)
                        if np.any(iokj):
                            iok2 = np.copy(iok)
                            iok2[iok2] = iokj
                            val[iok2, ni] = valr[iokj]*vala[iokj]

                        # if ii == 4 and jj == 3:
                            # import pdb; pdb.set_trace()         # DB
                        ni += 1

        return val

    # -----------------
    # constraints methods
    # -----------------

    def get_constraints_out_rlim(
        self,
        rlim=None,
        rm=None,
        lim=None,
    ):
        """
        Return indices of bslines fully out of rlim (min or max)

        Assumes rlim is a 1d array, one per time step
        """


        # ------------
        # check inputs

        if rm not in ['rmin', 'rmax']:
            msg = f"Invalid rm!\nShould be in ['rmax', 'rmin']\nProvided: {rm}"
            raise Exception(msg)

        if np.isscalar(rlim):
            rlim = np.r_[rlim]

        if not (isinstance(rlim, np.ndarray) and rlim.ndim == 1):
            msg = "rlim must ba a 1d array!"
            raise Exception(msg)

        # ------------
        # get index of bsplines out of limits

        kpbsr = self.knots_per_bs_r
        if rm == 'rmax':
            if lim == 'allout':
                ind = np.all(
                    kpbsr[None, :, :] > rlim[:, None, None],
                    axis=1,
                )
            elif lim == 'outer':
                ind = np.sum(
                    kpbsr[None, :, :] > rlim[:, None, None],
                    axis=1,
                ) > 1
            elif lim == 'inner':
                ind = np.any(
                    kpbsr[None, :, :] > rlim[:, None, None],
                    axis=1,
                )

        else:
            if lim == 'allin':
                ind = np.all(
                    kpbsr[None, :, :] < rlim[:, None, None],
                    axis=1,
                )
            elif lim == 'inner':
                ind = np.sum(
                    kpbsr[None, :, :] < rlim[:, None, None],
                    axis=1,
                ) > 1
            elif lim == 'outer':
                ind = np.any(
                    kpbsr[None, :, :] < rlim[:, None, None],
                    axis=1,
                )

        if self.knotsa is None:
            pass
        else:
            ind = np.repeat(ind, self.nbs_a_per_r, axis=1)

        # ------------
        # get coefs / offset

        offset = np.zeros((self.nbs,), dtype=float)

        return ind, offset

    def get_bs_per_pts(self, radius=None, angle=None):
        """ Given points in radius / angle, return the indices of bsplines

        Assumes:
            - radius and angle are arrays of the same shape

        Return:
            - ind = (radius.shape, nbs), using flattened indexing

        """


    def get_constraints_deriv(
        self,
        deriv=None,
        rad=None,
        val=None,
    ):
        """
        To set constraints on a derivative
        Retrun indices of bsplines + coefs + offset

        Assumes:
            - deriv in ['deriv0', 'deriv1']
            - rad and val are 1d arrays of the same shape

        return as flattened nbsplines indexing

        """
        # ------------
        # check inputs

        ld = ['deriv0', 'deriv1']
        if deriv not in ld:
            msg = f"Arg deriv must be in {ld}!\n Provided: {deriv}"
            raise Exception(msg)

        # --------
        # compute

        # coefs per radius per bs (nrad, nbs)
        ideriv = int(deriv[-1])
        vv = self.ev_details(radius=rad, deriv=ideriv)

        # check conflicts
        indok = (vv != 0)
        if np.unique(indok, axis=0).shape[0] < indok.shape[0]:
            msg = f"Conflicting constraints on {deriv}:\n{indok}"
            raise Exception(msg)

        coefs = vv
        offset = np.repeat(val[:, None], self.nbs, axis=1)

        return indok, coefs, offset

    # -----------------
    # operator methods
    # -----------------

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
    raise NotImplementedError()
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
