

import numpy as np






def get_bs2d_func(
        deg=None,
        knotsr=None,
        cents=None,
):

    # ----------------
    # Pre-compute bsplines basis elements

    lbr = [
        scpinterp.BSpline.basis_element(
            knots_per_bs_R[:, ii],
            extrapolate=False,
        )
        for ii in range(shapebs[0])
    ]


    # ----------------
    # Define functions

    def func_details(
        R,
        Z,
        shapebs=shapebs,
        RectBiv_scipy=RectBiv_scipy,
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

    def func_sum(
        R,
        Z,
        coefs=None,
        shapebs=shapebs,
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
