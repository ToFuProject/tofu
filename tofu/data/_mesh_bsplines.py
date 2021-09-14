# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import scipy.interpolate as scpinterp


# #############################################################################
# #############################################################################
#                           Mesh2DRect - bsplines
# #############################################################################


def _get_bs2d_func_check(
    coefs=None,
    ii=None,
    jj=None,
    r=None,
    z=None,
    bsshape=None,
):

    # ii, jj
    if ii is None:
        ii = np.arange(0, shapebs[0])
    else:
        ii = np.atleast_1d(ii).ravel().astype(int)

    if jj is None:
        jj = np.arange(0, shapebs[1])
    else:
        jj = np.atleast_1d(ii).ravel().astype(int)
    assert ii.size == jj.size

    # r, z
    r = np.atleast_1d(r)
    z = np.atleast_1d(z)
    assert r.shape == z.shape

    # coefs
    if coefs is None:
        shapec = tuple(np.r_[1, shapebs])
        coefs = np.ones(shapec, dtype=float)
    else:
        coefs = np.atleast_1d(coefs)
        if coefs.ndim < len(shapebs):
            msg = ()
        if coefs.ndim == len(shapebs):
            coefs = coefs.reshape(tuple(np.r_[1, coefs.shape]))
        if coefs.shape[1:] != shapebs:
            msg = ()
    return coefs, ii, jj, r, z


def _get_bs2d_func_knots(x, ii, deg=None):


def get_bs2d_func(deg=None, Rknots=None, Zknots=None):


    if deg == 0:

        nRbs = Rknots.size - 1
        nZbs = Zknots.size - 1
        bsshape = (nRbs, nZbs)

        RectBiv = [[None for jj in range(nfZ)] for ii in range(nfR)]
        for ii in range(shapebs[0]):
            for jj in range(shapebs[1]):
                knotsr = _get_bs2d_func_knots(Rknots, ii, deg=deg)
                knotsz = _get_bs2d_func_knots(Zknots, jj, deg=deg)
                br = scpinterp.Bspline.basis_element(knotsr, extrapolate=False)
                bz = scpinterp.Bspline.basis_element(knotsz, extrapolate=False)
                def func(r, z, coefs=None, br=br, bz=bz):
                    if hasattr(coef, '__iter__'):
                        val = coefs[:, None] * (br(r)*bz(z))[None, :]
                    else:
                        val = coefs * br(r)*bz(z)
                    return val
                RectBiv[ii][jj] = func


        def RectBiv_details(
            r,
            z,
            coefs=None,
            ii=None,
            jj=None,
            shapebs=shapebs,
        ):

            coefs, ii, jj, r, z = _get_bs2d_func_check(
                coefs=coefs,
                ii=ii,
                jj=jj,
                r=r,
                z=z,
            )
            nt = coefs.shape[0]
            shapepts = r.shape

            if ii is None:
                shape = tuple(np.r_[nt, shapepts, shapebs])
                val = np.full(shape, np.nan)
                for ii in range(shapebs[0]):
                    for jj in range(shapebs[1]):
                        val[..., ii, jj] = RectBiv[ii][jj](
                            r,
                            z,
                            coef=coefs[:, ii, jj],
                        )
            else:
                shape = tuple(np.r_[nt, shapepts])
                val = RectBiv[ii][jj](r, z, coef=coefs[:, ii, jj])
            return val

        def RectBiv_sum(r, z, coefs=None, ii=None, jj=None):
            coefs, ii, jj, r, z = _get_bs2d_func_check(
                coefs=coefs,
                ii=ii,
                jj=jj,
                r=r,
                z=z,
            )
            nt = coefs.shape[0]
            shapepts = r.shape

            shape = tuple(np.r_[nt, shapepts])
            val = np.zeros(shape, dtype=float)
            for ii in range(shapebs[0]):
                for jj in range(shapebs[1]):
                    boundr = _get_bs2d_func_bounds(Rknots, ii, deg=deg)
                    boundz = _get_bs2d_func_bounds(Zknots, jj, deg=deg)
                    indok = (
                        (boundr[0] <= r <= boundr[1])
                        & (boundz[0] <= z <= boundz[1])
                    )
                    val[indok] += RectBiv[ii][jj](
                        r[indok],
                        z[indok],
                        coef=coefs[:, ii, jj],
                    )
            return val

    return RectBiv_details, RectBiv_sum
