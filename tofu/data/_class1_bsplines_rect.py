# -*- coding: utf-8 -*-


# Built-in
import itertools as itt


# Common
import numpy as np
import scipy.interpolate as scpinterp


# specific
from . import _generic_check
from . import _utils_bsplines
from . import _class1_bsplines_operators_rect


# #############################################################################
# #############################################################################
#                       BivariateSplineRect - scipy subclass
# #############################################################################


class BivariateSplineRect(scpinterp.BivariateSpline):
    """ Subclass for tofu

    Defined from knots (unique) and deg
    coefs set to 1 by default

    Used self.set_coefs() to update
    """

    def __init__(self, knotsR=None, knotsZ=None, deg=None, shapebs=None):

        assert np.allclose(np.unique(knotsR), knotsR)
        assert np.allclose(np.unique(knotsZ), knotsZ)
        assert deg in [0, 1, 2, 3]

        # get knots pr bs
        self._get_knots_per_bs_for_basis_elements(
            knotsR=knotsR,
            knotsZ=knotsZ,
            deg=deg,
        )

        # full knots with multiplicity
        knotsR, nbsR = _utils_bsplines._get_knots_per_bs(
            knotsR, deg=deg, returnas='data', return_unique=True,
        )
        knotsZ, nbsZ = _utils_bsplines._get_knots_per_bs(
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

        # added for details
        knots_per_bs_x = _utils_bsplines._get_knots_per_bs(
            knotsR, deg=deg, returnas='data',
        )
        knots_per_bs_y = _utils_bsplines._get_knots_per_bs(
            knotsZ, deg=deg, returnas='data',
        )

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

    def _check_coefs(self, coefs=None, axis=None, r=None):
        """ None for ev_details, (nt, shapebs) for sum """
        c0 = (
            isinstance(coefs, np.ndarray)
            and coefs.ndim >= len(self.shapebs)
            and len(axis) == 2
            and axis[1] == axis[0] + 1
            and coefs.shape[axis[0]] == self.shapebs[0]
            and coefs.shape[axis[1]] == self.shapebs[1]
        )
        if not c0:
            msg = (
                f"Arg coefs must have a shape including {self.shapebs}"
            )
            raise Exception(msg)
        
        # shape or output    
        shape_pts, axis_rz, ind_coefs, ind_rz = [], [], [], []
        jj = 0
        for ii in range(coefs.ndim):
            if ii == axis[0]:
                for jj in range(r.ndim):
                    shape_pts.append(r.shape[jj])
                    axis_rz.append(ii + jj)
                    ind_rz.append(None)
                ind_coefs.append(None)
            elif ii == axis[1]:
                ind_coefs.append(None)
            else:
                shape_pts.append(coefs.shape[ii])
                ind_coefs.append(jj)
                ind_rz.append(jj)
                jj += 1
        
        # shape_other
        shape_other = tuple([
            ss for ii, ss in enumerate(coefs.shape)
            if ii not in axis
        ])
        
        return tuple(shape_pts), shape_other, axis_rz, ind_coefs, ind_rz

    def set_coefs(
        self,
        coefs=None,
        cropbs_neg_flat=None,
    ):

        if coefs.shape == self.shapebs:
            self.tck[2][...] = coefs.ravel()
        elif coefs.shape == (self.nbs,):
            self.tck[2][...] = coefs
        else:
            msg = f"Wrong coefs shape!\nProvided: {coefs.shape}"
            raise Exception(msg)

        # ------------
        # crop and set

        if cropbs_neg_flat is not None:
            self.tck[2][cropbs_neg_flat] = 0.

    def __call__(
        self,
        R=None,
        Z=None,
        coefs=None,
        axis=None,
        crop=None,
        cropbs=None,
        val_out=None,
        # for compatibility (unused)
        **kwdargs
    ):

        if val_out is None:
            val_out = np.nan

        # r, z have same shape
        r, z, crop = _check_RZ_crop(
            R=R,
            Z=Z,
            crop=crop,
            cropbs=cropbs,
        )

        # prepare
        # coefs
        shape_rz, shape_other, axis_rz, ind_coefs, ind_rz = self._check_coefs(
            coefs=coefs,
            axis=axis,
            r=r,
        )
        shape_coefs = coefs.shape
        
        val = np.zeros(shape_rz, dtype=float)
        cropbs_neg_flat = (~cropbs).ravel() if crop else None

        # interpolate
        for ind in itt.product(*[range(aa) for aa in shape_other]):
            
            # prepare TBF
            sli_c = tuple([
                slice(None) if ii in axis else ind[ind_coefs[ii]]
                for ii in range(len(shape_coefs))
            ])
            
            self.set_coefs(
                coefs=coefs[sli_c],
                cropbs_neg_flat=cropbs_neg_flat,
            )

            # compute
            sli_rz =tuple([
                slice(None) if ii in axis_rz else ind[ind_rz[ii]]
                for ii in range(len(shape_rz))
            ])
            
            val[sli_rz] = super().__call__(r, z, grid=False)

        # clean
        if val_out is not False:
            indout = (
                (r < self.tck[0][0]) | (r > self.tck[0][-1])
                | (z < self.tck[1][0]) | (z > self.tck[1][-1])
            )
            sli_out = tuple([
                indout if ii == axis[0] else slice(None)
                for ii in range(len(shape_coefs))
                if ii != axis[1]
            ])
            val[sli_out] = val_out
            
        return val

    def ev_details(
        self,
        R=None,
        Z=None,
        indbs_tf=None,
        crop=None,
        cropbs=None,
        # for compatibility (unused)
        **kwdargs
    ):
        """
        indbs_tf = (ar0, ar1)
            tuple of 2 flat arrays of int (for R and Z)
        """

        # -----------
        # check input

        x, y, crop = _check_RZ_crop(
            R=R,
            Z=Z,
            crop=crop,
            cropbs=cropbs,
        )

        # -----------
        # prepare

        deg = self.degrees[0]
        nbs = indbs_tf[0].size
        shape = x.shape
        x = np.ascontiguousarray(x.ravel(), dtype=float)
        y = np.ascontiguousarray(y.ravel(), dtype=float)
        coef = np.zeros((deg + 4, 1), dtype=float)
        coef[deg] = 1.
        outy = np.full((x.size, 1), np.nan)

        # -----------
        # compute

        val = np.zeros(tuple(np.r_[x.size, nbs]))
        indtot = np.arange(0, nbs)

        iz_u = np.unique(indbs_tf[1])

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

            indr = indbs_tf[1] == iz
            ir = indbs_tf[0][indr]
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

        if shape != x.shape:
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
        return _class1_bsplines_operators_rect.get_mesh2dRect_operators(
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
#                       Mesh2DRect - bsplines - overlap
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


def _check_RZ_crop(
    R=None,
    Z=None,
    crop=None,
    cropbs=None,
):

    # R, Z
    if not isinstance(R, np.ndarray):
        R = np.atleast_1d(R)
    if not isinstance(Z, np.ndarray):
        Z = np.atleast_1d(Z)
    assert R.shape == Z.shape

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

    return R, Z, crop


def get_bs2d_RZ(deg=None, Rknots=None, Zknots=None):

    # ----------------
    # get knots per bspline, nb of bsplines...

    knots_per_bs_R = _utils_bsplines._get_knots_per_bs(
        Rknots, deg=deg, returnas='data',
    )
    knots_per_bs_Z = _utils_bsplines._get_knots_per_bs(
        Zknots, deg=deg, returnas='data',
    )
    nbkbs = knots_per_bs_R.shape[0]
    shapebs = (knots_per_bs_R.shape[1], knots_per_bs_Z.shape[1])

    # ----------------
    # get centers of bsplines

    Rbs_apex = _utils_bsplines._get_apex_per_bs(
        knots=Rknots,
        knots_per_bs=knots_per_bs_R,
        deg=deg
    )
    Zbs_apex = _utils_bsplines._get_apex_per_bs(
        knots=Zknots,
        knots_per_bs=knots_per_bs_Z,
        deg=deg
    )
    return shapebs, Rbs_apex, Zbs_apex, knots_per_bs_R, knots_per_bs_Z


def get_bs2d_func(
    deg=None,
    Rknots=None,
    Zknots=None,
    shapebs=None,
    # knots_per_bs_R=None,
    # knots_per_bs_Z=None,
):

    # ----------------
    # Define functions

    clas = BivariateSplineRect(
        knotsR=Rknots,
        knotsZ=Zknots,
        deg=deg,
        shapebs=shapebs,
    )

    return clas.ev_details, clas.__call__, clas
