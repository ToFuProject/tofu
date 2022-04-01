# -*- coding: utf-8 -*-


# Built-in
import warnings


# Common
import numpy as np
from matplotlib.tri import Triangulation as mplTri


# specific
from . import _generic_check


_ELEMENTS = 'knots'


# #############################################################################
# #############################################################################
#                           mesh generic check
# #############################################################################


def _mesh2D_check(
    coll=None,
    domain=None,
    res=None,
    R=None,
    Z=None,
    knots=None,
    cents=None,
    trifind=None,
    key=None,
):

    # key
    key = _generic_check._check_var(
        key, 'key',
        types=str,
        excluded=list(coll.dobj.get('mesh', {}).keys())
    )

    # rect of tri ?
    lc = [
        domain is not None or (R is not None and Z is not None),
        knots is not None and cents is not None,
    ]
    if all(lc) or not any(lc):
        msg = (
            "Either domain xor (R, Z) xor (knots, cents) must be provided!\n"
            "Provided:\n"
            f"\t- domain, res: {domain}, {res}\n"
            f"\t- type(R), type(Z): {type(R)}, {type(Z)}\n"
            f"\t- type(knots), type(cents): {type(knots)}, {type(cents)}\n"
        )
        raise Exception(msg)

    elif lc[0]:
        dref, dmesh = _mesh2DRect_to_dict(
            domain=domain,
            res=res,
            R=R,
            Z=Z,
            key=key,
        )
        ddata = None

    elif lc[1]:
        dref, ddata, dmesh = _mesh2DTri_to_dict(
            knots=knots,
            cents=cents,
            trifind=trifind,
            key=key,
        )
    return dref, ddata, dmesh


# #############################################################################
# #############################################################################
#                           Mesh2DTri
# #############################################################################


def _mesh2DTri_conformity(knots=None, cents=None, key=None):

    # ---------------------------------
    # make sure np.ndarrays of dim = 2

    knots = np.atleast_2d(knots).astype(float)
    cents = np.atleast_2d(cents).astype(int)

    # --------------
    # check shapes

    c0 = (
        knots.shape[1] == 2
        and knots.shape[0] >= 3
        and cents.shape[1] in [3, 4]
        and cents.shape[0] >= 1
        and cents.dtype == np.int
    )
    if not c0:
        msg = (
            "Arg knots must be of shape (nknots>=3, 2) and "
            "arg cents must be of shape (ncents>=1, 3 or 4) and dtype = int\n"
            "Provided:\n"
            f"\t- knots.shape: {knots.shape}\n"
            f"\t- cents.shape: {cents.shape}\n"
            f"\t- cents.dtype: {cents.dtype}\n"
        )
        raise Exception(msg)

    nknots = knots.shape[0]
    ncents = cents.shape[0]

    # -------------------
    # Test for duplicates

    # knots (floats => distance)
    dist = np.full((nknots, nknots), np.nan)
    ind = np.zeros(dist.shape, dtype=bool)
    for ii in range(nknots):
        dist[ii, ii+1:] = np.sqrt(
            (knots[ii+1:, 0] - knots[ii, 0])**2
            + (knots[ii+1:, 1] - knots[ii, 1])**2
        )
        ind[ii, ii+1:] = True

    ind[ind] = dist[ind] < 1.e-6
    if np.any(ind):
        iind = np.any(ind, axis=1).nonzero()[0]
        lstr = [f'\t\t- {ii}: {ind[ii, :].nonzero()[0]}' for ii in iind]
        msg = (
            f"Non-valid mesh {key}: \n"
            f"  Duplicate knots: {ind.sum()}\n"
            f"\t- knots.shape: {cents.shape}\n"
            f"\t- duplicate indices:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # cents (indices)
    centsu = np.unique(cents, axis=0)
    if centsu.shape[0] != ncents:
        msg = (
            f"Non-valid mesh {key}: \n"
            f"  Duplicate cents: {ncents - centsu.shape[0]}\n"
            f"\t- cents.shape: {cents.shape}\n"
            f"\t- unique shape: {centsu.shape}"
        )
        raise Exception(msg)

    # ---------------------
    # Test for unused knots

    centsu = np.unique(centsu)
    c0 = np.all(centsu >= 0) and centsu.size == nknots
    if centsu.size < nknots:
        ino = (~np.in1d(
            range(0, nknots),
            centsu,
            assume_unique=False,
            invert=False,
        )).nonzero()[0]
        msg = (
            f"Unused knots in {key}:\n"
            f"\t- unused knots indices: {ino}"
        )
        warnings.warn(msg)
    elif centsu.size > nknots or centsu.max() != nknots - 1:
        unknown = np.setdiff1d(centsu, range(nknots), assume_sorted=True)
        msg = (
            "Unknown knots refered to in cents!\n"
            f"\t- unknown knots: {unknown}"
        )
        raise Exception(msg)

    return cents, knots


def _mesh2DTri_clockwise(knots=None, cents=None, key=None):

    x, y = knots[cents, 0], knots[cents, 1]
    orient = (
        (y[:, 1] - y[:, 0])*(x[:, 2] - x[:, 1])
        - (y[:, 2] - y[:, 1])*(x[:, 1] - x[:, 0])
    )

    clock = orient > 0.
    if np.any(clock):
        msg = (
            "Some triangles not counter-clockwise\n"
            "  (necessary for matplotlib.tri.Triangulation)\n"
            f"    => {clock.sum()}/{cents.shape[0]} triangles reshaped"
        )
        warnings.warn(msg)
        cents[clock, 1], cents[clock, 2] = cents[clock, 2], cents[clock, 1]
    return cents


def _mesh2DTri_to_dict(knots=None, cents=None, key=None, trifind=None):

    # ---------------------
    # check mesh conformity

    cents, knots = _mesh2DTri_conformity(knots=knots, cents=cents, key=key)

    # ---------------------------------------------
    # define triangular mesh and trifinder function

    # triangular mesh
    if cents.shape[1] == 3:

        # check clock-wise triangles
        cents = _mesh2DTri_clockwise(knots=knots, cents=cents, key=key)

        # check trifinder
        if trifind is None:
            mpltri = mplTri(knots[:, 0], knots[:, 1], cents)
            trifind = mpltri.get_trifinder()

        meshtype = 'tri'

    # Quadrangular mesh => convert to triangular
    elif cents.shape[1] == 4:

        cents2 = np.empty((cents.shape[0]*2, 3), dtype=int)
        cents2[::2, :] = cents[:, :3]
        cents2[1::2, :-1] = cents[:, 2:]
        cents2[1::2, -1] = cents[:, 0]
        cents = cents2

        # Re-check mesh conformity
        cents, knots = _mesh2DTri_conformity(knots=knots, cents=cents, key=key)
        cents = _mesh2DTri_clockwise(knots=knots, cents=cents, key=key)

        # check trifinder
        if trifind is None:
            mpltri = mplTri(knots[:, 0], knots[:, 1], cents)
            trifind = mpltri.get_trifinder()

        meshtype = 'quadtri'

    # ----------------------------
    # Check on trifinder function

    assert callable(trifind), "Arg trifind must be a callable!"

    try:
        out = trifind(np.r_[0.], np.r_[0])
        assert isinstance(out, np.ndarray)
    except Exception as err:
        msg = (
            "Arg trifind must return an array of indices when fed with arrays "
            "of (R, Z) coordinates!\n"
            f"\ttrifind(np.r_[0], np.r_[0.]) = {out}"
        )
        raise Exception(msg)

    # -----------------
    # Format ouput dict

    kcents = f"{key}-cents"
    kcentsR = f"{kcents}-R"
    kcentsZ = f"{kcents}-Z"
    kcents_pts = f"{kcents}-pts"
    kcents_ind = f"{kcents}-ind"
    kknots = f"{key}-knots"
    kknotsR = f"{kknots}-R"
    kknotsZ = f"{kknots}-Z"
    kknots_ind = f"{kknots}-ind"

    # dref
    dref = {
        kknots_ind: {
            'data': np.arange(0, knots.shape[0]),
            'units': '',
            # 'source': None,
            'dim': '',
            'quant': 'ind',
            'name': 'ind',
            'group': 'ind',
        },
        kcents_ind: {
            'data': np.arange(0, cents.shape[0]),
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'Z',
            'name': 'Z',
            'group': 'Z',
        },
        kcents_pts: {
            'data': np.arange(0, 3),
            'units': '',
            # 'source': None,
            'dim': '',
            'quant': 'ind',
            'name': 'ind',
            'group': 'ind',
        },
    }

    # ddata
    ddata = {
        kknotsR: {
            'data': knots[:, 0],
            'ref': (kknots_ind,),
            'units': 'm',
            'quant': 'R',
            'dim': 'distance',
            'group': 'R',
        },
        kknotsZ: {
            'data': knots[:, 1],
            'ref': (kknots_ind,),
            'units': 'm',
            'quant': 'Z',
            'dim': 'distance',
            'group': 'Z',
        },
        kcentsR: {
            'data': np.mean(knots[cents, 0], axis=1),
            'ref': (kcents_ind,),
            'units': 'm',
            'quant': 'R',
            'dim': 'distance',
            'group': 'R',
        },
        kcentsZ: {
            'data': np.mean(knots[cents, 1], axis=1),
            'ref': (kcents_ind,),
            'units': 'm',
            'quant': 'Z',
            'dim': 'distance',
            'group': 'Z',
        },
        kcents: {
            'data': cents,
            'ref': (kcents_ind, kcents_pts),
            'units': '',
            'quant': 'ind',
            'dim': 'ind',
            'group': 'ind',
        },
    }

    # dobj
    dmesh = {
        key: {
            'type': meshtype,
            'cents': kcents,
            'knots': kknots,
            'ref': (kcents,),
            'shape': (cents.shape[0],),
            'trifind': trifind,
            'crop': False,
        },
    }
    return dref, ddata, dmesh


# #############################################################################
# #############################################################################
#                           Mesh2DRect
# #############################################################################


def _mesh2DRect_X_check(
    x=None,
    res=None,
):
    """ Returns knots (x) and associated resolution

    res can be:
        - int: numbr of mesh elements desired between knots
        - float: desired average mesh element size
        - array of floats: (one for each x, desired approximate mesh size)

    """

    # ------------
    # Check inputs

    # x
    try:
        x = np.unique(np.ravel(x).astype(float))
    except Exception as err:
        msg = "x must be convertible to a sorted, flat array of floats!"
        raise Exception(msg)

    # res
    if res is None:
        res = 10

    lc = [
        isinstance(res, (int, np.int64, np.int32)) and len(x) == 2,
        isinstance(res, (float, np.floating)) and len(x) == 2,
        isinstance(res, (list, tuple, np.ndarray)) and len(x) == len(res),
    ]
    if not any(lc):
        msg = (
            "Arg res must be:\n"
            "\t- int: nb of mesh elements along x\n"
            "\t       requires len(x) = 2\n"
            "\t- float: approximate desired mesh element size along x\n"
            "\t       requires len(x) = 2\n"
            "\t- iterable: approximate desired mesh element sizes along x\n"
            "\t       requires len(x) = len(res)\n"
        )
        raise Exception(msg)

    if lc[0]:
        x_new = np.linspace(x[0], x[1], int(res)+1)
        res_new = res
        indsep = None

    elif lc[1]:
        nb = int(np.ceil((x[1]-x[0]) / res))
        x_new = np.linspace(x[0], x[1], nb+1)
        res_new = np.mean(np.diff(x))
        indsep = None

    else:

        # check conformity
        res = np.ravel(res).astype(float)
        delta = np.diff(x)
        res_sum = res[:-1] + res[1:]
        ind = res_sum > delta + 1.e-14
        if np.any(ind):
            msg = (
                "Desired resolution is not achievable for the following:\n"
                f"res_sum: {res_sum[ind]}\n"
                f"delta  : {delta[ind]}"
            )
            raise Exception(msg)

        # compute nn
        # nn = how many pairs can fit in the interval
        npairs = np.round(delta/res_sum).astype(int)
        res_sum_new = delta / npairs

        fract = res[:-1] / res_sum

        res_new = [None for ii in range(len(x)-1)]
        x_new = [None for ii in range(len(x)-1)]
        for ii in range(len(x)-1):
            res_new[ii] = (
                res_sum_new[ii]
                * np.linspace(fract[ii], 1.-fract[ii], 2*npairs[ii])
            )
            if ii == 0:
                res_add = np.concatenate(([0], np.cumsum(res_new[ii])))
            else:
                res_add = np.cumsum(res_new[ii])
            x_new[ii] = x[ii] + res_add

        indsep = np.cumsum(npairs[:-1]*2)
        res_new = np.concatenate(res_new)
        x_new = np.concatenate(x_new)

    return x_new, res_new, indsep


def _mesh2DRect_check(
    domain=None,
    res=None,
    R=None,
    Z=None,
):

    # --------------
    # check inputs

    # (domain, res) vs (R, Z)
    lc = [
        domain is not None,
        R is not None and Z is not None,
    ]
    if all(lc) or not any(lc):
        msg = (
            "Please provide (domain, res) xor (R, Z), not both:\n"
            "Provided:\n"
            f"\t- domain, res: {domain}, {res}\n"
            f"\t- R, Z: {R}, {Z}\n"
        )
        raise Exception(msg)

    if lc[0]:
        # domain
        c0 = (
            isinstance(domain, list)
            and len(domain) == 2
            and all([
                hasattr(dd, '__iter__') and len(dd) >= 2 for dd in domain
            ])
        )
        if not c0:
            msg = (
                "Arg domain must be a list of 2 iterables of len() >= 2\n"
                f"Provided: {domain}"
            )
            raise Exception(msg)

        # res
        c0 = (
            res is None
            or np.isscalar(res)
            or isinstance(res, list) and len(res) == 2
        )
        if not c0:
            msg = (
                "Arg res must be a int, float or array or a list of 2 such\n"
                f"Provided: {res}"
            )
            raise Exception(msg)

        if np.isscalar(res) or res is None:
            res = [res, res]

        # -------------
        # check R and Z

        R, resR, indR = _mesh2DRect_X_check(domain[0], res=res[0])
        Z, resZ, indZ = _mesh2DRect_X_check(domain[1], res=res[1])

    elif lc[1]:

        # R, Z check
        c0 = (
            hasattr(R, '__iter__')
            and np.asarray(R).ndim == 1
            and np.unique(R).size == np.array(R).size
            and np.allclose(np.unique(R), R)
        )
        if not c0:
            msg = "Arg R must be convertible to a 1d increasing array"
            raise Exception(msg)

        c0 = (
            hasattr(Z, '__iter__')
            and np.asarray(Z).ndim == 1
            and np.unique(Z).size == np.array(Z).size
            and np.allclose(np.unique(Z), Z)
        )
        if not c0:
            msg = "Arg Z must be convertible to a 1d increasing array"
            raise Exception(msg)

        R = np.unique(R)
        Z = np.unique(Z)

        resR = np.diff(R)
        resZ = np.diff(Z)

        if np.unique(resR).size == 1:
            resR = resR[0]
            indR = None
        else:
            raise NotImplementedError()
        if np.unique(resZ).size == 1:
            resZ = resZ[0]
            indZ = None
        else:
            raise NotImplementedError()

    return R, Z, resR, resZ, indR, indZ


def _mesh2DRect_to_dict(
    domain=None,
    res=None,
    R=None,
    Z=None,
    key=None,
):

    # --------------------
    # check / format input

    kRknots, kZknots = f"{key}-R-knots", f"{key}-Z-knots"
    kRcent, kZcent = f"{key}-R-cents", f"{key}-Z-cents"

    R, Z, resR, resZ, indR, indZ = _mesh2DRect_check(
        domain=domain,
        res=res,
        R=R,
        Z=Z,
    )
    Rcent = 0.5*(R[1:] + R[:-1])
    Zcent = 0.5*(Z[1:] + Z[:-1])

    variable = not (np.isscalar(resR) and np.isscalar(resZ))

    # --------------------
    # prepare dict

    dref = {
        kRknots: {
            'data': R,
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'R',
            'name': 'R',
            'group': 'R',
        },
        kZknots: {
            'data': Z,
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'Z',
            'name': 'Z',
            'group': 'Z',
        },
        kRcent: {
            'data': Rcent,
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'R',
            'name': 'R',
            'group': 'R',
        },
        kZcent: {
            'data': Zcent,
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'Z',
            'name': 'Z',
            'group': 'Z',
        },
    }

    # dobj
    dmesh = {
        key: {
            'type': 'rect',
            'knots': (kRknots, kZknots),
            'cents': (kRcent, kZcent),
            'ref': (kRcent, kZcent),
            'shape': (Rcent.size, Zcent.size),
            'variable': variable,
            'crop': False,
        },
    }
    return dref, dmesh


def _mesh2DRect_from_croppoly(crop_poly=None, domain=None):

    # ------------
    # check inputs

    c0 = hasattr(crop_poly, '__iter__') and len(crop_poly) == 2
    lc = [
        crop_poly is None,
        (
            c0
            and isinstance(crop_poly, tuple)
            and crop_poly[0].__class__.__name__ == 'Config'
            and (isinstance(crop_poly[1], str) or crop_poly[1] is None)
        )
        or crop_poly.__class__.__name__ == 'Config',
        c0
        and all([
            hasattr(cc, '__iter__') and len(cc) == len(crop_poly[0])
            for cc in crop_poly[1:]
        ])
        and np.asarray(crop_poly).ndim == 2
    ]

    if not any(lc):
        msg = (
            "Arg config must be a Config instance!"
        )
        raise Exception(msg)

    # -------------
    # Get polyand domain

    if lc[0]:
        # trivial case
        poly = None

    else:

        # -------------
        # Get poly from input

        if lc[1]:
            # (config, structure name)

            if crop_poly.__class__.__name__ == 'Config':
                config = crop_poly
                key_struct = None
            else:
                config, key_struct = crop_poly

            # key_struct if None
            if key_struct is None:
                lk, ls = zip(*[
                    (ss.Id.Name, ss.dgeom['Surf']) for ss in config.lStructIn
                ])
                key_struct = lk[np.argmin(ls)]

            # poly
            poly = config.dStruct['dObj']['Ves'][key_struct].Poly_closed

        else:

            # make sure poly is np.ndarraya and closed
            poly = np.asarray(crop_poly).astype(float)
            if not np.allclose(poly[:, 0], poly[:, -1]):
                poly = np.concatenate((poly, poly[:, 0:1]))

        # -------------
        # Get domain from poly

        if domain is None:
            domain = [
                [poly[0, :].min(), poly[0, :].max()],
                [poly[1, :].min(), poly[1, :].max()],
            ]

    return domain, poly


# #############################################################################
# #############################################################################
#                           Mesh2DRect - select
# #############################################################################


def _select_ind_check(
    ind=None,
    elements=None,
    returnas=None,
    crop=None,
    meshtype=None,
):

    # ----------------------
    # check basic conditions

    if meshtype == 'rect':
        lc = [
            ind is None,
            isinstance(ind, tuple)
            and len(ind) == 2
            and (
                all([np.isscalar(ss) for ss in ind])
                or all([
                    hasattr(ss, '__iter__')
                    and len(ss) == len(ind[0])
                    for ss in ind
                ])
                or all([isinstance(ss, np.ndarray) for ss in ind])
            ),
            (
                np.isscalar(ind)
                or (
                    hasattr(ind, '__iter__')
                    and all([np.isscalar(ss) for ss in ind])
                )
                or isinstance(ind, np.ndarray)
            )
        ]
    else:
        lc = [
            ind is None,
            np.isscalar(ind)
            or (
                hasattr(ind, '__iter__')
                and all([np.isscalar(ss) for ss in ind])
            )
            or isinstance(ind, np.ndarray)
        ]

    if not any(lc):
        if meshtype == 'rect':
            msg = (
                "Arg ind must be either:\n"
                "\t- None\n"
                "\t- int or array of int: int indices in mixed (R, Z) index\n"
                "\t- tuple of such: int indices in (R, Z) index respectively\n"
                f"Provided: {ind}"
            )
        else:
            msg = (
                "Arg ind must be either:\n"
                "\t- None\n"
                "\t- int or array of int: int indices\n"
                "\t- array of bool: bool indices\n"
                f"Provided: {ind}"
            )
        raise Exception(msg)

    # ----------------------
    # adapt to each case

    if lc[0]:
        pass
    elif lc[1] and meshtype == 'rect':
        if any([not isinstance(ss, np.ndarray) for ss in ind]):
            ind = (
                np.atleast_1d(ind[0]).astype(int),
                np.atleast_1d(ind[1]).astype(int),
            )
        lc0 = [
            [
                isinstance(ss, np.ndarray),
                np.issubdtype(ss.dtype, np.integer),
                ss.shape == ind[0].shape,
            ]
                for ss in ind
        ]
        if not all([all(cc) for cc in lc0]):
            ltype = [type(ss) for ss in ind]
            ltypes = [
                ss.dtype if isinstance(ss, np.ndarray) else False
                for ss in ind
            ]
            lshapes = [
                ss.shape if isinstance(ss, np.ndarray) else len(ss)
                for ss in ind
            ]
            msg = (
                "Arg ind must be a tuple of 2 arrays of int of same shape\n"
                f"\t- lc0: {lc0}\n"
                f"\t- types: {ltype}\n"
                f"\t- type each: {ltypes}\n"
                f"\t- shape: {lshapes}\n"
                f"\t- ind: {ind}"
            )
            raise Exception(msg)
    elif lc[1] and meshtype in ['tri', 'quadtri']:
        if not isinstance(ind, np.ndarray):
            ind = np.atleast_1d(ind).astype(int)
        c0 = (
            np.issubdtype(ind.dtype, np.integer)
            or np.issubdtype(ind.dtype, np.bool_)
        )
        if not c0:
            msg = (
                "Arg ind must be an array of bool or int\n"
                f"Provided: {ind.dtype}"
            )
            raise Exception(msg)

    else:
        if not isinstance(ind, np.ndarray):
            ind = np.atleast_1d(ind).astype(int)
        c0 = (
            np.issubdtype(ind.dtype, np.integer)
            or np.issubdtype(ind.dtype, np.bool_)
        )
        if not c0:
            msg = (
                "Arg ind must be an array of bool or int\n"
                f"Provided: {ind.dtype}"
            )
            raise Exception(msg)

    # elements
    elements = _generic_check._check_var(
        elements, 'elements',
        types=str,
        default=_ELEMENTS,
        allowed=['knots', 'cents'],
    )

    # returnas
    if meshtype == 'rect':
        retdef = tuple
        retok = [tuple, np.ndarray, 'tuple-flat', 'array-flat', bool]
    else:
        retdef = bool
        retok = [int, bool]
    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=None,
        default=retdef,
        allowed=retok,
    )

    # crop
    crop = _generic_check._check_var(
        crop, 'crop',
        types=bool,
        default=True,
    )

    return ind, elements, returnas, crop


def _select_check(
    elements=None,
    returnas=None,
    return_ind_as=None,
    return_neighbours=None,
):

    # elements
    elements = _generic_check._check_var(
        elements, 'elements',
        types=str,
        default=_ELEMENTS,
        allowed=['knots', 'cents'],
    )

    # returnas
    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=None,
        default='ind',
        allowed=['ind', 'data'],
    )

    # return_ind_as
    return_ind_as = _generic_check._check_var(
        return_ind_as, 'return_ind_as',
        types=None,
        default=int,
        allowed=[int, bool],
    )

    # return_neighbours
    return_neighbours = _generic_check._check_var(
        return_neighbours, 'return_neighbours',
        types=bool,
        default=True,
    )

    return elements, returnas, return_ind_as, return_neighbours,


# #############################################################################
# #############################################################################
#                           Mesh2DRect - bsplines
# #############################################################################


def _mesh2D_bsplines(key=None, lkeys=None, deg=None):

    # key
    key = _generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lkeys,
    )

    # deg
    deg = _generic_check._check_var(
        deg, 'deg',
        types=int,
        default=2,
        allowed=[0, 1, 2, 3],
    )

    # keybs
    keybs = f'{key}-bs{deg}'

    return key, keybs, deg
