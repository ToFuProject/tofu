# -*- coding: utf-8 -*-


import os


import numpy as np
import matplotlib.colors as mcolors
import datastock as ds


from . import _class02_save2stp


# #################################################################
# #################################################################
#          Default values
# #################################################################


_NAME = 'optics'
_COLOR = 'k'


# #################################################################
# #################################################################
#          Main
# #################################################################


def main(
    coll=None,
    # ---------------
    # input from tofu
    keys=None,
    # ---------------
    # options
    factor=None,
    color=None,
    chain=None,
    # ---------------
    # saving
    pfe=None,
    overwrite=None,
    verb=None,
):

    # ----------------
    # check inputs
    # --------------

    (
        keys,
        factor,
        color,
        chain,
        iso,
        pfe, overwrite, verb,
    ) = _check(
        coll=coll,
        keys=keys,
        # options
        factor=factor,
        color=color,
        chain=chain,
        # saving
        pfe=pfe,
        overwrite=overwrite,
        verb=verb,
    )

    fname = os.path.split(pfe)[-1][:-4]

    # ----------------
    # extract outlines
    # ----------------

    if chain is True:
        ptsx, ptsy, ptsz = [], [], []

    for kop in keys:
        px, py, pz = coll.get_optics_poly(
            key=kop,
            add_points=None,
            min_threshold=None,
            mode=None,
            closed=True,
            ravel=None,
            total=None,
            return_outline=None,
        )

        if chain is True:
            ptsx.append(np.append(px, np.nan))
            ptsy.append(np.append(py, np.nan))
            ptsz.append(np.append(pz, np.nan))

    dptsx = {fname: np.ravel(ptsx)}
    dptsy = {fname: np.ravel(ptsy)}
    dptsz = {fname: np.ravel(ptsz)}

    # scaling factor
    for k0 in dptsx.keys():
        dptsx[k0] = factor * dptsx[k0]
        dptsy[k0] = factor * dptsy[k0]
        dptsz[k0] = factor * dptsz[k0]

    # ---------------
    # get color dict
    # ---------------

    dcolor = _class02_save2stp._get_dcolor(dptsx=dptsx, color=color)

    # ----------------
    # get file content
    # ----------------

    # HEADER
    msg_header = _class02_save2stp._get_header(
        fname=fname,
        iso=iso,
    )

    # DATA
    msg_data = _class02_save2stp._get_data_polyline(
        dptsx=dptsx,
        dptsy=dptsy,
        dptsz=dptsz,
        fname=fname,
        # options
        dcolor=dcolor,
        color_by_pixel=False,
        # norm
        iso=iso,
    )

    # -------------
    # save to stp
    # -------------

    _class02_save2stp._save(
        msg=msg_header + "\n" + msg_data,
        pfe_save=pfe,
        overwrite=overwrite,
        verb=verb,
    )

    return


# #################################################################
# #################################################################
#          check
# #################################################################


def _check(
    coll=None,
    keys=None,
    # options
    factor=None,
    color=None,
    chain=None,
    # saving
    pfe=None,
    overwrite=None,
    verb=None,
):

    # --------------
    # keys
    # -------------

    if isinstance(keys, str):
        keys = [keys]

    lcls = ['aperture', 'filter', 'crystal', 'grating']
    lok = []
    for cl in lcls:
        lok += list(coll.dobj.get(cl, {}).keys())
    keys = ds._generic_check._check_var_iter(
        keys, 'keys',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
    )

    # ---------------
    # chain
    # ---------------

    chain = ds._generic_check._check_var(
        chain, 'chain',
        types=bool,
        default=True,
        allowed=[True],
    )

    # ---------------
    # factor
    # ---------------

    factor = float(ds._generic_check._check_var(
        factor, 'factor',
        types=(float, int),
        default=1.,
    ))

    # ---------------
    # color
    # ---------------

    if color is None:
        color = _COLOR
    if not mcolors.is_color_like(color):
        msg = (
            "Arg color must be color-like!\n"
            f"Provided: {color}\n"
        )
        raise Exception(msg)

    # ---------------
    # iso
    # ---------------

    iso = 'ISO-10303-21'

    # ---------------
    # pfe_save
    # ---------------

    # Default
    if pfe is None:
        path = os.path.abspath('.')
        pfe = os.path.join(path, f"{_NAME}.stp")

    # check
    c0 = (
        isinstance(pfe, str)
        and (
            os.path.split(pfe)[0] == ''
            or os.path.isdir(os.path.split(pfe)[0])
        )
    )
    if not c0:
        msg = (
            "Arg pfe must be a saving file str ending in '.stp'!\n"
            f"Provided: {pfe}\n"
        )
        raise Exception(msg)

    # makesure extension is included
    if not pfe.endswith('.stp'):
        pfe = f"{pfe}.stp"

    pfe = os.path.abspath(pfe)

    # ----------------
    # overwrite
    # ----------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    # ----------------
    # verb
    # ----------------

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    return (
        keys,
        factor,
        color,
        chain,
        iso,
        pfe, overwrite, verb,
    )
