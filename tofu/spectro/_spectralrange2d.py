# -*- coding: utf-8 -*-


import numpy as np


from ._spectralrange2d_check import main as _check
from ._spectralrange2d_compute import main as _compute
from . import _spectralrange2d_plot as _plot


# #################################################################
# #################################################################
#               Main
# #################################################################


def main(
    # optics
    dap=None,
    dcrystals=None,
    dcam=None,
    # matching
    dmatch=None,
    # large scans
    dscans=None,
    # geometry basis
    beta_max=None,
    # options
    npts=None,
    # plotting
    plot=None,
    dax=None,
    # saving
    save=None,
    pfe_fig=None,
    pfe_npz=None,
):

    """

    lamb0: target wavelength
    bragg0: target bragg angle
    rcurve: radii of curvature

    ap: point source position
    xx: distance between point source and crystals
    dist: lenght of rays after reflexion
    beta_max: maximum angular opening from point source (optionnal)
    npts: nb of rays from point source to crystals
    length: crystal length
    varrad: for variable-radii spiral

    """

    # -------------
    # check inputs
    # -------------

    (
        dap, dcrystals, dcam, dmatch,
        dscans, npts,
        plot, save, pfe_fig, pfe_npz,
    ) = _check(**locals())

    # --------------
    # compute
    # --------------

    dout = _compute(npts=npts, **dscans)

    # --------------
    # extract
    # --------------

    dout_match = None
    if dmatch is not None:
        dout_match = _dout_match(dout=dout, dmatch=dmatch)

    # ---------
    # plot
    # ---------

    if plot is True:
        if dmatch is None:
            dax = _plot.scans(
                dax=dax,
                pfe_fig=pfe_fig,
                **dout,
            )
        else:
            dax = _plot.match(
                dax=dax,
                pfe_fig=pfe_fig,
                dout=dout,
                dmatch=dmatch,
                dap=dap,
                dcam=dcam,
                dscans=dscans,
            )

    # ----------
    # save
    # ----------

    dout0 = dout if dmatch is None else dout_match

    if save is True:
        np.savez(pfe_npz, **dout0)
        msg = f"Saved in:\n\t{pfe_npz}"
        print(msg)

    # ---------
    # return

    if plot is True:
        return dout0, dax
    else:
        return dout0


# ############################################
# ############################################
#           Extract dmatch
# ############################################


def _dout_match(dout=None, dmatch=None):

    dout_match = {k0: {k1: {} for k1 in dout.keys()} for k0 in dmatch.keys()}
    for k0, v0 in dmatch.items():
        for k1, v1 in dout.items():

            # array
            if isinstance(v1, np.ndarray):
                if v1.ndim == dout['cryst0'].ndim:
                    sli = (slice(None),) + v0['ind']
                else:
                    sli = v0['ind']
                dout_match[k0][k1] = v1[sli]

            # dict
            else:
                for k2, v2 in v1.items():
                    if v2.ndim == dout['cryst0'].ndim:
                        sli = (slice(None),) + v0['ind']
                    else:
                        sli = v0['ind']
                    dout_match[k0][k1][k2] = v2[sli]

    return dout_match
