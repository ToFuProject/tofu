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

    if dmatch is not None:
        dout = {
            k0: {
                dout[k1][v0['ind']] for k1 in dout.keys()
            }
            for k0, v0 in dmatch.items()
        }

    # -------------
    # format output

    import pdb; pdb.set_trace()  # DB
    ilamb_min = np.full((lamb.shape[1],), -1)
    ilamb_max = np.full((lamb.shape[1],), -1)
    iok = np.any(np.isfinite(lamb), axis=0)
    ilamb_min[iok] = np.nanargmin(lamb[:, iok], axis=0)
    ilamb_max[iok] = np.nanargmax(lamb[:, iok], axis=0)

    lamb_min = np.array([
        lamb[imin, ii] if imin >= 0 else np.nan
        for ii, imin in enumerate(ilamb_min)
    ])
    lamb_max = np.array([
        lamb[imax, ii] if imax >= 0 else np.nan
        for ii, imax in enumerate(ilamb_max)
    ])

    dout = dict(din)
    dout.update({
        'key_crystals': key_crystals,
        'beta_max': beta_max,
        'crystx': crystx,
        'crysty': crysty,
        'endx': endx,
        'endy': endy,
        'lamb': lamb,
        'ilamb_min': ilamb_min,
        'ilamb_max': ilamb_max,
        'lamb_min': lamb_min,
        'lamb_max': lamb_max,
        'Dlamb': lamb_max - lamb_min,
    })

    if dcam is not None:
        dout['dcam'] = dcam

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
                **dout,
            )

    # ----------
    # save
    # ----------

    if save is True:
        np.savez(pfe_npz, **dout)
        msg = f"Saved in:\n\t{pfe_npz}"
        print(msg)

    # ---------
    # return

    if plot is True:
        return dout, dax
    else:
        return dout
