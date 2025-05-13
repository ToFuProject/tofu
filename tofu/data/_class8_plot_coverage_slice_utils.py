

import numpy as np


# #######################################################
# #######################################################
#           Check plot args
# #######################################################


def _check_dvminmax(
    dvminmax=None,
    etend0=None,
    etend_plane0=None,
    etend=None,
    etend_plane=None,
    etend_lamb=None,
    etend_plane_lamb=None,
    sang0=None,
    sang=None,
    ndet=None,
):

    # -----------
    # preliminary
    # -----------

    if dvminmax is None:
        dvminmax = {}

    # ---------
    # cam_etend0

    if etend0 is not None:
        vmin = min(np.nanmin(etend0), np.nanmin(etend_plane0))
        vmax = max(np.nanmax(etend0), np.nanmax(etend_plane0)),

        dvminmax['cam_etend0'] = {
            'min': dvminmax.get('cam_etend0', {}).get('min', vmin),
            'max': dvminmax.get('cam_etend0', {}).get('max', vmax),
        }

    # ---------
    # cam_etend

    if etend is not None:
        vmin = min(np.nanmin(etend), np.nanmin(etend_plane))
        vmax = max(np.nanmax(etend), np.nanmax(etend_plane)),

        dvminmax['cam_etend'] = {
            'min': dvminmax.get('cam_etend', {}).get('min', vmin),
            'max': dvminmax.get('cam_etend', {}).get('max', vmax),
        }

    # ---------
    # cam_lamb

    if etend_lamb is not None:
        vmin = min(np.nanmin(etend_lamb), np.nanmin(etend_plane_lamb))
        vmax = min(np.nanmax(etend_lamb), np.nanmax(etend_plane_lamb))

        dvminmax['cam_lamb'] = {
            'min': dvminmax.get('cam_lamb', {}).get('min', vmin),
            'max': dvminmax.get('cam_lamb', {}).get('max', vmax),
        }

    # ---------
    # plane0

    if sang0 is not None:
        vmin = np.nanmin(sang0['data'])
        vmax = np.nanmax(sang0['data'])

        dvminmax['plane0'] = {
            'min': dvminmax.get('plane0', {}).get('min', vmin),
            'max': dvminmax.get('plane0', {}).get('max', vmax),
        }

    # ---------
    # plane

    if sang is not None:

        vmin = np.nanmin(sang['data'])
        vmax = np.nanmax(sang['data'])
        dvminmax['plane'] = {
            'min': dvminmax.get('plane', {}).get('min', vmin),
            'max': dvminmax.get('plane', {}).get('max', vmax),
        }

    # ---------
    # ndet

    if ndet is not None:

        vmin = max(np.nanmin(ndet['data']), 1)
        vmax = np.nanmax(ndet['data'])
        dvminmax['ndet'] = {
            'min': dvminmax.get('ndet', {}).get('min', vmin),
            'max': dvminmax.get('ndet', {}).get('max', vmax),
        }

    return dvminmax


# #######################################################
# #######################################################
#           markers
# #######################################################


def _add_marker(ax=None, indref=None, indplot=None):

    # ref pixel
    ax.plot(
        [indref[0]],
        [indref[1]],
        marker='s',
        markerfacecolor='None',
        markeredgecolor='k',
        ms=4,
    )

    # plot pixel
    ax.plot(
        [indplot[0]],
        [indplot[1]],
        marker='s',
        markerfacecolor='None',
        markeredgecolor='g',
        ms=4,
    )
