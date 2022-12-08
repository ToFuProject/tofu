# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from ._class08_Diagnostic import Diagnostic as Previous
from . import _class9_compute as _compute
from . import _class9_plot as _plot


__all__ = ['GeometryMatrix']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class GeometryMatrix(Previous):

    _show_in_summary = 'all'

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'geom matrix': [
            'bsplines',
            'diagnostic',
            'camera',
            'shape',
            'data',
            'method',
            'res',
        ],
    })

    # -----------------
    # geometry matrix
    # ------------------

    def add_geometry_matrix(
        self,
        key=None,
        key_bsplines=None,
        key_diag=None,
        key_cam=None,
        # sampling
        res=None,
        mode=None,
        method=None,
        crop=None,
        verb=None,
        store=None,
    ):

        return _compute.compute(
            coll=self,
            key=key,
            key_bsplines=key_bsplines,
            key_diag=key_diag,
            key_cam=key_cam,
            # sampling
            res=res,
            mode=mode,
            method=method,
            crop=crop,
            verb=verb,
            store=store,
        )

    # -----------------
    # get concatenated geometry matrix
    # ------------------

    def get_geometry_matrix_concatenated(
        self,
        key=None,
    ):
        """ Assemble the geometry matrix """

        return _compute._concatenate(
            coll=self,
            key=key,
        )

    # -----------------
    # plotting
    # ------------------

    def plot_geometry_matrix(
        self,
        key=None,
        indbf=None,
        indchan=None,
        plot_mesh=None,
        vmin=None,
        vmax=None,
        res=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
    ):
        return _plot.plot_geometry_matrix(
            coll=self,
            key=key,
            indbf=indbf,
            indchan=indchan,
            plot_mesh=plot_mesh,
            vmin=vmin,
            vmax=vmax,
            res=res,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )
