# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class8_Diagnostic
from . import _matrix_comp
from . import _matrix_plot


__all__ = ['GeometryMatrix']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class GeometryMatrix(_class8_Diagnostic.Diagnostic):

    _show_in_summary = 'all'

    _dshow = dict(_class8_Diagnostic.Diagnostic._dshow)
    _dshow.update({
        'geom. matrix': [
        ],
    })

    # -----------------
    # geometry matrix
    # ------------------

    def add_geometry_matrix(
        self,
        key=None,
        key_chan=None,
        key_diag=None,
        key_cam=None,
        # sampling
        res=None,
        resMode=None,
        method=None,
        crop=None,
        name=None,
        verb=None,
        store=None,
    ):

        return _matrix_comp.compute(
            coll=self,
            key=key,
            key_chan=key_chan,
            cam=cam,
            res=res,
            resMode=resMode,
            method=method,
            crop=crop,
            name=name,
            verb=verb,
            store=store,
        )

    # -----------------
    # plotting
    # ------------------

    def plot_geometry_matrix(
        self,
        cam=None,
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
        return _matrix_plot.plot_geometry_matrix(
            cam=cam,
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
