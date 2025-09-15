# -*- coding: utf-8 -*-


# tofu
from ._class08_Diagnostic import Diagnostic as Previous
from . import _class09_show as _show
from . import _class9_compute as _compute
from . import _class9_plot as _plot


__all__ = ['GeometryMatrix']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class GeometryMatrix(Previous):

    _which_gmat = 'geom_matrix'

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
        dvos=None,
        # common ref
        ref_com=None,
        ref_vector_strategy=None,
        # options
        brightness=None,
        # output
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
            dvos=dvos,
            # common ref
            ref_com=ref_com,
            ref_vector_strategy=ref_vector_strategy,
            # options
            brightness=brightness,
            # output
            verb=verb,
            store=store,
        )

    # -------------------
    # show
    # -------------------

    def _get_show_obj(self, which=None):
        if which in [self._which_gmat, self._which_gmat.replace('_', ' ')]:
            return _show._show
        else:
            return super()._get_show_obj(which)

    def _get_show_details(self, which=None):
        if which in [self._which_gmat, self._which_gmat.replace('_', ' ')]:
            return _show._show_details
        else:
            super()._get_show_details(which)

    # -----------------
    # get concatenated geometry matrix
    # ------------------

    def get_geometry_matrix_concatenated(
        self,
        key=None,
        key_cam=None,
    ):
        """ Assemble the geometry matrix """

        return _compute._concatenate(
            coll=self,
            key=key,
            key_cam=key_cam,
        )

    # -----------------
    # plotting
    # ------------------

    def plot_geometry_matrix(
        self,
        key=None,
        indbf=None,
        indchan=None,
        # options
        plot_mesh=None,
        plot_config=None,
        # parameters
        vmin=None,
        vmax=None,
        res=None,
        cmap=None,
        # figure
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
            # options
            plot_mesh=plot_mesh,
            plot_config=plot_config,
            # parameters
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
