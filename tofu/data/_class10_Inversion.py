# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from ._class09_GeometryMatrix import GeometryMatrix as Previous
from . import _class10_compute as _compute
from . import _class10_plot as _plot


__all__ = ['Inversion']


# #############################################################################
# #############################################################################
#                           Inversion
# #############################################################################


class Inversion(Previous):

    _show_in_summary = 'all'

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'inversion': [
        ],
    })

    # -----------------
    # inversions
    # ------------------

    def add_inversion(
        self,
        # name of inversion
        key=None,
        # input data
        key_matrix=None,
        key_data=None,
        key_sigma=None,
        sigma=None,
        # choice of algo
        # isotropic=None,
        # sparse=None,
        # positive=None,
        # cholesky=None,
        # regparam_algo=None,
        algo=None,
        # regularity operator
        operator=None,
        geometry=None,
        # misc
        solver=None,
        conv_crit=None,
        chain=None,
        verb=None,
        store=None,
        # algo and solver-specific options
        kwdargs=None,
        method=None,
        options=None,
        # for polar mesh so far
        dconstraints=None,
    ):
        """ Compute tomographic inversion

        """

        return _compute.compute_inversions(
            # ressources
            coll=self,
            # name of inversion
            key=key,
            # input
            key_matrix=key_matrix,
            key_data=key_data,
            key_sigma=key_sigma,
            sigma=sigma,
            # choice of algo
            # isotropic=isotropic,
            # sparse=sparse,
            # positive=positive,
            # cholesky=cholesky,
            # regparam_algo=regparam_algo,
            algo=algo,
            # regularity operator
            operator=operator,
            geometry=geometry,
            # misc
            conv_crit=conv_crit,
            chain=chain,
            verb=verb,
            store=store,
            # algo and solver-specific options
            kwdargs=kwdargs,
            method=method,
            options=options,
            dconstraints=dconstraints,
        )

    # -----------------
    # synthetic data
    # -----------------

    def add_retrofit_data(
        self,
        key=None,
        key_diag=None,
        key_matrix=None,
        key_profile2d=None,
        t=None,
        store=None,
    ):
        """ Compute synthetic data using matching geometry matrix and profile2d

        Requires that a geometry matrix as been pre-computed
        Only profile2d with the same bsplines as the geometry matrix can be
        used

        """

        return _compute.compute_retrofit_data(
            coll=self,
            key=key,
            key_diag=key_diag,
            key_matrix=key_matrix,
            key_profile2d=key_profile2d,
            t=t,
            store=store,
        )

    # -----------------
    # plotting
    # ------------------

    def plot_inversion(
        self,
        key=None,
        indt=None,
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

        return _plot.plot_inversion(
            coll=self,
            key=key,
            indt=indt,
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
