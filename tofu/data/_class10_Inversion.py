# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class9_GeometryMatrix
from . import _inversions_comp
from . import _inversions_plot


__all__ = ['GeometryMatrix']


# #############################################################################
# #############################################################################
#                           Inversion
# #############################################################################


class GeometryMatrix(_class9_GeometryMatrix.GeometryMatrix):

    _show_in_summary = 'all'

    _dshow = dict(_class9_GeometryMatrix.GeometryMatrix._dshow)
    _dshow.update({
        'inversion': [
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

        return _inversions_comp.compute_inversions(
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

        return _matrix_comp.compute_retrofit_data(
            coll=self,
            key=key,
            key_matrix=key_matrix,
            key_profile2d=key_profile2d,
            t=t,
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

        return _inversions_plot.plot_inversion(
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
