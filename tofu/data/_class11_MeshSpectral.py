# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import datastock as ds


# tofu
# from tofu import __version__ as __version__
from ._class10_Inversion import Inversion as Previous
from . import _class1_checks
from . import _class1_compute
from . import _class11_checks as _checks
from . import _class11_compute as _compute
from . import _class11_plot as _plot


__all__ = ['MeshSpectral']


# #############################################################################
# #############################################################################
#                           Plasma2D
# #############################################################################


class MeshSpectral(Previous):

    _which_msp = 'mesh_spectral'
    _which_bssp = 'bsplines_spectral'    

    _ddef = copy.deepcopy(Previous._ddef)
    _ddef['params']['ddata'].update({
        _which_bssp: {'cls': str, 'def': ''},
    })
    
    def add_mesh_spectral(
        self,
        # mesh
        key=None,
        E=None,
        # direct addition of bsplines
        deg=None,
        **kwdargs,
    ):
        """ Add an spectral mesh by key

        The mesh is defined by a strictly increasing vector of knots / edges.
        The knots E are provided in energy (eV).

        If deg is provided, immediately adds a bsplines

        Example:
        --------
                >>> import tofu as tf
                >>> conf = tf.load_config('ITER')
                >>> mesh = tf.data.Plasma2D()
                >>> mesh.add_mesh(config=conf, res=0.1, deg=1)

        """

        # check input data and get input dicts
        dref, ddata, dmesh = _checks._mesh1D_check(
            coll=self,
            # mesh knots
            E=E,
            # key
            key=key,
        )

        # add kwdargs
        key = list(dmesh.keys())[0]
        dmesh[key].update(**kwdargs)

        # define dobj['mesh']
        dobj = {
            self._which_msp: dmesh,
        }

        # update data source
        for k0, v0 in ddata.items():
            ddata[k0]['source'] = kwdargs.get('source')

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # optional bspline
        if deg is not None:
            self.add_bsplines_spectral(key=key, deg=deg)

    # -----------------
    # bsplines
    # ------------------

    def add_bsplines_spectral(self, key=None, deg=None):
        """ Add bspline basis functions on the chosen spectral mesh """

        # --------------
        # check inputs

        keym, keybs, deg = _class1_checks._mesh_bsplines(
            key=key,
            lkeys=list(self.dobj[self._which_msp].keys()),
            deg=deg,
        )

        # --------------
        # get bsplines

        dref, ddata, dobj = _class1_compute._mesh1d_bsplines(
            coll=self,
            keym=keym,
            keybs=keybs,
            deg=deg,
            which_mesh=self._which_msp,
            which_bsplines=self._which_bssp,
        )

        # --------------
        # update dict and crop if relevant

        self.update(dobj=dobj, ddata=ddata, dref=dref)

    # -----------------
    # add_data
    # ------------------

    def update(
        self,
        dobj=None,
        ddata=None,
        dref=None,
        harmonize=None,
    ):
        """ Overload datastock update() method """

        # if ddata => check ref for bsplines
        if ddata is not None:
            for k0, v0 in ddata.items():
                (
                    ddata[k0]['ref'], ddata[k0]['data'],
                ) = _class1_checks.add_data_meshbsplines_ref(
                    coll=self,
                    ref=v0['ref'],
                    data=v0['data'],
                    which_mesh=self._which_msp,
                    which_bsplines=self._which_bssp,
                )

        # update
        super().update(
            dobj=dobj,
            ddata=ddata,
            dref=dref,
            harmonize=harmonize,
        )

        # assign bsplines
        if self._dobj.get(self._which_bssp) is not None:
            for k0, v0 in self._ddata.items():
                lbs = [
                    k1 for k1, v1 in self._dobj[self._which_bssp].items()
                    if v1['ref'] == tuple([
                        rr for rr in v0['ref']
                        if rr in v1['ref']
                    ])
                ]
                if len(lbs) == 0:
                    pass
                elif len(lbs) == 1:
                    self._ddata[k0][self._which_bssp] = lbs[0]
                else:
                    msg = f"Multiple nsplines spectral:\n{lbs}"
                    raise Exception(msg)

    # -----------------
    # interp tools
    # ------------------
    
    def get_sample_mesh_spectral(
        self,
        key=None,
        res=None,
        mode=None,
        DE=None,
    ):
        """ Return a sampled version of the chosen mesh """
        return _class1_compute.sample_mesh_1d(
            coll=self,
            key=key,
            res=res,
            mode=mode,
            Dx=DE,
            which_mesh=self._which_msp,
        )

    def interpolate_spectral(
        self,
        key=None,
        E=None,
        Ebins=None,
        res=None,
        mode=None,
        DE=None,
    ):
        """  Return pectrally interpolated coeeficients for each E value """
        
        return _compute.interpolate_spectral(
            coll=self,
            key=key,
            E=E,
            Ebins=Ebins,
            res=res,
            mode=mode,
            DE=DE,
        )

    # -----------------
    # plotting
    # -----------------

    def plot_mesh_spectral(
        self,
        key=None,
        ind_knot=None,
        ind_cent=None,
        units=None,
        color=None,
        dax=None,
        dmargin=None,
        fs=None,
        dleg=None,
        connect=None,
    ):

        return _plot.plot_mesh_spectral(
            coll=self,
            key=key,
            ind_knot=ind_knot,
            ind_cent=ind_cent,
            units=units,
            color=color,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dleg=dleg,
            connect=connect,
        )

    def plot_bsplines_spectral(
        self,
        key=None,
        indbs=None,
        indt=None,
        knots=None,
        cents=None,
        res=None,
        plot_mesh=None,
        val_out=None,
        nan0=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dleg=None,
    ):

        return _plot.plot_bspline_spectral(
            coll=self,
            key=key,
            indbs=indbs,
            indt=indt,
            knots=knots,
            cents=cents,
            res=res,
            plot_mesh=plot_mesh,
            val_out=val_out,
            nan0=nan0,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dleg=dleg,
        )

    def plot_spectrum(
        self,
        # inputs
        key=None,
        coefs=None,
        indt=None,
        res=None,
        # plot options
        vmin=None,
        vmax=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
        # interactivity
        dinc=None,
        connect=None,
    ):
        return _plot.plot_spectrum(
            coll=self,
            # inputs
            key=key,
            coefs=coefs,
            indt=indt,
            res=res,
            # plot options
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            # interactivity
            dinc=dinc,
            connect=connect,
        )

    def plot_profile2d_spectral(
        self,
        key=None,
    ):
        """ plot a spectral-dependent 2d emissivity field """
        
        return _plot.plot_profile2d_spectral()
