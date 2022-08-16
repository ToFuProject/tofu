# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class1_Rays
from . import _class2_check
from . import _class2_compute


__all__ = ['Diagnostic']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Diagnostic(_class1_Rays.Rays):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
    #       'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'
    _dshow = {
        'aperture': [
            'planar', 'area',
            'outline', 'poly',
            'cent', 'dgeom.area2'
        ],
        'camera': [
            'type', 'parallel',
            'shape', 'ref',
            'pix area', 'pix nb',
            'outline',
            'cent', 'cents',
            'qeff_energy',
            'qeff',
            'model',
        ],
        'diagnostic': [
            'type',
            'optics',
            'etendue',
            'los',
            'spectrum',
            'time res.',
        ],
    }

    def add_aperture(
        self,
        key=None,
        # 2d outline
        outline_x0=None,
        outline_x1=None,
        cent=None,
        # 3d outline
        poly_x=None,
        poly_y=None,
        poly_z=None,
        # normal vector
        nin=None,
        e0=None,
        e1=None,
    ):
        """ Add an aperture

        Can be defined from:
            - 2d outline + 3d center + unit vectors (nin, e0, e1)
            - 3d polygon + nin

        Unit vectors will be checked and normalized
        If planar, area will be computed
        Outline will be made counter-clockwise

        """

        # check / format input
        dref, ddata, dobj = _class2_check._aperture(
            coll=self,
            key=key,
            # 2d outline
            outline_x0=outline_x0,
            outline_x1=outline_x1,
            cent=cent,
            # 3d outline
            poly_x=poly_x,
            poly_y=poly_y,
            poly_z=poly_z,
            # normal vector
            nin=nin,
            e0=e0,
            e1=e1,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def add_camera_1d(
        self,
        key=None,
        # common 2d outline
        outline_x0=None,
        outline_x1=None,
        # centers of all pixels
        cents_x=None,
        cents_y=None,
        cents_z=None,
        # inwards normal vectors
        nin_x=None,
        nin_y=None,
        nin_z=None,
        # orthonormal direct base
        e0_x=None,
        e0_y=None,
        e0_z=None,
        e1_x=None,
        e1_y=None,
        e1_z=None,
        # quantum efficiency
        lamb=None,
        energy=None,
        qeff=None,
    ):
        # check / format input
        dref, ddata, dobj = _class2_check._camera_1d(
            coll=self,
            key=key,
            # common 2d outline
            outline_x0=outline_x0,
            outline_x1=outline_x1,
            # centers of all pixels
            cents_x=cents_x,
            cents_y=cents_y,
            cents_z=cents_z,
            # inwards normal vectors
            nin_x=nin_x,
            nin_y=nin_y,
            nin_z=nin_z,
            # orthonormal direct base
            e0_x=e0_x,
            e0_y=e0_y,
            e0_z=e0_z,
            e1_x=e1_x,
            e1_y=e1_y,
            e1_z=e1_z,
            # quantum efficiency
            lamb=lamb,
            energy=energy,
            qeff=qeff,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def add_camera_2d(
        self,
        key=None,
        # common 2d outline
        outline_x0=None,
        outline_x1=None,
        # centers of all pixels
        cent=None,
        cents_x0=None,
        cents_x1=None,
        # inwards normal vectors
        nin=None,
        e0=None,
        e1=None,
        # quantum efficiency
        lamb=None,
        energy=None,
        qeff=None,
    ):
        # check / format input
        dref, ddata, dobj = _class2_check._camera_2d(
            coll=self,
            key=key,
            # common 2d outline
            outline_x0=outline_x0,
            outline_x1=outline_x1,
            # centers of all pixels
            cent=cent,
            cents_x0=cents_x0,
            cents_x1=cents_x1,
            # inwards normal vectors
            nin=nin,
            e0=e0,
            e1=e1,
            # quantum efficiency
            lamb=lamb,
            energy=energy,
            qeff=qeff,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def add_diagnostic(
        self,
        key=None,
        optics=None,
        **kwdargs,
    ):
        # check / format input
        dref, ddata, dobj = _class2_check._diagnostics(
            coll=self,
            key=key,
            optics=optics,
            **kwdargs,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    # ---------------
    # utilities
    # ---------------

    def get_camera_unit_vectors(self, key=None):
        """ Return unit vectors components as dict """
        return _class2_check.get_camera_unitvectors(
            coll=self,
            key=key,
        )

    def get_camera_cents_xyz(self, key=None):
        """ Return cents_x, cents_y, cents_z """
        return _class2_check.get_camera_cents_xyz(
            coll=self,
            key=key,
        )

    def get_optics_outline(self, key=None, add_points=None, closed=None):
        """ Return the optics outline """
        return _class2_check.get_optics_outline(
            coll=self,
            key=key,
            add_points=None,
            closed=None,
        )

    def get_as_dict(self, which=None, key=None):
        """ Return the desired object as a dict (input to some routines) """

        return _class2_check._return_as_dict(
            coll=self,
            which=which,
            key=key,
        )

    # -----------------
    # etendue computing
    # -----------------

    def compute_diagnostic_etendue(
        self,
        key=None,
        analytical=None,
        numerical=None,
        res=None,
        check=None,
        verb=None,
        plot=None,
        store=None,
    ):
        """ Compute the etendue of the diagnostic (per pixel)

        Etendue (m2.sr) can be computed analytically or numerically
        If plot, plot the comparison between all computations
        If store = 'analytical' or 'numerical', overwrites the diag etendue

        """
        _class2_compute._diag_compute_etendue(
            coll=self,
            key=key,
            analytical=analytical,
            numerical=numerical,
            res=res,
            check=check,
            verb=verb,
            plot=plot,
            store=store,
        )

    # -----------------
    # plotting
    # -----------------

    def get_diagnostic_dplot(
        self,
        key=None,
        optics=None,
        elements=None,
    ):
        """ Return a dict with all that's necessary for plotting

        If no optics is provided, all are returned

        elements indicate, for each optics, what should be represented:
            - 'o': outline
            - 'v': unit vectors
            - 's': summit ( = center for non-curved)
            - 'c': center (of curvature)
            - 'r': rowland circle / axis of cylinder

        returned as a dict:

        dplot = {
            'optics0': {
                'o': {
                    'x': ...,
                    'y': ...,
                    'z': ...,
                    'r': ...,
                },
                'v': {
                    'x': ...,
                    'y': ...,
                    'z': ...,
                    'r': ...,
                },
            },
        }

        """

        return _class2_compute._dplot(
            coll=self,
            key=key,
            optics=optics,
            element=element,
        )

    def plot_diagnostic(
        self,
        key=None,
        optics=None,
        element=None,
        # figure
        dax=None,
        dmargin=None,
        fs=None,
        # interactivity
        connect=None,
    ):

        return _class2_plot._plot_diagnostic(
            coll=self,
            key=key,
            optics=optics,
            element=element,
            # figure
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            # interactivity
            connect=connect,
        )
