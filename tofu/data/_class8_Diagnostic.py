# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class7_Camera
from . import _class8_check as _check
from . import _class8_compute as _compute
from . import _class8_etendue_los as _etendue_los
from . import _class8_los_angles as _los_angles
from . import _class8_plot as _plot


__all__ = ['Diagnostic']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Diagnostic(_class7_Camera.Camera):

    _show_in_summary = 'all'

    _dshow = dict(_class7_Camera.Camera._dshow)
    _dshow.update({
        'diagnostic': [
            'type',
            'optics',
            'spectro',
            'ref',
            'etendue',
            'etend_type',
            'los',
            'spectrum',
            'time res.',
        ],
    })

    def add_diagnostic(
        self,
        key=None,
        optics=None,
        # etendue
        etendue=None,
        # config for los
        config=None,
        length=None,
        reflections_nb=None,
        reflections_type=None,
        # others
        **kwdargs,
    ):

        # -----------
        # adding diag

        # check / format input
        dref, ddata, dobj = _check._diagnostics(
            coll=self,
            key=key,
            optics=optics,
            **kwdargs,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # --------------
        # adding etendue

        key = list(dobj['diagnostic'].keys())[0]
        optics = self.dobj['diagnostic'][key]['optics']

        if len(optics) > 1:
            self.compute_diagnostic_etendue_los(
                key=key,
                analytical=True,
                numerical=False,
                res=None,
                check=False,
                # los
                config=config,
                length=length,
                reflections_nb=reflections_nb,
                reflections_type=reflections_type,
                # bool
                verb=False,
                plot=False,
                store='analytical',
            )

    # -----------------
    # utilities
    # -----------------

    def get_diagnostic_ref(self, key=None):
        return _check.get_ref(coll=self, key=key)

    # -----------------
    # etendue computing
    # -----------------

    def compute_diagnostic_etendue_los(
        self,
        key=None,
        # parameters
        analytical=None,
        numerical=None,
        res=None,
        check=None,
        # for storing los
        config=None,
        length=None,
        reflections_nb=None,
        reflections_type=None,
        # bool
        verb=None,
        plot=None,
        store=None,
    ):
        """ Compute the etendue of the diagnostic (per pixel)

        Etendue (m2.sr) can be computed analytically or numerically
        If plot, plot the comparison between all computations
        If store = 'analytical' or 'numerical', overwrites the diag etendue

        """
        dlos, store = _etendue_los.compute_etendue_los(
            coll=self,
            key=key,
            analytical=analytical,
            numerical=numerical,
            res=res,
            check=check,
            # bool
            verb=verb,
            plot=plot,
            store=store,
        )

        if store is not False and np.any(np.isfinite(dlos['los_x'])):
            _los_angles.compute_los_angles(
                coll=self,
                key=key,
                # los
                config=config,
                length=length,
                reflections_nb=reflections_nb,
                reflections_type=reflections_type,
                **dlos,
            )

    # ---------------
    # utilities
    # ---------------

    def get_diagnostic_optics(self, key=None):
        """ Get list of optics and list of corresponding classes """
        return _check._get_optics(coll=self, key=key)

    def get_optics_outline(
        self,
        key=None,
        add_points=None,
        closed=None,
        ravel=None,
    ):
        """ Return the optics outline """
        return _compute.get_optics_outline(
            coll=self,
            key=key,
            add_points=add_points,
            closed=closed,
            ravel=ravel,
        )

    def set_optics_color(self, key=None, color=None):
        return _check._set_optics_color(
            coll=self,
            key=key,
            color=color,
        )

    # -----------------
    # plotting
    # -----------------

    def get_diagnostic_dplot(
        self,
        key=None,
        optics=None,
        elements=None,
        vect_length=None,
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
                    'x0': ...,
                    'x1': ...,
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

        return _compute._dplot(
            coll=self,
            key=key,
            optics=optics,
            elements=elements,
            vect_length=vect_length,
        )

    def plot_diagnostic(
        self,
        key=None,
        optics=None,
        elements=None,
        proj=None,
        los_res=None,
        # data plot
        data=None,
        cmap=None,
        vmin=None,
        vmax=None,
        #config
        plot_config=None,
        # figure
        dax=None,
        dmargin=None,
        fs=None,
        wintit=None,
        # interactivity
        color_dict=None,
        nlos=None,
        dinc=None,
        connect=None,
    ):

        return _plot._plot_diagnostic(
            coll=self,
            key=key,
            optics=optics,
            elements=elements,
            proj=proj,
            los_res=los_res,
            # data plot
            data=data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            #config
            plot_config=plot_config,
            # figure
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            # interactivity
            color_dict=color_dict,
            nlos=nlos,
            dinc=dinc,
            connect=connect,
        )
