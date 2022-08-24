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
from . import _class8_plot as _plot


__all__ = ['Diagnostic']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Diagnostic(_class7_Camera.Camera):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
    #       'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'
    _dshow = dict(_class7_Camera.Camera._dshow)
    _dshow.update({
        'diagnostic': [
            'type',
            'optics',
            'spectro',
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
        c0 = (
            len(self.dobj['diagnostic'][key]['optics']) > 1
            and self.dobj['diagnostic'][key]['spectro'] is False
        )
        if c0:
            self.compute_diagnostic_etendue(
                key=key,
                analytical=True,
                numerical=False,
                res=None,
                check=False,
                verb=False,
                plot=False,
                store='analytical',
            )

        # --------------
        # adding los

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
        _compute._diag_compute_etendue(
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

    # ---------------
    # utilities
    # ---------------

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
        # figure
        dax=None,
        dmargin=None,
        fs=None,
        wintit=None,
        # interactivity
        connect=None,
    ):

        return _plot._plot_diagnostic(
            coll=self,
            key=key,
            optics=optics,
            elements=elements,
            proj=proj,
            # figure
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            # interactivity
            connect=connect,
        )
